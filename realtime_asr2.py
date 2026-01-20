import asyncio
import aiohttp
import json
import struct
import gzip
import uuid
import logging
import os
import subprocess
import sounddevice as sd
import webrtcvad
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, AsyncGenerator

# =========================
# 配置日志：文件记录INFO；控制台只显示ERROR
# =========================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 文件日志（保留完整日志）
file_handler = logging.FileHandler('run.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# 控制台日志（只输出错误，避免刷屏）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# 避免重复添加 handler
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# 常量定义
DEFAULT_SAMPLE_RATE = 16000
VAD_AGGRESSIVENESS = 3  # 语音检测灵敏度 0-3，3最灵敏
SILENCE_THRESHOLD = 1  # 静音阈值(秒)，超过此时长认为说话结束
RECORDING_CHUNK_DURATION = 30  # 录音块时长(毫秒)
CHUNK_SIZE = int(DEFAULT_SAMPLE_RATE * RECORDING_CHUNK_DURATION / 1000)  # 每个音频块的样本数


class ProtocolVersion:
    V1 = 0b0001


class MessageType:
    CLIENT_FULL_REQUEST = 0b0001
    CLIENT_AUDIO_ONLY_REQUEST = 0b0010
    SERVER_FULL_RESPONSE = 0b1001
    SERVER_ERROR_RESPONSE = 0b1111


class MessageTypeSpecificFlags:
    NO_SEQUENCE = 0b0000
    POS_SEQUENCE = 0b0001
    NEG_SEQUENCE = 0b0010
    NEG_WITH_SEQUENCE = 0b0011


class SerializationType:
    NO_SERIALIZATION = 0b0000
    JSON = 0b0001


class CompressionType:
    GZIP = 0b0001


class Config:
    def __init__(self):
        # 填入控制台获取的app id和access token
        self.auth = {
            "app_key": "2704273799",
            "access_key": "r-50x_Sojl9QmFyvhFny7ZAWFx_Zs1Be"
        }

    @property
    def app_key(self) -> str:
        return self.auth["app_key"]

    @property
    def access_key(self) -> str:
        return self.auth["access_key"]


config = Config()


class CommonUtils:
    @staticmethod
    def gzip_compress(data: bytes) -> bytes:
        return gzip.compress(data)

    @staticmethod
    def gzip_decompress(data: bytes) -> bytes:
        return gzip.decompress(data)

    @staticmethod
    def judge_wav(data: bytes) -> bool:
        if len(data) < 44:
            return False
        return data[:4] == b'RIFF' and data[8:12] == b'WAVE'

    @staticmethod
    def convert_wav_with_path(audio_path: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
        try:
            cmd = [
                "ffmpeg", "-v", "quiet", "-y", "-i", audio_path,
                "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate),
                "-f", "wav", "-"
            ]
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # 尝试删除原始文件
            try:
                os.remove(audio_path)
            except OSError as e:
                logger.warning(f"Failed to remove original file: {e}")

            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
            raise RuntimeError(f"Audio conversion failed: {e.stderr.decode()}")

    @staticmethod
    def read_wav_info(data: bytes) -> Tuple[int, int, int, int, bytes]:
        if len(data) < 44:
            raise ValueError("Invalid WAV file: too short")

        # 解析WAV头
        chunk_id = data[:4]
        if chunk_id != b'RIFF':
            raise ValueError("Invalid WAV file: not RIFF format")

        format_ = data[8:12]
        if format_ != b'WAVE':
            raise ValueError("Invalid WAV file: not WAVE format")

        # 解析fmt子块
        audio_format = struct.unpack('<H', data[20:22])[0]
        num_channels = struct.unpack('<H', data[22:24])[0]
        sample_rate = struct.unpack('<I', data[24:28])[0]
        bits_per_sample = struct.unpack('<H', data[34:36])[0]

        # 查找data子块
        pos = 36
        while pos < len(data) - 8:
            subchunk_id = data[pos:pos + 4]
            subchunk_size = struct.unpack('<I', data[pos + 4:pos + 8])[0]
            if subchunk_id == b'data':
                wave_data = data[pos + 8:pos + 8 + subchunk_size]
                return (
                    num_channels,
                    bits_per_sample // 8,
                    sample_rate,
                    subchunk_size // (num_channels * (bits_per_sample // 8)),
                    wave_data
                )
            pos += 8 + subchunk_size

        raise ValueError("Invalid WAV file: no data subchunk found")

    @staticmethod
    def pcm_to_wav(pcm_data: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
        """将PCM数据转换为WAV格式"""
        n_channels = 1
        sample_width = 2  # 16位
        byte_rate = sample_rate * n_channels * sample_width
        block_align = n_channels * sample_width
        bits_per_sample = sample_width * 8

        # WAV文件头
        header = b'RIFF'
        header += struct.pack('<I', 36 + len(pcm_data))  # 文件大小
        header += b'WAVEfmt '
        header += struct.pack('<I', 16)  # 子块大小
        header += struct.pack('<H', 1)  # 音频格式 (PCM)
        header += struct.pack('<H', n_channels)
        header += struct.pack('<I', sample_rate)
        header += struct.pack('<I', byte_rate)
        header += struct.pack('<H', block_align)
        header += struct.pack('<H', bits_per_sample)
        header += b'data'
        header += struct.pack('<I', len(pcm_data))  # 数据大小

        return header + pcm_data


class AudioRecorder:
    """音频录制和语音活动检测类"""

    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.is_recording = False
        self.audio_buffer = b""
        self.silence_counter = 0
        self.total_silence = 0.0
        self.recording_started = False

    def start(self):
        """开始录音"""
        self.is_recording = True
        self.audio_buffer = b""
        self.silence_counter = 0
        self.total_silence = 0.0
        self.recording_started = False
        print("开始监听语音...")

    def stop(self):
        """停止录音"""
        self.is_recording = False
        recorded_data = self.audio_buffer
        self.audio_buffer = b""
        return recorded_data

    def process_chunk(self, indata: np.ndarray) -> Tuple[bool, bool]:
        """
        处理音频块并检测语音活动
        返回: (是否正在说话, 是否检测到语音结束)
        """
        if not self.is_recording:
            return False, False

        # 将numpy数组转换为16位PCM
        pcm_data = indata.astype(np.int16).tobytes()

        # 检查是否有语音活动
        is_speech = self.vad.is_speech(pcm_data, DEFAULT_SAMPLE_RATE)

        if is_speech:
            self.recording_started = True
            self.audio_buffer += pcm_data
            self.total_silence = 0.0  # 重置静音计时器
            return True, False
        else:
            if self.recording_started:
                # 已经开始录音，累计静音时间
                self.total_silence += RECORDING_CHUNK_DURATION / 1000.0

                # 如果静音时间超过阈值，认为说话结束
                if self.total_silence >= SILENCE_THRESHOLD:
                    self.is_recording = False
                    return False, True
            return False, False

    async def record_until_silence(self) -> bytes:
        """录制音频直到检测到足够长的静音"""
        self.start()

        # 使用异步方式录制音频
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Recording status: {status}")

            is_speaking, is_complete = self.process_chunk(indata)

            if is_complete:
                recorded_data = self.stop()
                loop.call_soon_threadsafe(future.set_result, recorded_data)
                raise sd.CallbackStop()

        # 开始录音
        stream = sd.InputStream(
            samplerate=DEFAULT_SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            channels=1,
            dtype='int16',
            callback=callback
        )

        with stream:
            return await future


class AsrRequestHeader:
    def __init__(self):
        self.message_type = MessageType.CLIENT_FULL_REQUEST
        self.message_type_specific_flags = MessageTypeSpecificFlags.POS_SEQUENCE
        self.serialization_type = SerializationType.JSON
        self.compression_type = CompressionType.GZIP
        self.reserved_data = bytes([0x00])

    def with_message_type(self, message_type: int) -> 'AsrRequestHeader':
        self.message_type = message_type
        return self

    def with_message_type_specific_flags(self, flags: int) -> 'AsrRequestHeader':
        self.message_type_specific_flags = flags
        return self

    def with_serialization_type(self, serialization_type: int) -> 'AsrRequestHeader':
        self.serialization_type = serialization_type
        return self

    def with_compression_type(self, compression_type: int) -> 'AsrRequestHeader':
        self.compression_type = compression_type
        return self

    def with_reserved_data(self, reserved_data: bytes) -> 'AsrRequestHeader':
        self.reserved_data = reserved_data
        return self

    def to_bytes(self) -> bytes:
        header = bytearray()
        header.append((ProtocolVersion.V1 << 4) | 1)
        header.append((self.message_type << 4) | self.message_type_specific_flags)
        header.append((self.serialization_type << 4) | self.compression_type)
        header.extend(self.reserved_data)
        return bytes(header)

    @staticmethod
    def default_header() -> 'AsrRequestHeader':
        return AsrRequestHeader()


class RequestBuilder:
    @staticmethod
    def new_auth_headers() -> Dict[str, str]:
        reqid = str(uuid.uuid4())
        return {
            "X-Api-Resource-Id": "volc.bigasr.sauc.duration",
            "X-Api-Request-Id": reqid,
            "X-Api-Access-Key": config.access_key,
            "X-Api-App-Key": config.app_key
        }

    @staticmethod
    def new_full_client_request(seq: int) -> bytes:  # 添加seq参数
        header = AsrRequestHeader.default_header() \
            .with_message_type_specific_flags(MessageTypeSpecificFlags.POS_SEQUENCE)

        payload = {
            "user": {
                "uid": "demo_uid"
            },
            "audio": {
                "format": "wav",
                "codec": "raw",
                "rate": 16000,
                "bits": 16,
                "channel": 1
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": True,
                "enable_ddc": True,
                "show_utterances": True,
                "enable_nonstream": False
            }
        }

        payload_bytes = json.dumps(payload).encode('utf-8')
        compressed_payload = CommonUtils.gzip_compress(payload_bytes)
        payload_size = len(compressed_payload)

        request = bytearray()
        request.extend(header.to_bytes())
        request.extend(struct.pack('>i', seq))  # 使用传入的seq
        request.extend(struct.pack('>I', payload_size))
        request.extend(compressed_payload)

        return bytes(request)

    @staticmethod
    def new_audio_only_request(seq: int, segment: bytes, is_last: bool = False) -> bytes:
        header = AsrRequestHeader.default_header()
        if is_last:  # 最后一个包特殊处理
            header.with_message_type_specific_flags(MessageTypeSpecificFlags.NEG_WITH_SEQUENCE)
            seq = -seq  # 设为负值
        else:
            header.with_message_type_specific_flags(MessageTypeSpecificFlags.POS_SEQUENCE)
        header.with_message_type(MessageType.CLIENT_AUDIO_ONLY_REQUEST)

        request = bytearray()
        request.extend(header.to_bytes())
        request.extend(struct.pack('>i', seq))

        compressed_segment = CommonUtils.gzip_compress(segment)
        request.extend(struct.pack('>I', len(compressed_segment)))
        request.extend(compressed_segment)

        return bytes(request)


class AsrResponse:
    def __init__(self):
        self.code = 0
        self.event = 0
        self.is_last_package = False
        self.payload_sequence = 0
        self.payload_size = 0
        self.payload_msg = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "event": self.event,
            "is_last_package": self.is_last_package,
            "payload_sequence": self.payload_sequence,
            "payload_size": self.payload_size,
            "payload_msg": self.payload_msg
        }


class ResponseParser:
    @staticmethod
    def parse_response(msg: bytes) -> AsrResponse:
        response = AsrResponse()

        header_size = msg[0] & 0x0f
        message_type = msg[1] >> 4
        message_type_specific_flags = msg[1] & 0x0f
        serialization_method = msg[2] >> 4
        message_compression = msg[2] & 0x0f

        payload = msg[header_size * 4:]

        # 解析message_type_specific_flags
        if message_type_specific_flags & 0x01:
            response.payload_sequence = struct.unpack('>i', payload[:4])[0]
            payload = payload[4:]
        if message_type_specific_flags & 0x02:
            response.is_last_package = True
        if message_type_specific_flags & 0x04:
            response.event = struct.unpack('>i', payload[:4])[0]
            payload = payload[4:]

        # 解析message_type
        if message_type == MessageType.SERVER_FULL_RESPONSE:
            response.payload_size = struct.unpack('>I', payload[:4])[0]
            payload = payload[4:]
        elif message_type == MessageType.SERVER_ERROR_RESPONSE:
            response.code = struct.unpack('>i', payload[:4])[0]
            response.payload_size = struct.unpack('>I', payload[4:8])[0]
            payload = payload[8:]

        if not payload:
            return response

        # 解压缩
        if message_compression == CompressionType.GZIP:
            try:
                payload = CommonUtils.gzip_decompress(payload)
            except Exception as e:
                logger.error(f"Failed to decompress payload: {e}")
                return response

        # 解析payload
        try:
            if serialization_method == SerializationType.JSON:
                response.payload_msg = json.loads(payload.decode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to parse payload: {e}")

        return response


class AsrWsClient:
    def __init__(self, url: str, segment_duration: int = 200):
        self.seq = 1
        self.url = url
        self.segment_duration = segment_duration
        self.conn = None
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.conn and not self.conn.closed:
            await self.conn.close()
        if self.session and not self.session.closed:
            await self.session.close()

    async def read_audio_data(self, file_path: str) -> bytes:
        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            if not CommonUtils.judge_wav(content):
                logger.info("Converting audio to WAV format...")
                content = CommonUtils.convert_wav_with_path(file_path, DEFAULT_SAMPLE_RATE)

            return content
        except Exception as e:
            logger.error(f"Failed to read audio data: {e}")
            raise

    def get_segment_size(self, content: bytes) -> int:
        try:
            channel_num, samp_width, frame_rate, _, _ = CommonUtils.read_wav_info(content)[:5]
            size_per_sec = channel_num * samp_width * frame_rate
            segment_size = size_per_sec * self.segment_duration // 1000
            return segment_size
        except Exception as e:
            logger.error(f"Failed to calculate segment size: {e}")
            raise

    async def create_connection(self) -> None:
        headers = RequestBuilder.new_auth_headers()
        try:
            self.conn = await self.session.ws_connect(  # 使用self.session
                self.url,
                headers=headers
            )
            logger.info(f"Connected to {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise

    async def send_full_client_request(self) -> None:
        request = RequestBuilder.new_full_client_request(self.seq)
        self.seq += 1  # 发送后递增
        try:
            await self.conn.send_bytes(request)
            logger.info(f"Sent full client request with seq: {self.seq - 1}")

            msg = await self.conn.receive()
            if msg.type == aiohttp.WSMsgType.BINARY:
                response = ResponseParser.parse_response(msg.data)
                logger.info(f"Received response: {response.to_dict()}")
            else:
                logger.error(f"Unexpected message type: {msg.type}")
        except Exception as e:
            logger.error(f"Failed to send full client request: {e}")
            raise

    async def send_messages(self, segment_size: int, content: bytes) -> AsyncGenerator[None, None]:
        audio_segments = self.split_audio(content, segment_size)
        total_segments = len(audio_segments)

        for i, segment in enumerate(audio_segments):
            is_last = (i == total_segments - 1)
            request = RequestBuilder.new_audio_only_request(
                self.seq,
                segment,
                is_last=is_last
            )
            await self.conn.send_bytes(request)
            logger.info(f"Sent audio segment with seq: {self.seq} (last: {is_last})")

            if not is_last:
                self.seq += 1

            await asyncio.sleep(self.segment_duration / 1000)  # 逐个发送，间隔时间模拟实时流
            # 让出控制权，允许接受消息
            yield

    async def recv_messages(self) -> AsyncGenerator[AsrResponse, None]:
        try:
            async for msg in self.conn:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    response = ResponseParser.parse_response(msg.data)

                    # 只在最后一个包输出最终识别结果
                    if response.code == 0 and response.is_last_package:
                        if response.payload_msg:
                            recognized_text = response.payload_msg.get("result", {}).get("text", "")
                            if recognized_text:
                                print(f"\n识别结果: {recognized_text}")

                    yield response

                    if response.is_last_package or response.code != 0:
                        break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg.data}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("WebSocket connection closed")
                    break
        except Exception as e:
            logger.error(f"Error receiving messages: {e}")
            raise

    async def start_audio_stream(self, segment_size: int, content: bytes) -> AsyncGenerator[AsrResponse, None]:
        async def sender():
            async for _ in self.send_messages(segment_size, content):
                pass

        # 启动发送和接收任务
        sender_task = asyncio.create_task(sender())

        try:
            async for response in self.recv_messages():
                yield response
        finally:
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass

    @staticmethod
    def split_audio(data: bytes, segment_size: int) -> List[bytes]:
        if segment_size <= 0:
            return []

        segments = []
        for i in range(0, len(data), segment_size):
            end = i + segment_size
            if end > len(data):
                end = len(data)
            segments.append(data[i:end])
        return segments

    async def recognize_audio(self, audio_data: bytes) -> AsyncGenerator[AsrResponse, None]:
        """识别给定的音频数据"""
        if not self.url:
            raise ValueError("URL is empty")

        self.seq = 1

        try:
            # 转换为WAV格式（如果需要）
            if not CommonUtils.judge_wav(audio_data):
                audio_data = CommonUtils.pcm_to_wav(audio_data)

            # 计算分段大小
            segment_size = self.get_segment_size(audio_data)

            # 创建WebSocket连接
            await self.create_connection()

            # 发送完整客户端请求
            await self.send_full_client_request()

            # 启动音频流处理
            async for response in self.start_audio_stream(segment_size, audio_data):
                yield response

        except Exception as e:
            logger.error(f"Error in ASR execution: {e}")
            raise
        finally:
            if self.conn:
                await self.conn.close()

    async def execute(self, file_path: str) -> AsyncGenerator[AsrResponse, None]:
        """处理文件的原有方法"""
        if not file_path:
            raise ValueError("File path is empty")

        content = await self.read_audio_data(file_path)
        async for response in self.recognize_audio(content):
            yield response


async def realtime_recognition_demo():
    """实时语音识别演示"""
    url = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_nostream"
    seg_duration = 200
    recorder = AudioRecorder()

    print("实时语音识别已启动，开始说话吧（停止说话1秒后将自动识别）...")
    print("按Ctrl+C退出程序")

    try:
        while True:
            # 录制音频直到检测到足够长的静音
            pcm_data = await recorder.record_until_silence()

            if pcm_data:
                print("检测到说话结束，正在识别...")

                # 进行语音识别
                async with AsrWsClient(url, seg_duration) as client:
                    async for _ in client.recognize_audio(pcm_data):
                        pass

                print("\n准备好接收下一段语音...")
    except KeyboardInterrupt:
        print("\n程序已退出")
    except Exception as e:
        logger.error(f"实时识别出错: {e}")


async def main():
    # 切换到实时识别模式
    await realtime_recognition_demo()

    # 如果需要使用文件识别，可使用下面的代码
    """
    file_path = "D:/python_project/语音助手/豆包/语音识别/sauc_python (3)/whoareyou.wav"
    url = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_nostream"
    seg_duration = 200

    async with AsrWsClient(url, seg_duration) as client:
        try:
            async for _ in client.execute(file_path):
                pass
        except Exception as e:
            logger.error(f"ASR processing failed: {e}")
    """


if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 安装必要的库
    try:
        import sounddevice as sd
        import webrtcvad
    except ImportError:
        print("正在安装必要的库...")
        subprocess.run(["pip", "install", "sounddevice", "webrtcvad", "numpy"])
        import sounddevice as sd
        import webrtcvad

    asyncio.run(main())