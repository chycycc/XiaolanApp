# -*- coding: utf-8 -*-
"""
火山引擎 TTS 语音合成客户端

功能:
    - HTTP方式调用火山TTS API
    - 支持流式播放 (边下载边播放)
    - 支持长文本智能分句
    - 支持停止播放控制
    - 连接池优化网络性能

核心函数:
    - play_long_text(): 播放长文本语音
    - tts_synthesize(): 合成并保存为MP3文件
    - stop_playback(): 停止当前播放
    - close_client(): 关闭HTTP客户端

API文档: https://www.volcengine.com/docs/6561/79823

Author: XiaolanApp Team
"""

import asyncio
import base64
import gzip
import json
import logging
import uuid
import time
import os
import re
import io

import httpx

from config import VOLCANO_TTS

logger = logging.getLogger(__name__)

class VolcTTS:
    def __init__(self):
        self.app_id = VOLCANO_TTS["app_id"]
        self.access_token = VOLCANO_TTS["access_token"]
        # HTTP 接口地址
        self.endpoint = "https://openspeech.bytedance.com/api/v1/tts"
        self.cluster = "volcano_tts" 
        self.voice_type = VOLCANO_TTS["voice_type"]
        # 复用 HTTP 客户端（单例）
        self._client: httpx.AsyncClient = None

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端（单例复用）"""
        if self._client is None or self._client.is_closed:
            # verify=False 跳过SSL验证，显著减少握手时间(内网/代理环境下)
            self._client = httpx.AsyncClient(timeout=10.0, verify=False)
        return self._client

    async def close(self):
        """关闭 HTTP 客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        logger.info("[Volc TTS] HTTP客户端已关闭")

    async def _synthesize_raw(self, text: str) -> bytes:
        """核心合成逻辑：返回音频二进制数据，不播放不保存"""
        headers = {
            "Authorization": f"Bearer; {self.access_token}",
            "Content-Type": "application/json"
        }
        
        request_json = {
            "app": {
                "appid": self.app_id,
                "token": self.access_token,
                "cluster": self.cluster
            },
            "user": {"uid": "python_client"},
            "audio": {
                "voice_type": self.voice_type,
                "encoding": "mp3",
                "rate": 24000,
                "speed": 12,
                "pitch": 10,
                "volume": 0
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "text_type": "plain",
                "operation": "query"
            }
        }

        try:
            client = await self._get_client()
            response = await client.post(self.endpoint, json=request_json, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"[Volc TTS] HTTP Error: {response.status_code}")
                return None
            
            resp_json = response.json()
            if "data" in resp_json:
                audio_base64 = resp_json["data"]
                if audio_base64:
                    return base64.b64decode(audio_base64)
            
            logger.error(f"[Volc TTS] 合成失败: {resp_json.get('message', 'Unknown')}")
            return None

        except Exception as e:
            logger.error(f"[Volc TTS] Connection Error: {e}")
            return None

    async def synthesize(self, text: str, output_file: str = "reply.mp3") -> str:
        """
        合成语音并保存为文件 (HTTP POST)
        """
        if output_file == "reply.mp3":
             # 主动清理所有的 reply_*.mp3
             self._cleanup_old_files()
             output_file = f"reply_{uuid.uuid4().hex[:8]}.mp3"
        
        audio_data = await self._synthesize_raw(text)
        if audio_data:
            with open(output_file, "wb") as f:
                f.write(audio_data)
            
            logger.info(f"[Volc TTS] 合成完成: {output_file}")
            self._schedule_cleanup(output_file)
            return output_file
        return None

    async def synthesize_and_play(self, text: str, perf_stats: dict = None) -> bool:
        """
        合成语音并直接播放
        """
        # 开始新的一句播放前，清除停止标志
        global _stop_event
        _stop_event.clear()

        audio_data = await self._synthesize_raw(text)
        
        if audio_data:
            logger.info(f"[Volc TTS] 合成完成，直接播放...")
            
            if perf_stats:
                t_play = time.time()
                t_asr = perf_stats.get('t_asr', 0)
                t_llm = perf_stats.get('t_llm', 0)
                
                llm_cost = t_llm - t_asr
                tts_cost = t_play - t_llm
                total_cost = t_play - t_asr
                
                logger.info(f"\n"
                            f"⚡ [性能详情]\n"
                            f"  - 大模型思考 : {llm_cost:.3f}s\n"
                            f"  - TTS合成播放: {tts_cost:.3f}s\n"
                            f"  ------------------------\n"
                            f"  = 总首字延迟 : {total_cost:.3f}s (从识别结束到发声)")
            
            # 尝试内存播放
            play_audio_bytes(audio_data)
            return True
        return False

    async def play_long_text(self, text: str, perf_stats: dict = None):
        """
        长文本优化模式：智能动态分句
        策略：首句短（追求响应速度），后续长（追求流畅自然）
        """
        # 开始新的一句播放前，清除停止标志
        global _stop_event
        _stop_event.clear()

        chunks = []
        buffer = ""
        # 4级渐进式缓冲区 (Fibonacci Ramp-up)
        # 策略: 6字(秒开) -> 6字(快速衔接) -> 15字(稳定缓冲) -> 50字(长句模式)
        # 调整: 第二包也设为6字，防止第二包下载慢导致断流
        thresholds = [6, 6, 15] 
        max_threshold = 50
        current_threshold = thresholds.pop(0)
        
        # 标点定义
        strong_delimiters = "。！？；!?;:"
        weak_delimiters = "，、,（）()"
        
        for char in text:
            buffer += char
            
            # 1. 遇到强标点（句号等），强制切分
            if char in strong_delimiters:
                chunks.append(buffer)
                buffer = ""
                # 升级阈值
                current_threshold = thresholds.pop(0) if thresholds else max_threshold
                
            # 2. 遇到弱标点（逗号），检查缓冲区是否足够长
            elif char in weak_delimiters:
                if len(buffer) >= current_threshold:
                    chunks.append(buffer)
                    buffer = ""
                    # 升级阈值
                    current_threshold = thresholds.pop(0) if thresholds else max_threshold
            
            # 3. 强制切分
            elif len(buffer) >= current_threshold + 25: 
                 chunks.append(buffer)
                 buffer = ""
                 # 升级阈值
                 current_threshold = thresholds.pop(0) if thresholds else max_threshold 
        
        # 3. 处理剩余文本
        if buffer:
            chunks.append(buffer)
            
        # 过滤空白
        chunks = [c for c in chunks if c.strip()]
            
        # 如果只有一句，直接走普通模式
        if len(chunks) <= 1:
            return await self.synthesize_and_play(text, perf_stats)

        logger.info(f"[Volc TTS] 智能分句模式: {len(chunks)} 句 (首句优化+长文连贯)")
        
        # 重新实现流水线逻辑：
        # 创建一个下载任务队列
        queue = asyncio.Queue()
        
        # 生产者：负责下载
        async def producer():
            for idx, c in enumerate(chunks):
                # 检查全局停止标志
                if _stop_event.is_set():
                    logger.info("[Volc TTS] 检测到停止标志，终止后续分片下载")
                    break

                # 过滤掉只有标点符号或空白的块，防止 API 报错 400
                if not re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', c):
                    continue
                
                # 第一句需要特殊处理（打印首字延迟）
                if idx == 0:
                    data = await self._synthesize_raw(c)
                    if data and perf_stats:
                         t_play = time.time()
                         t_asr = perf_stats.get('t_asr', 0)
                         t_llm = perf_stats.get('t_llm', 0)
                         llm = t_llm - t_asr
                         tts = t_play - t_llm
                         total = t_play - t_asr
                         logger.info(f"\n⚡ [流式首句] LLM:{llm:.2f}s | TTS:{tts:.2f}s | Total:{total:.2f}s")
                    await queue.put(data)
                else:
                    data = await self._synthesize_raw(c)
                    await queue.put(data)
                    
            await queue.put(None) # 结束标记
            
        # 启动生产者（后台下载）
        prod_task = asyncio.create_task(producer())
        
        # 消费者：负责播放（当前线程）
        while True:
            data = await queue.get()
            if data is None:
                break
            if data:
                # 在线程池中播放，以免阻塞事件循环导致生产者无法下载
                await asyncio.to_thread(play_audio_bytes, data)
        
        await prod_task
        return True

    def _schedule_cleanup(self, filename):
        """延迟删除临时文件"""
        def cleanup():
            time.sleep(10) # 保留10秒供播放
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except:
                pass
        import threading
        threading.Thread(target=cleanup, daemon=True).start()

    def _cleanup_old_files(self):
        """清理旧的 reply_*.mp3 文件"""
        try:
            current_dir = os.getcwd()
            for filename in os.listdir(current_dir):
                if filename.startswith("reply_") and filename.endswith(".mp3"):
                    try:
                        # 尝试删除，如果被占用则忽略
                        os.remove(filename)
                    except Exception:
                        pass
        except Exception:
            pass

# 全局实例
volc_tts = VolcTTS()

async def tts_synthesize(text: str, output_file: str = "reply.mp3") -> str:
    """兼容接口"""
    return await volc_tts.synthesize(text, output_file)

async def tts_synthesize_and_play(text: str, perf_stats: dict = None) -> bool:
    """合成并直接播放，自动判断长文本模式"""
    if len(text) > 12: # 智能判定
        return await volc_tts.play_long_text(text, perf_stats)
    else:
        return await volc_tts.synthesize_and_play(text, perf_stats)

def play_wav(filename: str):
    """播放音频文件"""
    if not filename or not os.path.exists(filename):
        return

    try:
        from playsound import playsound
        abs_path = os.path.abspath(filename)
        playsound(abs_path)
    except Exception as e:
        logger.error(f"[Volc TTS] 播放失败: {e}")


import threading

# 全局停止标志
_stop_event = threading.Event()

def stop_playback():
    """立即停止当前播放"""
    _stop_event.set()
    try:
        import sounddevice as sd
        sd.stop()
    except Exception:
        pass

def play_audio_bytes(audio_data: bytes):
    """从内存直接播放音频数据（跳过文件IO）"""
    try:
        import sounddevice as sd
        import io
        import numpy as np # Fixed: Import numpy
        from pydub import AudioSegment
        
        # 1. 优先查找项目目录下的 ffmpeg/bin
        project_ffmpeg = os.path.join(os.getcwd(), "ffmpeg", "bin")
        ffmpeg_exe = os.path.join(project_ffmpeg, "ffmpeg.exe")
        ffprobe_exe = os.path.join(project_ffmpeg, "ffprobe.exe")
        
        if os.path.exists(ffmpeg_exe) and os.path.exists(ffprobe_exe):
            # 将项目内的 ffmpeg 目录加入 PATH，这样 pydub 就能找到 ffprobe
            os.environ["PATH"] += os.pathsep + project_ffmpeg
            AudioSegment.converter = ffmpeg_exe
        else:
            # 2. 回退到系统路径
            sys_ffmpeg = r"c:\Windows\System32\ffmpeg.exe"
            if os.path.exists(sys_ffmpeg):
               AudioSegment.converter = sys_ffmpeg
        
        # 使用 pydub 解码 MP3
        try:
            # 1. pydub 加载
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            
            # 2. 转 numpy
            samples = np.array(audio_segment.get_array_of_samples())
            
            # 3. 归一化 (int16 -> float32)
            if audio_segment.sample_width == 2:
                samples = samples.astype(np.float32) / 32768.0
            
            # 4. 声道处理
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2))
                
            # 5. 播放 (非阻塞模式，但在本函数内我们会用 wait 等待播放完)
            #    为了支持打断，我们需要自己控制等待循环
            # sd.play(samples, samplerate=audio_segment.frame_rate)
            # sd.wait()
            
            stream = sd.OutputStream(
                samplerate=audio_segment.frame_rate,
                channels=audio_segment.channels,
                dtype='float32'
            )
            with stream:
                # 分块写入 stream，以便中间可以 check stop_event
                chunk_size = 1024
                total_len = len(samples)
                current_pos = 0
                
                # 如果之前被置位了，先清除，允许新的一轮播放
                # (注意：如果是全局“停止”状态，调用者应该负责重置，或者在这里做策略)
                # 这里简单处理：每次 play 认为是一个新的开始，
                # 如果希望“即使 play 被调用也不播”，需要在调用前判断。
                # 但 stop_playback 语意往往是“打断当前”，所以这里我们在 check 之后清除标志不太合适
                # 应该由外部控制 reset。但在简单的打断场景下，可以在 play 开始时 reset？
                # 不，stop 可能是异步来的。
                
                # 使用 callback 方式或者 block write 方式
                while current_pos < total_len:
                    if _stop_event.is_set():
                        break
                    
                    # 写入一块
                    end_pos = min(current_pos + chunk_size, total_len)
                    chunk = samples[current_pos:end_pos]
                    stream.write(chunk)
                    current_pos = end_pos
                
        except Exception as e:
            logger.warning(f"[Volc TTS] pydub内存播放失败: {e}")
            raise e # 抛出异常以触发回退逻辑
            
    except Exception as e:
        logger.warning(f"[Volc TTS] 内存播放不可用 ({e})，回退到文件播放")
        
        # 回退到临时文件播放
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            play_wav(temp_path)
            try:
                os.remove(temp_path)
            except:
                pass
        except Exception as file_e:
            logger.error(f"[Volc TTS] 文件播放也失败: {file_e}")

# 导出全局清理函数
def volc_tts_close():
    """手动关闭 Volc TTS 客户端连接"""
    try:
        # 由于我们是在 threading 中调用 asyncio.run，可能无法直接 await
        # 但如果是 session 结束，其实这里只需要置空 client 即可
        # 如果能 await 最好，不能的话，直接丢弃引用
        # 这里为了安全，尝试运行
        loop = asyncio.new_event_loop()
        loop.run_until_complete(volc_tts.close())
        loop.close()
    except Exception as e:
        volc_tts._client = None
        logger.warning(f"[Volc TTS] 强制重置客户端: {e}")
