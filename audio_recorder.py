# -*- coding: utf-8 -*-
"""
音频录制模块
提供麦克风录音和语音活动检测(VAD)功能
"""

import asyncio
import collections
import logging
import sounddevice as sd
import webrtcvad
import numpy as np
from typing import AsyncGenerator

# =========================
# 配置日志
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
VAD_AGGRESSIVENESS = 1  # 语音检测灵敏度 0-3，1为中等
SILENCE_THRESHOLD = 0.6  # 静音阈值(秒)
RECORDING_CHUNK_DURATION = 30  # 录音块时长(毫秒)
CHUNK_SIZE = int(DEFAULT_SAMPLE_RATE * RECORDING_CHUNK_DURATION / 1000)


class AudioRecorder:
    """
    音频录制器：使用 sounddevice 捕获麦克风音频，使用 webrtcvad 检测语音活动
    """
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.is_recording = False
        self.audio_buffer = b""
        self.recording_started = False
        self.silence_counter = 0
        self.total_silence = 0.0
        
        # 预录音缓冲：保存最近300ms的音频（约10帧）
        self.pre_buffer = collections.deque(maxlen=10)
        
        # 常驻流对象
        self.stream = None
        self._on_chunk_callback = None

    def start(self):
        """开始录音"""
        self.is_recording = True
        self.audio_buffer = b""
        self.silence_counter = 0
        self.total_silence = 0.0
        self.recording_started = False
        self.pre_buffer.clear()

    def stop(self):
        """停止录音"""
        self.is_recording = False
        recorded_data = self.audio_buffer
        self.audio_buffer = b""
        return recorded_data

    def process_chunk(self, indata: np.ndarray):
        """
        处理音频块并检测语音活动
        返回: (是否正在说话, 是否检测到语音结束)
        """
        if not self.is_recording:
            return False, False

        pcm_data = indata.astype(np.int16).tobytes()
        is_speech = self.vad.is_speech(pcm_data, DEFAULT_SAMPLE_RATE)

        if is_speech:
            if not self.recording_started:
                self.recording_started = True
                print("检测到语音，开始录制...")
                for chunk in self.pre_buffer:
                    self.audio_buffer += chunk
                self.pre_buffer.clear()
            
            self.audio_buffer += pcm_data
            self.total_silence = 0.0
            return True, False
        else:
            if self.recording_started:
                self.audio_buffer += pcm_data
                self.total_silence += RECORDING_CHUNK_DURATION / 1000.0
                if self.total_silence >= SILENCE_THRESHOLD:
                    self.is_recording = False
                    return False, True
            else:
                self.pre_buffer.append(pcm_data)
            return False, False

    async def stream_until_silence(self) -> AsyncGenerator[bytes, None]:
        """
        流式录制音频 (Persistent Stream Mode)
        基于常驻流，不再反复开关硬件
        """
        self.start_background_recording()
        self.resume()
        
        current_loop = asyncio.get_running_loop()
        self.stream_queue = asyncio.Queue()
        
        def on_audio_chunk(chunk):
            if self.stream_queue:
                current_loop.call_soon_threadsafe(self.stream_queue.put_nowait, chunk)

        self._on_chunk_callback = on_audio_chunk
        
        try:
            while self.is_recording:
                chunk = await self.stream_queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            self._on_chunk_callback = None
            self.stream_queue = None

    def start_background_recording(self):
        """启动常驻录音流（只启动一次）"""
        if self.stream and self.stream.active:
            return

        print("[AudioRecorder] 启动常驻麦克风流...")
        
        self.stream_queue = asyncio.Queue()
        self._current_loop = asyncio.get_event_loop()
        
        def on_audio_chunk_persistent(chunk):
            if self.stream_queue and self._current_loop and not self._current_loop.is_closed():
                try:
                    self._current_loop.call_soon_threadsafe(self.stream_queue.put_nowait, chunk)
                except RuntimeError:
                    pass
        
        self._on_chunk_callback = on_audio_chunk_persistent
        
        def callback(indata, frames, time, status):
            if status:
                pass
            self.process_chunk_stream_persistent(indata)

        self.stream = sd.InputStream(
            samplerate=DEFAULT_SAMPLE_RATE,
            channels=1,
            dtype='int16',
            blocksize=CHUNK_SIZE,
            callback=callback
        )
        self.stream.start()
    
    def process_chunk_stream_persistent(self, indata: np.ndarray):
        """持久连接模式的流式处理：不判断VAD，所有数据都发送"""
        if not self.is_recording:
            return

        pcm_data = indata.astype(np.int16).tobytes()
        if self._on_chunk_callback:
            self._on_chunk_callback(pcm_data)

    def process_chunk_stream(self, indata: np.ndarray):
        """流式处理版本（带VAD）"""
        if not self.is_recording:
            return

        pcm_data = indata.astype(np.int16).tobytes()
        is_speech = self.vad.is_speech(pcm_data, DEFAULT_SAMPLE_RATE)

        if is_speech:
            if not self.recording_started:
                self.recording_started = True
                if self._on_chunk_callback:
                    for chunk in self.pre_buffer:
                        self._on_chunk_callback(chunk)
                self.pre_buffer.clear()
            
            self.total_silence = 0.0
            if self._on_chunk_callback:
                self._on_chunk_callback(pcm_data)
        else:
            if self.recording_started:
                if self._on_chunk_callback:
                    self._on_chunk_callback(pcm_data)
                    
                self.total_silence += RECORDING_CHUNK_DURATION / 1000.0
                if self.total_silence >= SILENCE_THRESHOLD:
                    self.recording_started = False
                    if self._on_chunk_callback:
                        self._on_chunk_callback(None)
            else:
                self.pre_buffer.append(pcm_data)

    def pause(self):
        """暂停业务层录音（硬件继续）"""
        self.is_recording = False
        self.recording_started = False
        self.pre_buffer.clear()

    def resume(self):
        """恢复业务层录音"""
        self.is_recording = True
        self.recording_started = False
        self.total_silence = 0.0
        self.pre_buffer.clear()

    def close_stream(self):
        """关闭麦克风流"""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                print("[AudioRecorder] 麦克风流已关闭")
            except Exception as e:
                logger.error(f"关闭麦克风流失败: {e}")

    async def record_until_silence(self) -> bytes:
        """录制音频直到检测到足够长的静音 (Buffer Mode)"""
        chunks = []
        async for chunk in self.stream_until_silence():
            chunks.append(chunk)
        return b"".join(chunks)