# -*- coding: utf-8 -*-
"""
火山引擎 TTS WebSocket 客户端
"""

import asyncio
import base64
import gzip
import json
import logging
import uuid
import uuid
import time
import os

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
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self):
        """关闭 HTTP 客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        logger.info("[Volc TTS] HTTP客户端已关闭")

    async def synthesize(self, text: str, output_file: str = "reply.mp3") -> str:
        """
        合成语音并保存为文件 (HTTP POST)
        """
        if output_file == "reply.mp3":
             # 主动清理所有的 reply_*.mp3
             self._cleanup_old_files()
             output_file = f"reply_{uuid.uuid4().hex[:8]}.mp3"
             
        # 构建请求头
        headers = {
            "Authorization": f"Bearer; {self.access_token}",
            "Content-Type": "application/json"
        }
        
        # 构建请求体 (HTTP接口通常与WebSocket结构类似)
        request_json = {
            "app": {
                "appid": self.app_id,
                "token": self.access_token,
                "cluster": self.cluster
            },
            "user": {
                "uid": "python_client"
            },
            "audio": {
                "voice_type": self.voice_type,
                "encoding": "mp3",
                "rate": 24000,
                "speed": 10,
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
                logger.error(f"[Volc TTS] HTTP Error: {response.status_code} - {response.text}")
                return None
            
            # HTTP 接口通常返回 JSON，其中 'data' 字段是 Base64 编码的音频
            resp_json = response.json()
            
            if "data" in resp_json:
                audio_base64 = resp_json["data"]
                if audio_base64:
                    audio_data = base64.b64decode(audio_base64)
                    with open(output_file, "wb") as f:
                        f.write(audio_data)
                    
                    logger.info(f"[Volc TTS] 合成完成: {output_file}")
                    self._schedule_cleanup(output_file)
                    return output_file
            
            # 如果没有 data 字段，可能是错误
            if "message" in resp_json:
                 logger.error(f"[Volc TTS] API Error: {resp_json['message']}")
            else:
                 logger.error(f"[Volc TTS] Unknown Response: {resp_json}")
            return None

        except Exception as e:
            logger.error(f"[Volc TTS] Connection Error: {e}")
            return None

    async def synthesize_and_play(self, text: str, perf_stats: dict = None):
        """
        合成语音并直接播放
        """
        # 构建请求头
        headers = {
            "Authorization": f"Bearer; {self.access_token}",
            "Content-Type": "application/json"
        }
        
        # 构建请求体
        request_json = {
            "app": {
                "appid": self.app_id,
                "token": self.access_token,
                "cluster": self.cluster
            },
            "user": {
                "uid": "python_client"
            },
            "audio": {
                "voice_type": self.voice_type,
                "encoding": "mp3",
                "rate": 24000,
                "speed": 10,
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
                return False
            
            resp_json = response.json()
            
            if "data" in resp_json:
                audio_base64 = resp_json["data"]
                if audio_base64:
                    audio_data = base64.b64decode(audio_base64)
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
            
            logger.error(f"[Volc TTS] 合成失败: {resp_json.get('message', 'Unknown')}")
            return False

        except Exception as e:
            logger.error(f"[Volc TTS] Connection Error: {e}")
            return False

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
    """合成并直接播放（跳过文件IO）"""
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


def play_audio_bytes(audio_data: bytes):
    """从内存直接播放音频数据（跳过文件IO）"""
    try:
        import sounddevice as sd
        import io
        from pydub import AudioSegment
        

        
        # 1. 优先查找项目目录下的 ffmpeg/bin
        project_ffmpeg = os.path.join(os.getcwd(), "ffmpeg", "bin")
        ffmpeg_exe = os.path.join(project_ffmpeg, "ffmpeg.exe")
        ffprobe_exe = os.path.join(project_ffmpeg, "ffprobe.exe")
        
        if os.path.exists(ffmpeg_exe) and os.path.exists(ffprobe_exe):
            # 将项目内的 ffmpeg 目录加入 PATH，这样 pydub 就能找到 ffprobe
            os.environ["PATH"] += os.pathsep + project_ffmpeg
            AudioSegment.converter = ffmpeg_exe
            logger.info(f"[Volc TTS] 使用本地 FFmpeg: {project_ffmpeg}")
        else:
            # 2. 回退到系统路径
            sys_ffmpeg = r"c:\Windows\System32\ffmpeg.exe"
            if os.path.exists(sys_ffmpeg):
               AudioSegment.converter = sys_ffmpeg
        
        # 使用 pydub 解码 MP3
        try:
            # pydub 也就是 ffmpeg 解码
            audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
            
            # 转换为 numpy 数组供 sounddevice 播放
            samples = audio.get_array_of_samples()
            import numpy as np
            audio_array = np.array(samples).astype(np.float32) / 32768.0
            
            # 如果是立体声，需要重塑数组
            if audio.channels == 2:
                audio_array = audio_array.reshape((-1, 2))
            
            # 播放
            sd.play(audio_array, samplerate=audio.frame_rate)
            sd.wait()
            return
            
        except Exception as e:
            logger.warning(f"[Volc TTS] pydub内存播放失败: {e}")
            raise # 抛出异常以触发回退逻辑
            
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

