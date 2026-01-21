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
            async with httpx.AsyncClient() as client:
                response = await client.post(self.endpoint, json=request_json, headers=headers, timeout=10.0)
                
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

