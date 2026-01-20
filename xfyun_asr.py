# -*- coding: utf-8 -*-
"""
科大讯飞实时语音识别 WebSocket 客户端
基于讯飞开放平台 语音听写（流式版）WebAPI
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import websockets

from config import XFYUN_ASR, AUDIO_SAMPLE_RATE

logger = logging.getLogger(__name__)


class XfyunASR:
    """科大讯飞实时语音识别客户端"""
    
    def __init__(self):
        self.app_id = XFYUN_ASR["app_id"]
        self.api_key = XFYUN_ASR["api_key"]
        self.api_secret = XFYUN_ASR["api_secret"]
        self.url = XFYUN_ASR["url"]
    
    def _create_url(self) -> str:
        """生成带鉴权的WebSocket URL"""
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        
        # 拼接签名原文
        signature_origin = f"host: iat-api.xfyun.cn\ndate: {date}\nGET /v2/iat HTTP/1.1"
        
        # HMAC-SHA256 签名
        signature_sha = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_origin.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        signature = base64.b64encode(signature_sha).decode('utf-8')
        
        # 构建授权信息
        authorization_origin = (
            f'api_key="{self.api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature}"'
        )
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')
        
        # 拼接URL参数
        params = {
            "authorization": authorization,
            "date": date,
            "host": "iat-api.xfyun.cn"
        }
        return self.url + "?" + urlencode(params)
    
    def _build_first_frame(self, first_audio: bytes) -> str:
        """构建第一帧请求（包含业务参数和第一段音频）"""
        audio_base64 = base64.b64encode(first_audio).decode('utf-8') if first_audio else ""
        return json.dumps({
            "common": {"app_id": self.app_id},
            "business": {
                "language": "zh_cn",
                "domain": "iat",
                "accent": "mandarin",
                "vad_eos": 3000,  # 静音检测时长(ms)
                "dwa": "wpgs",   # 动态修正
                "ptt": 1,        # 标点
                "nunum": 1       # 数字转写
            },
            "data": {
                "status": 0,  # 第一帧
                "format": "audio/L16;rate=16000",
                "encoding": "raw",
                "audio": audio_base64
            }
        })
    
    def _build_audio_frame(self, audio_data: bytes, is_last: bool = False) -> str:
        """构建音频帧"""
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        return json.dumps({
            "data": {
                "status": 2 if is_last else 1,  # 2=最后一帧, 1=中间帧
                "format": "audio/L16;rate=16000",
                "encoding": "raw",
                "audio": audio_base64
            }
        })
    
    async def recognize(self, pcm_data: bytes) -> str:
        """
        识别PCM音频数据
        
        Args:
            pcm_data: 16kHz 16bit 单声道 PCM 数据
            
        Returns:
            识别出的文本
        """
        if not pcm_data:
            logger.warning("[讯飞ASR] 音频数据为空")
            return ""
        
        logger.info(f"[讯飞ASR] 开始识别，音频长度: {len(pcm_data)} bytes")
        
        url = self._create_url()
        recognized_text = ""
        
        try:
            async with websockets.connect(url) as ws:
                # 增大帧大小以减少帧数，加快发送速度
                # 每帧约500ms = 16000字节 @16kHz 16bit
                frame_size = 16000
                frames = []
                for i in range(0, len(pcm_data), frame_size):
                    end = min(i + frame_size, len(pcm_data))
                    frames.append(pcm_data[i:end])
                
                total_frames = len(frames)
                logger.info(f"[讯飞ASR] 共 {total_frames} 帧音频")
                
                if total_frames == 0:
                    return ""
                
                # 发送第一帧（包含业务参数和第一段音频）
                first_frame = self._build_first_frame(frames[0])
                await ws.send(first_frame)
                
                # 快速发送剩余帧（无延迟）
                for i in range(1, total_frames):
                    is_last = (i == total_frames - 1)
                    await ws.send(self._build_audio_frame(frames[i], is_last))
                
                # 如果只有一帧，需要发送结束帧
                if total_frames == 1:
                    await ws.send(json.dumps({
                        "data": {
                            "status": 2,
                            "format": "audio/L16;rate=16000",
                            "encoding": "raw",
                            "audio": ""
                        }
                    }))
                
                # 接收识别结果
                while True:
                    response = await ws.recv()
                    result = json.loads(response)
                    
                    # 打印完整响应用于调试
                    logger.info(f"[讯飞ASR] 响应: {json.dumps(result, ensure_ascii=False)}")
                    
                    code = result.get("code", -1)
                    if code != 0:
                        error_msg = result.get('message', '未知错误')
                        logger.error(f"[讯飞ASR] 错误 code={code}, message={error_msg}")
                        break
                    
                    data = result.get("data", {})
                    status = data.get("status", 0)
                    
                    # 提取识别结果
                    ws_result = data.get("result", {})
                    if ws_result:
                        ws_list = ws_result.get("ws", [])
                        for ws_item in ws_list:
                            cw_list = ws_item.get("cw", [])
                            for cw in cw_list:
                                recognized_text += cw.get("w", "")
                    
                    # status=2 表示识别结束
                    if status == 2:
                        break
                        
        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"[讯飞ASR] WebSocket连接关闭: {e}")
        except Exception as e:
            logger.error(f"[讯飞ASR] 识别异常: {e}")
        
        logger.info(f"[讯飞ASR] 识别结果: {recognized_text}")
        return recognized_text.strip()


# 全局实例
xfyun_asr = XfyunASR()


async def recognize_once(pcm_data: bytes) -> str:
    """
    兼容接口：识别一次语音
    用于替换原有的火山引擎 recognize_once 函数
    """
    return await xfyun_asr.recognize(pcm_data)
