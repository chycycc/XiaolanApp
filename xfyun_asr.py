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
                # "dwa": "wpgs",   # 动态修正 (开启会导致客户端简单拼接时出现重复，故关闭)
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

    async def recognize_stream(self, audio_generator) -> str:
        """
        [New] 流式识别接口
        
        Args:
            audio_generator: 异步生成器，yield bytes 音频数据
            
        Returns:
            识别出的文本
        """
        url = self._create_url()
        recognized_text = ""
        
        # 创建音频缓冲队列
        audio_buffer_queue = asyncio.Queue()
        
        # 定义音频采集任务（立即启动麦克风）
        async def collect_audio_task():
            try:
                # 标记开始采集
                logger.info("[讯飞ASR] 麦克风已开启，正在缓冲音频...")
                async for chunk in audio_generator:
                    await audio_buffer_queue.put(chunk)
                await audio_buffer_queue.put(None) # 结束标记
            except Exception as e:
                logger.error(f"[讯飞ASR] 音频采集出错: {e}")
                await audio_buffer_queue.put(None)

        # 立即启动采集任务
        collector = asyncio.create_task(collect_audio_task())
        # logger.info("[讯飞ASR] 监听中...") 

        try:
            # --- Lazy Connection Logic ---
            # 1. 先等待麦克风采集到第一个数据包（意味着VAD触发）
            #    这样可以避免在静音时建立无用的WebSocket连接
            first_chunk_peek = await audio_buffer_queue.get()
            
            # 放回队列（或者在sender里特殊处理，这里简单起见放回去，如果不方便放回，就传递给sender）
            # 由于 asyncio.Queue 没有 peek，我们只能取出来再传给 sender
            
            if first_chunk_peek is None:
                # 刚开始就结束了？
                return ""
            
            logger.info("[讯飞ASR] 检测到语音，正在建立连接...")
            
            # 2. 建立连接
            async with websockets.connect(url) as ws:
                # 发送任务
                async def sender():
                    first_chunk = True
                    
                    # 先发送我们 peek 到的这一帧
                    if first_chunk_peek:
                        await ws.send(self._build_first_frame(first_chunk_peek))
                        first_chunk = False
                        
                    while True:
                        # 从缓冲队列读取后续帧
                        chunk = await audio_buffer_queue.get()
                        
                        if chunk is None:
                            # 录音结束
                            break
                        
                        if not chunk: # Skip empty
                            continue
                        
                        if first_chunk:
                            # 第一帧 (如果pre-buffer logic改变导致peek为空，这里兜底)
                            await ws.send(self._build_first_frame(chunk))
                            first_chunk = False
                        else:
                            # 中间帧
                            await ws.send(self._build_audio_frame(chunk, is_last=False))
                    
                    # 发送结束帧
                    await ws.send(json.dumps({
                        "data": {
                            "status": 2,
                            "format": "audio/L16;rate=16000",
                            "encoding": "raw",
                            "audio": ""
                        }
                    }))
                    logger.info("[讯飞ASR] 音频发送完毕")

                # 接收任务
                async def receiver():
                    nonlocal recognized_text
                    while True:
                        response = await ws.recv()
                        result = json.loads(response)
                        
                        code = result.get("code", -1)
                        if code != 0:
                            logger.error(f"[讯飞ASR] 错误 code={code}, message={result.get('message')}")
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
                                    word = cw.get("w", "")
                                    recognized_text += word
                        
                        if status == 2:
                            # 最终结果
                            break

                # 并发执行发送和接收
                send_task = asyncio.create_task(sender())
                recv_task = asyncio.create_task(receiver())
                
                # 等待直到接收完成
                await recv_task
                
                # 确保发送任务也正常结束
                try:
                    await send_task
                except Exception:
                    pass
                    
        except websockets.exceptions.ConnectionClosedOK:
            # 正常关闭，无需报错
            pass
        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"[讯飞ASR] WebSocket连接关闭: {e}")
        except Exception as e:
            logger.error(f"[讯飞ASR] 识别异常: {e}")
        finally:
            # 确保采集任务被清理
            if not collector.done():
                collector.cancel()
                try:
                    await collector
                except asyncio.CancelledError:
                    pass
        
        return recognized_text.strip()


# 全局实例
xfyun_asr = XfyunASR()


async def recognize_once(pcm_data: bytes) -> str:
    """兼容接口"""
    return await xfyun_asr.recognize(pcm_data)

async def recognize_stream(audio_generator) -> str:
    """流式接口导出"""
    return await xfyun_asr.recognize_stream(audio_generator)



# 全局实例
xfyun_asr = XfyunASR()


async def recognize_once(pcm_data: bytes) -> str:
    """兼容接口"""
    return await xfyun_asr.recognize(pcm_data)

async def recognize_stream(audio_generator) -> str:
    """流式接口导出"""
    return await xfyun_asr.recognize_stream(audio_generator)
