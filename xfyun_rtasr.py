# -*- coding: utf-8 -*-
"""
科大讯飞实时语音转写客户端 (rtasr)
支持持久WebSocket长连接，适合连续对话场景

特点：
- 只要持续发送音频数据，连接就不会断开
- 15秒无数据才会超时断开
- 支持不限时长的持续识别

API文档: https://www.xfyun.cn/doc/asr/rtasr/API.html
"""

import asyncio
import hashlib
import hmac
import base64
import json
import time
import logging
from urllib.parse import urlencode

import websockets

from config import XFYUN_RTASR_BIGMODEL as XFYUN_RTASR, AUDIO_SAMPLE_RATE

logger = logging.getLogger(__name__)


class XfyunRTASR:
    """讯飞实时语音转写客户端（持久连接版）"""
    
    def __init__(self):
        self.app_id = XFYUN_RTASR["app_id"]
        self.api_key = XFYUN_RTASR["api_key"]
        self.base_url = XFYUN_RTASR["url"]
        
        # WebSocket连接
        self.ws = None
        self.is_connected = False
        
        # 识别结果回调
        self.on_result_callback = None
        
        # 当前识别的句子
        self.current_sentence = ""
        self.seg_results = {}  # seg_id -> text
        
    def _create_url(self) -> str:
        """生成带签名的WebSocket URL（大模型版）"""
        import uuid as uuid_module
        from datetime import datetime, timezone, timedelta
        
        # 参数准备
        params = {
            "accessKeyId": self.api_key,  # 大模型版用api_key作为accessKeyId
            "appId": self.app_id,
            "uuid": str(uuid_module.uuid4()).replace("-", ""),
            "utc": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%dT%H:%M:%S+0800"),
            "audio_encode": "pcm_s16le",
            "lang": "autodialect",  # 自动方言识别
            "samplerate": "16000",
        }
        
        # 1. 按参数名升序排序
        sorted_keys = sorted(params.keys())
        
        # 2. URL编码并拼接为 baseString
        from urllib.parse import quote
        encoded_pairs = []
        for key in sorted_keys:
            encoded_key = quote(key, safe='')
            encoded_value = quote(params[key], safe='')
            encoded_pairs.append(f"{encoded_key}={encoded_value}")
        
        base_string = "&".join(encoded_pairs)
        
        # 3. 使用 accessKeySecret (api_secret) 进行 HmacSHA1 签名
        api_secret = XFYUN_RTASR.get("api_secret", "")
        signature = hmac.new(
            api_secret.encode(),
            base_string.encode(),
            hashlib.sha1
        ).digest()
        signature = base64.b64encode(signature).decode()
        
        # 4. 将signature添加到参数中
        params["signature"] = signature
        
        # 5. 构建最终URL
        return f"{self.base_url}?{urlencode(params)}"
    
    async def connect(self):
        """建立WebSocket连接"""
        if self.is_connected:
            return
            
        url = self._create_url()
        logger.info("[讯飞RTASR] 正在建立持久连接...")
        
        try:
            self.ws = await websockets.connect(url)
            
            # 等待握手响应
            response = await self.ws.recv()
            result = json.loads(response)
            
            # 大模型版响应格式: {"msg_type": "action", "data": {"action": "started", ...}}
            # 标准版响应格式: {"code": "0", "action": "started", ...}
            msg_type = result.get("msg_type")
            data = result.get("data", {})
            
            if msg_type == "action" and data.get("action") == "started":
                # 大模型版连接成功
                self.is_connected = True
                self.session_id = data.get("sessionId", "")
                logger.info(f"[讯飞RTASR] ✓ 持久连接已建立 (大模型版, sessionId={self.session_id[:16]}...)")
            elif result.get("code") == "0":
                # 标准版连接成功
                self.is_connected = True
                logger.info("[讯飞RTASR] ✓ 持久连接已建立 (标准版)")
            else:
                logger.error(f"[讯飞RTASR] 连接失败: {result}")
                raise Exception(f"连接失败: {result.get('desc', result)}")
                
        except Exception as e:
            logger.error(f"[讯飞RTASR] 连接异常: {e}")
            self.is_connected = False
            raise
    
    async def disconnect(self):
        """断开WebSocket连接"""
        if self.ws and self.is_connected:
            try:
                # 发送结束标志
                await self.ws.send(json.dumps({"end": True}))
                await self.ws.close()
            except:
                pass
            finally:
                self.is_connected = False
                self.ws = None
                logger.info("[讯飞RTASR] 连接已断开")
    
    async def send_audio(self, audio_chunk: bytes):
        """发送音频数据（建议每40ms发送1280字节）"""
        if not self.is_connected or not self.ws:
            return
            
        try:
            await self.ws.send(audio_chunk)
        except Exception as e:
            logger.error(f"[讯飞RTASR] 发送音频失败: {e}")
            self.is_connected = False
    
    async def receive_results(self):
        """接收识别结果的协程（应在后台持续运行）"""
        while self.is_connected and self.ws:
            try:
                response = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
                result = json.loads(response)
                
                # 大模型版格式: {"msg_type": "result", "res_type": "asr", "data": {...}}
                # 标准版格式: {"action": "result", "code": "0", "data": "..."}
                msg_type = result.get("msg_type")
                
                if msg_type == "result":
                    # 大模型版识别结果
                    data = result.get("data", {})
                    text = self._parse_result_bigmodel(data)
                    
                    if text and self.on_result_callback:
                        await self.on_result_callback(text, data)
                        
                elif msg_type == "action":
                    # 大模型版动作消息（如started等），跳过
                    pass
                else:
                    # 标准版格式
                    action = result.get("action")
                    code = result.get("code")
                    
                    if code and code != "0":
                        logger.error(f"[讯飞RTASR] 错误: {result}")
                        continue
                    
                    if action == "result":
                        data = json.loads(result.get("data", "{}"))
                        text = self._parse_result(data)
                        
                        if text and self.on_result_callback:
                            await self.on_result_callback(text, data)
                        
            except asyncio.TimeoutError:
                # 超时是正常的，继续等待
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.info("[讯飞RTASR] 接收循环：连接已关闭")
                self.is_connected = False
                break
            except Exception as e:
                logger.error(f"[讯飞RTASR] 接收结果异常: {e}")
                break
    
    def _parse_result_bigmodel(self, data: dict) -> str:
        """解析大模型版识别结果"""
        cn = data.get("cn", {})
        st = cn.get("st", {})
        rt_list = st.get("rt", [])
        
        # 提取文字
        words = []
        for rt in rt_list:
            ws_list = rt.get("ws", [])
            for ws in ws_list:
                cw_list = ws.get("cw", [])
                for cw in cw_list:
                    word = cw.get("w", "")
                    words.append(word)
        
        text = "".join(words)
        
        # 去除开头的错位标点（这些标点属于上一句，不应出现在新句子开头）
        text = text.lstrip("。，、！？；：,.!?;:")
        
        # type: "0" 表示最终结果，"1" 表示中间结果
        result_type = st.get("type", "1")
        seg_id = data.get("seg_id", 0)
        
        if result_type == "0":
            # 最终结果
            self.seg_results[seg_id] = text
            logger.info(f"[讯飞RTASR] 识别结果: {text}")
            return text
        else:
            # 中间结果，暂不返回
            return None
    
    def _parse_result(self, data: dict) -> str:
        """解析识别结果，返回当前句子的文本"""
        cn = data.get("cn", {})
        st = cn.get("st", {})
        rt_list = st.get("rt", [])
        
        # 提取文字
        words = []
        for rt in rt_list:
            ws_list = rt.get("ws", [])
            for ws in ws_list:
                cw_list = ws.get("cw", [])
                for cw in cw_list:
                    word = cw.get("w", "")
                    words.append(word)
        
        text = "".join(words)
        
        # 判断是否是最终结果
        # type: "0" 表示最终结果，"1" 表示中间结果
        result_type = st.get("type", "1")
        seg_id = data.get("seg_id", 0)
        
        if result_type == "0":
            # 最终结果，保存
            self.seg_results[seg_id] = text
            return text
        else:
            # 中间结果，可用于实时显示
            return None
    
    def get_full_text(self) -> str:
        """获取所有已识别的完整文本"""
        sorted_segs = sorted(self.seg_results.items())
        return "".join([text for _, text in sorted_segs])
    
    def clear_results(self):
        """清空识别结果"""
        self.seg_results.clear()
        self.current_sentence = ""


# 全局实例
rtasr_client = XfyunRTASR()


async def start_persistent_asr(on_sentence_callback=None):
    """
    启动持久ASR连接
    
    Args:
        on_sentence_callback: 当识别出完整句子时的回调函数，签名: async def callback(text, data)
    """
    rtasr_client.on_result_callback = on_sentence_callback
    await rtasr_client.connect()
    
    # 启动后台接收任务
    asyncio.create_task(rtasr_client.receive_results())
    
    return rtasr_client


async def stop_persistent_asr():
    """停止持久ASR连接"""
    await rtasr_client.disconnect()


async def send_audio_chunk(chunk: bytes):
    """发送音频数据块"""
    await rtasr_client.send_audio(chunk)
