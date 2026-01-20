# -*- coding: utf-8 -*-
"""
Edge TTS 语音合成模块
使用微软 Edge TTS，速度快且免费
"""

import asyncio
import edge_tts
import logging
import os
import uuid
import time

logger = logging.getLogger(__name__)

# Edge TTS 中文语音选项
VOICE = "zh-CN-YunxiNeural"  # 云希 - 年轻男声

# 记录最后生成的文件名映射
_last_generated = {}


async def tts_synthesize(text: str, filename: str = "reply.mp3") -> str:
    """合成语音为MP3文件"""
    if not text:
        return ""
    
    # 对于动态内容使用随机文件名，避免文件锁定问题
    base_name = filename.rsplit('.', 1)[0]
    if base_name == "reply":
        # 动态回复使用随机文件名
        actual_filename = f"reply_{uuid.uuid4().hex[:8]}.mp3"
    else:
        # 固定提示使用指定文件名
        actual_filename = base_name + '.mp3'
    
    try:
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(actual_filename)
        logger.info(f"[Edge TTS] 合成完成: {actual_filename}")
        
        # 记录映射
        _last_generated[filename] = actual_filename
        
        return actual_filename
    except Exception as e:
        logger.error(f"[Edge TTS] 合成失败: {e}")
        return ""


def play_wav(filename: str):
    """播放音频文件"""
    # 查找实际文件
    actual_file = filename
    
    # 检查映射
    if filename in _last_generated:
        actual_file = _last_generated[filename]
    elif not os.path.exists(filename):
        # 尝试.mp3版本
        base_name = filename.rsplit('.', 1)[0]
        mp3_file = base_name + '.mp3'
        if os.path.exists(mp3_file):
            actual_file = mp3_file
    
    if not actual_file or not os.path.exists(actual_file):
        logger.error(f"[Edge TTS] 文件不存在: {filename}")
        return
    
    try:
        from playsound import playsound
        abs_path = os.path.abspath(actual_file)
        playsound(abs_path)
        
        # 播放完成后删除临时回复文件（reply_xxx.mp3）
        if "reply_" in actual_file:
            try:
                time.sleep(0.1)  # 确保文件释放
                os.remove(actual_file)
            except:
                pass
    except Exception as e:
        logger.error(f"[Edge TTS] 播放失败: {e}")
