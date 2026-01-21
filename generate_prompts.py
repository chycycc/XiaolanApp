# -*- coding: utf-8 -*-
"""
预生成固定语音提示文件
运行一次即可生成所有固定提示音
"""

import asyncio
import asyncio
from volc_tts_client import tts_synthesize

# 固定语音提示列表
FIXED_PROMPTS = {
    "voice_on.mp3": "语音问答已开启，请说你好小蓝唤醒我。",
    "voice_off.mp3": "语音问答已关闭。",
    "awake.mp3": "我在的，请问有什么可以帮您？",
    "bye.mp3": "好的，我会继续等待您的呼唤。",
}


async def generate_all():
    """生成所有固定语音提示"""
    for filename, text in FIXED_PROMPTS.items():
        print(f"生成中: {filename}")
        await tts_synthesize(text, filename)
        print(f"✅ 完成: {filename}")
    print("\n所有固定语音提示已生成！")


if __name__ == "__main__":
    asyncio.run(generate_all())
