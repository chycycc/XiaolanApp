# -*- coding: utf-8 -*-
"""测试科大讯飞ASR - 保存音频并测试识别"""

import asyncio
import wave
from realtime_asr2 import AudioRecorder
from xfyun_asr import recognize_once

async def test_asr():
    print("开始录音测试...")
    print("请说一段话（说完后停顿1秒自动结束）:")
    
    recorder = AudioRecorder()
    pcm_data = await recorder.record_until_silence()
    
    if not pcm_data:
        print("没有录到音频！")
        return
    
    print(f"录制完成: {len(pcm_data)} bytes")
    
    # 保存为WAV文件供检查
    with wave.open("test_audio.wav", 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16bit = 2 bytes
        wf.setframerate(16000)
        wf.writeframes(pcm_data)
    print("音频已保存到 test_audio.wav (可用播放器检查)")
    
    # 测试讯飞识别
    print("\n正在调用讯飞ASR识别...")
    result = await recognize_once(pcm_data)
    
    if result:
        print(f"✅ 识别结果: {result}")
    else:
        print("❌ 识别结果为空")
        print("\n可能原因:")
        print("1. 请确认已在讯飞开放平台开通'语音听写（流式版）'服务")
        print("2. 请播放 test_audio.wav 确认录音正常")

if __name__ == "__main__":
    asyncio.run(test_asr())
