# -*- coding: utf-8 -*-
"""
小蓝助手 - 统一配置文件
"""

# ========== 科大讯飞语音识别配置 ==========
XFYUN_ASR = {
    "app_id": "e07ef16e",
    "api_key": "60899c6b3c8e6d71a7f39dca82abd78f",
    "api_secret": "OTMxZjJjN2NhYjU3YzY5Nzc2YmUzMDdi",
    "url": "wss://iat-api.xfyun.cn/v2/iat"
}

# ========== 大模型配置 ==========
LLM_PROVIDER = "doubao"  # 可选: doubao, deepseek, qwen, glm

LLM_CONFIGS = {
    "doubao": {
        "api_key": "9a67bf10-9397-49ad-a695-b64f19f56d1b",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-1-5-pro-32k-250115"
    },
    "deepseek": {
        "api_key": "your-deepseek-key",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat"
    },
    "qwen": {
        "api_key": "your-qwen-key",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus"
    },
    "glm": {
        "api_key": "your-glm-key",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4"
    }
}

# ========== 火山TTS配置（保持不变）==========
VOLCANO_TTS = {
    "app_id": "2704273799",
    "access_token": "r-50x_Sojl9QmFyvhFny7ZAWFx_Zs1Be",
    "voice_type": "zh_male_shaonianzixin_moon_bigtts",
    "endpoint": "wss://openspeech.bytedance.com/api/v1/tts/ws_binary"
}

# ========== 音频参数 ==========
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_BITS = 16
