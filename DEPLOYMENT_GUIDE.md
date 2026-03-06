# 小蓝语音助手 - 部署指南

## 环境要求

| 项目 | 版本 |
|:-----|:-----|
| 操作系统 | Windows 10/11 (64位) |
| Python | 3.12+ |
| 麦克风 | 必须 (语音功能) |
| 网络 | 必须 (API调用) |

---

## 1. 克隆项目

```bash
git clone https://github.com/chycycc/XiaolanApp.git
cd XiaolanApp
```

---

## 2. 创建虚拟环境

```bash
python -m venv .venv

# Windows 激活
.venv\Scripts\activate
```

---

## 3. 安装依赖

```bash
pip install customtkinter sounddevice webrtcvad-wheels pypinyin numpy pandas scikit-learn openpyxl aiohttp pydub volcengine-python-sdk
```

### 依赖说明

| 包名 | 用途 |
|:-----|:-----|
| `customtkinter` | 现代化 GUI 界面框架 |
| `sounddevice` | 麦克风录音 |
| `webrtcvad-wheels` | 语音活动检测 (VAD) |
| `pypinyin` | 汉字转拼音 (唤醒词匹配) |
| `numpy` | 音频数据处理 |
| `pandas` | 知识库数据读取 |
| `scikit-learn` | TF-IDF 知识库匹配 |
| `openpyxl` | Excel 文件读取 |
| `aiohttp` | 异步 HTTP 请求 (TTS) |
| `pydub` | 音频格式转换/播放 |
| `volcengine-python-sdk` | 火山引擎 SDK (LLM) |

---

## 4. 配置 API 密钥

编辑 `config.py`，替换为你自己的 API 密钥：

### 4.1 科大讯飞 - 实时语音转写 (ASR)

> 申请地址: https://www.xfyun.cn/services/rtasr

```python
XFYUN_RTASR_BIGMODEL = {
    "app_id": "你的app_id",
    "api_key": "你的api_key",
    "api_secret": "你的api_secret",
    "url": "wss://office-api-ast-dx.iflyaisol.com/ast/communicate/v1"
}
```

**说明**: 使用的是讯飞**实时语音转写大模型版**，特点：
- 持久 WebSocket 连接，支持连续对话
- 支持 200+ 中文方言
- 15秒无数据才超时断开

### 4.2 火山引擎 - 豆包大模型 (LLM)

> 申请地址: https://console.volcengine.com/ark

```python
LLM_PROVIDER = "doubao"  # 可选: doubao, deepseek, qwen, glm

LLM_CONFIGS = {
    "doubao": {
        "api_key": "你的api_key",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-1-5-lite-32k-250115"
    },
    # 也可以切换为其他大模型:
    "deepseek": {
        "api_key": "你的api_key",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat"
    },
}
```

**说明**: 默认使用**豆包 1.5 Lite**，也可切换为 DeepSeek/通义千问/GLM。修改 `LLM_PROVIDER` 即可切换。

### 4.3 火山引擎 - TTS 语音合成

> 申请地址: https://console.volcengine.com/speech/service/8

```python
VOLCANO_TTS = {
    "app_id": "你的app_id",
    "access_token": "你的access_token",
    "voice_type": "zh_male_shaonianzixin_moon_bigtts",  # 音色
}
```

**说明**: 使用火山引擎**大模型语音合成**，可选音色：
| voice_type | 效果 |
|:-----------|:-----|
| `zh_male_shaonianzixin_moon_bigtts` | 少年新声 (男) |
| `zh_female_daimengchuanmei_moon_bigtts` | 呆萌川妹 (女) |

> 更多音色: https://www.volcengine.com/docs/6561/97465

---

## 5. FFmpeg 配置

项目内置了 `ffmpeg/bin/` 目录。如果缺失或需要更新：

1. 下载 FFmpeg: https://github.com/BtbN/FFmpeg-Builds/releases
2. 解压 `ffmpeg.exe` 和 `ffprobe.exe` 到 `ffmpeg/bin/` 目录

---

## 6. 运行

### 图形界面版 (文本+语音)

```bash
python GUI.py
```

### 纯语音版

```bash
python Play.py
```

---

## 7. 使用方法

1. 启动后点击 **"开启语音"** 按钮
2. 对着麦克风说 **"你好小蓝"** 唤醒
3. 听到提示音后，**直接说出你的问题**
4. 说 **"退出"** / **"再见"** / **"谢谢"** 退出语音模式

---

## 8. 打包为 EXE

```bash
pip install pyinstaller
pyinstaller 小蓝助手.spec
```

生成的 exe 在 `dist/小蓝助手/` 目录下。

---

## 9. 常见问题

### Q: 提示找不到麦克风？

确保系统有可用的录音设备。可运行以下命令检查：

```python
import sounddevice as sd
print(sd.query_devices())
```

### Q: ASR 连接失败？

1. 检查 `config.py` 中的讯飞 API 密钥是否正确
2. 确认网络可以访问 `xfyun.cn`
3. 检查讯飞控制台的用量是否超限

### Q: TTS 没有声音？

1. 检查 `config.py` 中的火山 TTS 配置
2. 确认 `ffmpeg/bin/ffmpeg.exe` 存在
3. 检查系统音频输出设备

### Q: 知识库匹配不到问题？

知识库在 `data/knowledge.xlsx`，可以直接用 Excel 编辑添加新的 Q&A 条目。格式：

| 列名 | 说明 |
|:-----|:-----|
| question | 问题 |
| answer | 答案 |

---

## 10. 技术架构

```
用户说话 → 麦克风录音 (sounddevice + VAD)
               ↓
         讯飞 RTASR (语音→文字)
               ↓
         知识库匹配 (TF-IDF)
               ↓ (无匹配)
         豆包 LLM (大模型回答)
               ↓
         火山 TTS (文字→语音)
               ↓
         扬声器播放 (pydub + ffmpeg)
```
