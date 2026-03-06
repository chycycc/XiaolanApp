# 小蓝语音助手 - 代码目录说明

## 项目结构

```
XiaolanApp/
│
├── 📄 核心代码
│   ├── GUI.py                 # 图形界面版 (文本+语音双模式) - 633行
│   ├── Play.py                # 纯语音版 (无文本输入) - 563行
│   ├── config.py              # 统一配置文件 (API密钥/模型参数) - 91行
│   ├── audio_recorder.py      # 麦克风录音 + 语音活动检测(VAD) - 242行
│   ├── xfyun_rtasr.py         # 科大讯飞实时语音转写客户端 - 309行
│   └── volc_tts_client.py     # 火山引擎 TTS 语音合成客户端 - 454行
│
├── 📄 工具脚本
│   └── generate_prompts.py    # 预生成固定提示音(唤醒音/休眠音等) - 30行
│
├── 📁 audio/                  # 音频资源目录
│   ├── awake.mp3              # 唤醒提示音 ("我在")
│   ├── bye.mp3                # 休眠提示音 ("再见")
│   ├── im_here.mp3            # 待机提示音
│   ├── voice_on.mp3           # 语音开启提示
│   └── voice_off.mp3          # 语音关闭提示
│
├── 📁 data/                   # 数据资源目录
│   └── knowledge.xlsx         # 知识库 (1264条 Q&A)
│
├── 📁 hooks/                  # PyInstaller 打包钩子
│   └── hook-webrtcvad.py      # webrtcvad 库打包配置
│
├── 📁 ffmpeg/                 # 内置 FFmpeg (音频解码用)
│   └── bin/
│       ├── ffmpeg.exe         # 音频转码工具
│       └── ffprobe.exe        # 音频探测工具
│
├── 📄 配置文件
│   ├── 小蓝助手.spec          # PyInstaller 打包配置 (主用)
│   ├── GUI.spec               # PyInstaller 打包配置 (备用)
│   ├── .gitignore             # Git 忽略规则
│   └── robot.ico              # 应用图标
│
└── 📄 文档
    └── PROJECT_OPTIMIZATION_REPORT.md  # 项目技术改造报告
```

---

## 核心文件功能说明

### GUI.py - 图形界面版

主入口之一，提供**文本+语音双模式**的交互界面。

| 模块分区 | 功能 |
|:---------|:-----|
| 1) 大模型配置 | 从 config.py 加载 LLM 配置 |
| 2) 知识库 | TF-IDF 匹配 1264 条本地 Q&A |
| 3) TTS | 火山引擎语音合成 |
| 4) 唤醒词检测 | 模糊拼音匹配 "你好小蓝" |
| 5) 统一问答入口 | 知识库优先，无匹配时调用 LLM |
| 6) ChatGUI 主类 | 界面构建 + 语音循环 + 事件处理 |

### Play.py - 纯语音版

主入口之二，提供**纯语音交互**界面（无文本输入框）。

结构与 GUI.py 类似，但去掉了文本输入功能，适合嵌入式/展示场景。

### config.py - 统一配置

集中管理所有 API 密钥和参数：
- 科大讯飞 ASR 配置 (标准版/大模型版)
- LLM 大模型配置 (豆包/DeepSeek/通义千问/GLM)
- 火山引擎 TTS 配置
- 音频参数

### audio_recorder.py - 录音模块

提供 `AudioRecorder` 类：
- `start_background_recording()` - 启动常驻麦克风流
- `stream_until_silence()` - 流式返回音频数据
- `pause()` / `resume()` - 控制录音状态
- 内置 WebRTC VAD 语音活动检测

### xfyun_rtasr.py - 语音识别

科大讯飞实时语音转写客户端 (大模型版)：
- `start_persistent_asr()` - 建立持久 WebSocket 连接
- `send_audio_chunk()` - 发送音频数据
- `stop_persistent_asr()` - 断开连接
- 支持 200+ 方言，15秒无数据才超时

### volc_tts_client.py - 语音合成

火山引擎 TTS 客户端：
- `tts_synthesize_and_play()` - 合成并流式播放
- `tts_synthesize()` - 合成保存为 MP3
- `stop_playback()` - 停止当前播放
- 智能分句 + 连接池优化

---

## 数据流

```
用户说话 → audio_recorder (VAD) → xfyun_rtasr (ASR) → 文本
                                                        ↓
                                            knowledge.xlsx (本地匹配)
                                                        ↓ (无匹配)
                                               豆包 LLM (大模型)
                                                        ↓
                                            volc_tts_client (TTS) → 语音播报
```
