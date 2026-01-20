#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import uuid
import time
import wave
import queue
import threading
import asyncio
import logging

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    import difflib

import sounddevice as sd
import websockets
import customtkinter as ctk
from pypinyin import lazy_pinyin
from volcenginesdkarkruntime import Ark

from realtime_asr2 import AudioRecorder, AsrWsClient
from protocols import MsgType, full_client_request, receive_message


def resource_path(relative_path: str) -> str:
    """兼容 PyInstaller 打包后的资源路径"""
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS  # PyInstaller 临时解压目录
    else:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)


# ---------------------------
# 日志
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 1) 豆包大模型
# ============================================================
ARK_API_KEY = "9a67bf10-9397-49ad-a695-b64f19f56d1b"
ark_client = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=ARK_API_KEY,
)
MODEL_NAME = "doubao-1-5-pro-32k-250115"


def chat_with_ark(messages):
    completion = ark_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )
    return completion.choices[0].message.content


# ============================================================
# 2) 知识库（knowledge.xlsx）
# ============================================================
KB_PATH = resource_path("knowledge.xlsx")


class KnowledgeBase:
    def __init__(self, xlsx_path: str):
        self.xlsx_path = xlsx_path
        self.questions = []
        self.answers = []
        self.vectorizer = None
        self.q_matrix = None
        self._load()

    def _load(self):
        if not os.path.exists(self.xlsx_path):
            logger.warning(f"[知识库] 未找到文件: {self.xlsx_path}")
            return

        try:
            df = pd.read_excel(self.xlsx_path, sheet_name=0)
        except Exception as e:
            logger.error(f"[知识库] 读取失败: {e}")
            return

        if df.empty:
            logger.warning("[知识库] 文件为空")
            return

        cols = list(df.columns)
        q_col, a_col = None, None

        for c in cols:
            c_str = str(c)
            if q_col is None and ("问题" in c_str or "问句" in c_str):
                q_col = c
            if a_col is None and ("回答" in c_str or "答案" in c_str):
                a_col = c

        if q_col is None or a_col is None:
            q_col = cols[0]
            a_col = cols[1] if len(cols) > 1 else cols[0]

        qs = df[q_col].astype(str).fillna("").tolist()
        ans = df[a_col].astype(str).fillna("").tolist()

        qa = [(q.strip(), a.strip()) for q, a in zip(qs, ans) if q.strip()]
        self.questions = [x[0] for x in qa]
        self.answers = [x[1] for x in qa]

        if not self.questions:
            logger.warning("[知识库] 没有有效问题")
            return

        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 4),
                min_df=1
            )
            self.q_matrix = self.vectorizer.fit_transform(self.questions)
            logger.info(f"[知识库] 加载成功，共 {len(self.questions)} 条 Q/A（TF-IDF 已建立）")
        else:
            logger.info(f"[知识库] 加载成功，共 {len(self.questions)} 条 Q/A（difflib 模糊匹配模式）")

    def query(self, text: str, threshold: float = 0.60):
        if not text or not self.questions:
            return None
        text = text.strip()

        if SKLEARN_AVAILABLE and self.vectorizer is not None and self.q_matrix is not None:
            v = self.vectorizer.transform([text])
            sims = cosine_similarity(v, self.q_matrix)[0]
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])

            logger.info(f"[知识库] best_score={best_score:.3f}, best_q={self.questions[best_idx]}")
            if best_score >= threshold:
                return self.answers[best_idx]
            return None

        # fallback
        scores = [difflib.SequenceMatcher(None, text, q).ratio() for q in self.questions]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        logger.info(f"[知识库-模糊] best_score={best_score:.3f}, best_q={self.questions[best_idx]}")
        if best_score >= threshold:
            return self.answers[best_idx]
        return None


knowledge_base = KnowledgeBase(KB_PATH)
KB_COUNT = len(knowledge_base.questions)


# ============================================================
# 3) TTS（火山）
# ============================================================
TTS_APP_ID = "2704273799"
TTS_ACCESS_TOKEN = "r-50x_Sojl9QmFyvhFny7ZAWFx_Zs1Be"
TTS_VOICE_TYPE = "zh_male_shaonianzixin_moon_bigtts"
TTS_ENDPOINT = "wss://openspeech.bytedance.com/api/v1/tts/ws_binary"
TTS_ENCODING = "wav"


def get_tts_cluster(voice: str) -> str:
    if voice.startswith("S_"):
        return "volcano_icl"
    return "volcano_tts"


async def tts_synthesize(text: str, filename: str = "reply.wav") -> str:
    cluster = get_tts_cluster(TTS_VOICE_TYPE)
    headers = {"Authorization": f"Bearer;{TTS_ACCESS_TOKEN}"}

    websocket = await websockets.connect(
        TTS_ENDPOINT,
        additional_headers=headers,
        max_size=10 * 1024 * 1024
    )

    try:
        request = {
            "app": {"appid": TTS_APP_ID, "token": TTS_ACCESS_TOKEN, "cluster": cluster},
            "user": {"uid": str(uuid.uuid4())},
            "audio": {"voice_type": TTS_VOICE_TYPE, "encoding": TTS_ENCODING},
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "operation": "submit",
            },
        }

        await full_client_request(websocket, json.dumps(request).encode())

        audio_data = bytearray()
        while True:
            msg = await receive_message(websocket)
            if msg.type == MsgType.AudioOnlyServer:
                audio_data.extend(msg.payload)
                if msg.sequence < 0:
                    break

    except Exception as e:
        logger.error(f"TTS error: {e}")
        return ""
    finally:
        await websocket.close()

    with open(filename, "wb") as f:
        f.write(audio_data)
    return filename


def play_wav(filename: str):
    with wave.open(filename, 'rb') as wf:
        channels = wf.getnchannels()
        framerate = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())

    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    if channels > 1:
        audio_np = audio_np.reshape(-1, channels)

    sd.play(audio_np, framerate)
    sd.wait()


# ============================================================
# 4) ASR 识别一次
# ============================================================
async def recognize_once(pcm_data: bytes) -> str:
    url = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_nostream"
    seg_duration = 200
    recognized_text = ""

    async with AsrWsClient(url, seg_duration) as client:
        async for response in client.recognize_audio(pcm_data):
            if response.is_last_package and response.payload_msg:
                recognized_text = response.payload_msg.get("result", {}).get("text", "")

    return recognized_text.strip()


def is_wakeup(text: str) -> bool:
    if not text:
        return False
    raw = text.replace("，", "").replace("。", "").replace("！", "").replace(" ", "")
    if "你好" not in raw and "nihao" not in raw.lower():
        return False
    py = "".join(lazy_pinyin(raw)).lower()
    possible_xiaolan = [
        "xiaolan", "xiaolang", "xiaonan", "shaolan",
        "chaolan", "xaolan", "xalan", "xlaolan"
    ]
    return any(key in py for key in possible_xiaolan)


# ============================================================
# 5) 统一问答入口（语音用这个）
# ============================================================
def answer_question(user_text: str) -> str:
    # 1) 先查知识库
    kb_answer = knowledge_base.query(user_text, threshold=0.60)
    if kb_answer:
        return kb_answer

    # 2) 走大模型
    messages = [
        {"role": "system", "content": "你是智能语音助手小蓝，请简洁回答用户问题。"},
        {"role": "user", "content": user_text}
    ]
    try:
        return chat_with_ark(messages)
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "抱歉，我遇到了一些问题，请再问一次。"


# ============================================================
# 6) GUI（仅语音版）
# ============================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ChatGUI(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("迪尔空分 · 智能语音助手 小蓝")
        self.geometry("1100x650")
        self.minsize(900, 560)

        self.voice_running = False
        self.voice_thread = None
        self.gui_queue = queue.Queue()

        self._build_ui()
        self._start_polling_queue()

        # 启动欢迎（仅语音版说明）
        self.append_chat("小蓝", "您好，我是迪尔空分公司的智能语音助手小蓝。")

    def _build_ui(self):
        # 顶部标题区
        top = ctk.CTkFrame(self, height=90, corner_radius=0)
        top.pack(fill="x", side="top")

        title = ctk.CTkLabel(
            top, text="迪尔空分 · 智能语音助手  小蓝",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.pack(anchor="w", padx=20, pady=(14, 0))

        subtitle = ctk.CTkLabel(
            top, text="企业级知识库驱动 · 语音唤醒 · 语音问答",
            font=ctk.CTkFont(size=14)
        )
        subtitle.pack(anchor="w", padx=22, pady=(2, 10))

        # 顶部右侧状态+按钮
        right = ctk.CTkFrame(top, fg_color="transparent")
        right.place(relx=1.0, rely=0.5, x=-20, y=0, anchor="e")

        self.status_label = ctk.CTkLabel(
            right,
            text=f"状态：空闲  |  已加载知识库 {KB_COUNT} 条问答",
            font=ctk.CTkFont(size=13)
        )
        self.status_label.pack(side="left", padx=(0, 12))

        self.voice_btn = ctk.CTkButton(
            right, text="开启语音问答", width=130, height=36,
            command=self.toggle_voice
        )
        self.voice_btn.pack(side="left")

        # 主体：聊天显示区
        mid = ctk.CTkFrame(self)
        mid.pack(fill="both", expand=True, padx=15, pady=(10, 10))

        self.chat_box = ctk.CTkTextbox(
            mid, wrap="word",
            font=ctk.CTkFont(size=15),
            corner_radius=12
        )
        self.chat_box.pack(fill="both", expand=True, padx=8, pady=8)
        self.chat_box.configure(state="disabled")

        # 底部提示区（替代原文本输入框）
        bottom = ctk.CTkFrame(self, height=60, corner_radius=12)
        bottom.pack(fill="x", side="bottom", padx=15, pady=(0, 12))

        hint = ctk.CTkLabel(
            bottom,
            text="点击【开启语音问答】→ 说“你好小蓝”唤醒 → 提问；说“退出/再见/谢谢”返回等待唤醒。",
            font=ctk.CTkFont(size=13)
        )
        hint.pack(anchor="w", padx=14, pady=16)

    # -------------------- GUI Chat helpers --------------------
    def append_chat(self, role, text):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", f"{role}：{text}\n\n")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def set_status(self, text):
        self.status_label.configure(text=text)

    # -------------------- Voice toggle --------------------
    def toggle_voice(self):
        if not self.voice_running:
            self.voice_running = True
            self.voice_btn.configure(text="关闭语音问答")
            self.set_status("状态：等待唤醒（你好小蓝）...")

            self.voice_thread = threading.Thread(target=self._voice_loop_thread, daemon=True)
            self.voice_thread.start()

            self.append_chat("系统", "🎙️ 语音问答已开启：请说“你好小蓝”唤醒。")
        else:
            self.voice_running = False
            self.voice_btn.configure(text="开启语音问答")
            self.set_status(f"状态：空闲  |  已加载知识库 {KB_COUNT} 条问答")
            self.append_chat("系统", "🛑 语音问答已关闭。")

    def _voice_loop_thread(self):
        """单独线程跑 asyncio 语音循环"""
        asyncio.run(self._voice_loop_async())

    async def _voice_loop_async(self):
        recorder = AudioRecorder()

        # 语音欢迎
        try:
            wav = await tts_synthesize("语音问答已开启，请说“你好小蓝”唤醒我。", "voice_on.wav")
            if wav:
                play_wav(wav)
        except Exception:
            pass

        while self.voice_running:
            # 1) 等待唤醒
            while self.voice_running:
                pcm_data = await recorder.record_until_silence()
                if not pcm_data:
                    continue

                wake_text = await recognize_once(pcm_data)

                if is_wakeup(wake_text):
                    self.gui_queue.put(("wake_ok", None))
                    try:
                        wav = await tts_synthesize("我在的，请问有什么可以帮您？", "awake.wav")
                        if wav:
                            play_wav(wav)
                    except Exception:
                        pass
                    break

            if not self.voice_running:
                break

            # 2) 问答循环
            while self.voice_running:
                pcm_data = await recorder.record_until_silence()
                if not pcm_data:
                    continue

                user_text = await recognize_once(pcm_data)
                if not user_text:
                    continue

                self.gui_queue.put(("user_voice", user_text))

                # 退出语音模式（回到等待唤醒）
                if any(x in user_text for x in ["退出", "再见", "谢谢"]):
                    try:
                        wav = await tts_synthesize("好的，我会继续等待您的呼唤。", "bye.wav")
                        if wav:
                            play_wav(wav)
                    except Exception:
                        pass
                    self.gui_queue.put(("back_to_wake", None))
                    break

                # 回答
                reply = answer_question(user_text)
                self.gui_queue.put(("bot_reply", reply))

                try:
                    wav = await tts_synthesize(reply, "reply.wav")
                    if wav:
                        play_wav(wav)
                except Exception:
                    pass

        # 关闭播报
        try:
            wav = await tts_synthesize("语音问答已关闭。", "voice_off.wav")
            if wav:
                play_wav(wav)
        except Exception:
            pass

    # -------------------- Queue polling --------------------
    def _start_polling_queue(self):
        self.after(100, self._poll_queue)

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.gui_queue.get_nowait()

                if kind == "user_voice":
                    self.append_chat("你", payload)
                    self.set_status("状态：思考中 ...")

                elif kind == "bot_reply":
                    self.append_chat("小蓝", payload)
                    if self.voice_running:
                        self.set_status("状态：语音对话中 ...")
                    else:
                        self.set_status(f"状态：空闲  |  已加载知识库 {KB_COUNT} 条问答")

                elif kind == "wake_ok":
                    self.append_chat("系统", "✅ 已唤醒：你好，小蓝！")
                    self.set_status("状态：语音对话中 ...")

                elif kind == "back_to_wake":
                    self.append_chat("系统", "（已返回等待唤醒模式）")
                    self.set_status("状态：等待唤醒（你好小蓝）...")

        except queue.Empty:
            pass

        self.after(100, self._poll_queue)


if __name__ == "__main__":
    app = ChatGUI()
    app.mainloop()
