#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def resource_path(relative_path: str) -> str:
    """兼容 PyInstaller 打包后的资源路径"""
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS  # PyInstaller 临时解压目录
    else:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)


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

from realtime_asr2 import AudioRecorder
from xfyun_asr import recognize_once
from protocols import MsgType, full_client_request, receive_message


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
# 3) TTS（已切换到 Edge TTS）
# ============================================================
from edge_tts_client import tts_synthesize, play_wav



# ============================================================
# 4) ASR 识别一次（已切换到科大讯飞）
# ============================================================
# recognize_once 函数已从 xfyun_asr 模块导入


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
# 5) 统一问答入口（GUI 文本和语音都用这个）
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
# 6) GUI
# ============================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ChatGUI(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("迪尔空分 · 智能问答机器人 小蓝（文本 + 语音版）")
        self.geometry("1100x650")
        self.minsize(900, 560)

        self.voice_running = False
        self.voice_thread = None
        self.gui_queue = queue.Queue()

        self._build_ui()
        self._start_polling_queue()

        # 启动欢迎
        self.append_chat("小蓝", "您好，我是迪尔空分公司的问答机器人小蓝，支持文本和语音两种问答方式。")

    def _build_ui(self):
        # 顶部标题区
        top = ctk.CTkFrame(self, height=80, corner_radius=0)
        top.pack(fill="x", side="top")

        title = ctk.CTkLabel(
            top, text="迪尔空分 · 智能问答机器人  小蓝",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.pack(anchor="w", padx=20, pady=(12, 0))

        subtitle = ctk.CTkLabel(
            top, text="企业级知识库驱动 · 语义理解 · 文本 + 语音问答",
            font=ctk.CTkFont(size=14)
        )
        subtitle.pack(anchor="w", padx=22)

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
            right, text="开启语音问答", width=120, height=36,
            command=self.toggle_voice
        )
        self.voice_btn.pack(side="left")

        # 聊天显示区
        mid = ctk.CTkFrame(self)
        mid.pack(fill="both", expand=True, padx=15, pady=10)

        self.chat_box = ctk.CTkTextbox(
            mid, wrap="word",
            font=ctk.CTkFont(size=15),
            corner_radius=12
        )
        self.chat_box.pack(fill="both", expand=True, padx=8, pady=8)
        self.chat_box.configure(state="disabled")

        # 底部输入区
        bottom = ctk.CTkFrame(self, height=70)
        bottom.pack(fill="x", side="bottom", padx=15, pady=(0, 12))

        self.input_var = ctk.StringVar()
        self.input_entry = ctk.CTkEntry(
            bottom,
            textvariable=self.input_var,
            placeholder_text="请输入您的问题，然后按回车或点击【发送】...",
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(10, 10), pady=10)
        self.input_entry.bind("<Return>", lambda e: self.on_send())

        send_btn = ctk.CTkButton(
            bottom, text="发送", width=120, height=40, command=self.on_send
        )
        send_btn.pack(side="left", padx=(0, 10))

    # -------------------- GUI Chat helpers --------------------
    def append_chat(self, role, text):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", f"{role}：{text}\n\n")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def set_status(self, text):
        self.status_label.configure(text=text)

    # -------------------- Text send --------------------
    def on_send(self):
        user_text = self.input_var.get().strip()
        if not user_text:
            return
        self.input_var.set("")
        self.append_chat("你", user_text)

        self.set_status("状态：思考中 ...")

        def worker():
            reply = answer_question(user_text)
            self.gui_queue.put(("bot_reply", reply))

        threading.Thread(target=worker, daemon=True).start()

    # -------------------- Voice toggle --------------------
    # -------------------- Voice toggle --------------------
    def toggle_voice(self):
        if not self.voice_running:
            self.voice_running = True
            # 生成新的运行ID
            self.voice_run_id = self.voice_run_id + 1 if hasattr(self, 'voice_run_id') else 1
            current_run_id = self.voice_run_id
            
            self.voice_btn.configure(text="关闭语音问答")
            self.set_status("状态：等待唤醒（你好小蓝）...")
            print(f"语音问答已开启 (Session {current_run_id})，等待唤醒...")

            self.voice_thread = threading.Thread(target=self._voice_loop_thread, args=(current_run_id,), daemon=True)
            self.voice_thread.start()
        else:
            self.voice_running = False
            self.voice_btn.configure(text="开启语音问答")
            self.set_status(f"状态：空闲  |  已加载知识库 {KB_COUNT} 条问答")
            print("语音问答已关闭")
            
            # 立即播放关闭提示音
            def play_off_sound():
                import asyncio
                asyncio.run(self._play_voice_off())
            threading.Thread(target=play_off_sound, daemon=True).start()
    
    async def _play_voice_off(self):
        """播放关闭提示音"""
        try:
            play_wav("voice_off.mp3")
        except Exception:
            pass

    def _voice_loop_thread(self, run_id):
        """单独线程跑 asyncio 语音循环"""
        asyncio.run(self._voice_loop_async(run_id))

    # check helper
    def _should_continue(self, run_id):
        return self.voice_running and getattr(self, 'voice_run_id', 0) == run_id

    async def _voice_loop_async(self, run_id):
        recorder = AudioRecorder()

        # 语音欢迎（直接播放预生成文件）
        try:
            play_wav("voice_on.mp3")
        except Exception:
            pass

        while self._should_continue(run_id):
            # 1) 等待唤醒
            while self._should_continue(run_id):
                pcm_data = await recorder.record_until_silence()
                if not pcm_data:
                    continue

                wake_text = await recognize_once(pcm_data)
                if wake_text:
                    self.gui_queue.put(("voice_heard", f"[唤醒监听] {wake_text}"))

                if is_wakeup(wake_text):
                    self.gui_queue.put(("wake_ok", None))
                    try:
                        play_wav("awake.mp3")
                    except Exception:
                        pass
                    break

                if not self._should_continue(run_id):
                    break

            # 2) 问答循环
            while self._should_continue(run_id):
                pcm_data = await recorder.record_until_silence()
                if not pcm_data:
                    continue

                user_text = await recognize_once(pcm_data)
                if not user_text:
                    continue

                self.gui_queue.put(("user_voice", user_text))

                # 退出语音模式
                if any(x in user_text for x in ["退出", "再见", "谢谢"]):
                    try:
                        play_wav("bye.mp3")
                    except Exception:
                        pass
                    # 彻底关闭语音
                    self.voice_running = False
                    self.gui_queue.put(("voice_stop", None))
                    break

                # 回答
                reply = answer_question(user_text)
                self.gui_queue.put(("bot_reply", reply))

                try:
                    wav = await tts_synthesize(reply, "reply.mp3")
                    if wav:
                        play_wav(wav)
                except Exception:
                    pass

        # 关闭播报已在toggle_voice中处理
        pass

    # -------------------- Queue polling --------------------
    def _start_polling_queue(self):
        self.after(100, self._poll_queue)

    def _poll_queue(self):
        try:
            while True:
                item = self.gui_queue.get_nowait()
                kind, payload = item

                if kind == "user_voice":
                    self.append_chat("你", payload)
                    self.set_status("状态：思考中 ...")

                elif kind == "bot_reply":
                    self.append_chat("小蓝", payload)
                    if self.voice_running:
                        self.set_status("状态：语音对话中 ...")
                    else:
                        self.set_status(f"状态：空闲  |  已加载知识库 {KB_COUNT} 条问答")

                elif kind == "voice_heard":
                    # 可选：你也可以不显示监听日志
                    pass

                elif kind == "wake_ok":
                    self.append_chat("系统", "✅ 已唤醒：你好，小蓝！")
                    self.set_status("状态：语音对话中 ...")

                elif kind == "back_to_wake":
                    self.append_chat("系统", "（已返回等待唤醒模式）")
                    self.set_status("状态：等待唤醒（你好小蓝）...")
                    
                elif kind == "voice_stop":
                    self.append_chat("系统", "（语音问答已关闭）")
                    self.voice_btn.configure(text="开启语音问答")
                    self.set_status(f"状态：空闲  |  已加载知识库 {KB_COUNT} 条问答")
                    print("语音问答已通过指令关闭")

        except queue.Empty:
            pass

        self.after(100, self._poll_queue)


if __name__ == "__main__":
    app = ChatGUI()
    app.mainloop()
