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
from xfyun_asr import recognize_once, recognize_stream
from xfyun_rtasr import rtasr_client, start_persistent_asr, stop_persistent_asr, send_audio_chunk
from protocols import MsgType, full_client_request, receive_message


# ---------------------------
# 日志
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 1) 豆包大模型
# ============================================================
from config import LLM_PROVIDER, LLM_CONFIGS

current_llm_conf = LLM_CONFIGS.get(LLM_PROVIDER, LLM_CONFIGS["doubao"])

ARK_API_KEY = current_llm_conf["api_key"]
MODEL_NAME = current_llm_conf["model"]

ark_client = Ark(
    base_url=current_llm_conf["base_url"],
    api_key=ARK_API_KEY,
)


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
                ngram_range=(1, 4), # 单字匹配 (1-gram)
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
# 3) TTS（火山引擎 HTTP 版 + 流式版）
# ============================================================
from volc_tts_client import tts_synthesize, tts_synthesize_and_play, play_wav, volc_tts_close, stop_playback




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
        "chaolan", "xaolan", "xalan", "xlaolan","xiaola"
    ]
    return any(key in py for key in possible_xiaolan)


# ============================================================
# 5) 统一问答入口（文本和语音都用这个）
# ============================================================
def answer_question(user_text: str) -> str:
    # 1) 先查知识库
    kb_answer = knowledge_base.query(user_text, threshold=0.55)
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
# 6) GUI (纯语音极简版)
# ============================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class VoiceGUI(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("小蓝语音助手 (Pure Voice Mode)")
        self.geometry("900x600") # 增大窗口
        self.minsize(900, 600)  # 限制最小尺寸

        self.voice_running = False
        self.voice_thread = None
        self.gui_queue = queue.Queue()

        self._build_ui()
        self._start_polling_queue()
        
        # 绑定关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        # 停止播放和语音循环
        self.voice_running = False
        stop_playback()
        self.destroy()

    def _build_ui(self):
        # 顶部标题区
        top = ctk.CTkFrame(self, height=80, corner_radius=0)
        top.pack(fill="x", side="top", pady=(0, 0))

        title = ctk.CTkLabel(
            top, text="迪尔空分 · 智能语音助手  小蓝",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.pack(anchor="w", padx=20, pady=(15, 0))

        subtitle = ctk.CTkLabel(
            top, text="企业级知识库驱动 · 语音唤醒 · 语音问答",
            font=ctk.CTkFont(size=14)
        )
        subtitle.pack(anchor="w", padx=22, pady=(0, 10))

        # 顶部右侧状态
        right = ctk.CTkFrame(top, fg_color="transparent")
        right.place(relx=1.0, rely=0.5, x=-20, y=0, anchor="e")

        self.status_bar = ctk.CTkLabel(
            right, text=f"知识库: {KB_COUNT} | 引擎: 科大讯飞 + Doubao + VolcTTS",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.status_bar.pack(side="right")

        # 底部控制区 (先 pack side=bottom) - 只有按钮
        bottom = ctk.CTkFrame(self, height=100)
        bottom.pack(fill="x", side="bottom", padx=15, pady=(0, 15))

        self.voice_btn = ctk.CTkButton(
            bottom, text="开启语音问答", 
            height=60, 
            font=ctk.CTkFont(size=22, weight="bold"),
            command=self.toggle_voice,
            fg_color="#1F6AA5", hover_color="#144870"
        )
        self.voice_btn.pack(fill="x", padx=30, pady=15)

        # 中间聊天显示区 (最后 pack，占据剩余空间)
        mid = ctk.CTkFrame(self)
        mid.pack(fill="both", expand=True, padx=10, pady=10)

        self.chat_box = ctk.CTkTextbox(
            mid, 
            font=ctk.CTkFont(size=18), # 适中的字体
            wrap="word",
            corner_radius=12,
            fg_color="transparent" # 透明背景适应主题
        )
        # 如果需要更像 ChatBox，可以给 text_color
        self.chat_box.pack(fill="both", expand=True, padx=5, pady=5)
        self.chat_box.configure(state="disabled")
        
        # 初始欢迎语
        self.append_chat("系统", "已就绪。点击下方按钮开始语音对话...", "yellow")


    def append_chat(self, role, text, color=None):
        self.chat_box.configure(state="normal")
        
        # 检查是否是第一条消息（如果为空，则不需要开头的换行）
        is_empty = self.chat_box.get("1.0", "end-1c").strip() == ""
        prefix = "" if is_empty else "\n"

        # 格式化输出
        if role == "你":
            header = f"{prefix}{role} ({time.strftime('%H:%M:%S')}):\n"
        elif role == "小蓝":
            header = f"{prefix}{role} ({time.strftime('%H:%M:%S')}):\n"
        else:
            header = f"{prefix}{role}: " # 去掉换行，改为空格
            
        self.chat_box.insert("end", header)
        self.chat_box.insert("end", f"{text}\n")
        
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    # -------------------- Voice toggle --------------------
    def toggle_voice(self):
        if not self.voice_running:
            self.voice_running = True
            self.voice_run_id = self.voice_run_id + 1 if hasattr(self, 'voice_run_id') else 1
            current_run_id = self.voice_run_id
            
            self.voice_btn.configure(text="关闭 (Listening...)", fg_color="#C0392B", hover_color="#922B21")
            
            self.append_chat("系统", f"语音问答已开启，等待唤醒 (你好小蓝)...")
            print(f"语音问答已开启 (Session {current_run_id})，等待唤醒...")

            self.voice_thread = threading.Thread(target=self._voice_loop_thread, args=(current_run_id,), daemon=True)
            self.voice_thread.start()
        else:
            self.voice_running = False
            # 立即停止播放
            stop_playback()
            
            self.voice_btn.configure(text="开启语音问答", fg_color="#1F6AA5", hover_color="#144870")
            
            self.append_chat("系统", "语音已关闭")
            
            # 立即播放关闭提示音
            def play_off_sound():
                try:
                    play_wav("voice_off.mp3")
                except Exception:
                    pass
            threading.Thread(target=play_off_sound, daemon=True).start()

    def _voice_loop_thread(self, run_id):
        """单独线程跑 asyncio 语音循环"""
        try:
            asyncio.run(self._voice_loop_async(run_id))
        finally:
            # 清理 TTS 客户端
            volc_tts_close()
            logger.info(f"[语音循环] 结束清理完成 (Session {run_id})")

    def _should_continue(self, run_id):
        return self.voice_running and getattr(self, 'voice_run_id', 0) == run_id

    async def _voice_loop_async(self, run_id):
        """
        持久连接模式的语音循环
        """
        recorder = AudioRecorder()
        recognized_sentences = asyncio.Queue()
        
        async def on_sentence(text, data):
            if text:
                await recognized_sentences.put(text)
        
        try:
            threading.Thread(target=play_wav, args=("voice_on.mp3",), daemon=True).start()
        except Exception:
            pass
        
        try:
            await start_persistent_asr(on_sentence_callback=on_sentence)
            logger.info("[语音循环] 持久连接已建立")
        except Exception as e:
            logger.error(f"[语音循环] 建立连接失败: {e}")
            self.gui_queue.put(("error", f"连接失败: {e}"))
            return
        
        try:
            recorder.start_background_recording()
            recorder.resume()
            
            async def audio_sender():
                while self._should_continue(run_id) and rtasr_client.is_connected:
                    if recorder.stream_queue:
                        try:
                            chunk = await asyncio.wait_for(recorder.stream_queue.get(), timeout=0.1)
                            if chunk:
                                await send_audio_chunk(chunk)
                                last_send_time = time.time()
                        except asyncio.TimeoutError:
                            if time.time() - last_send_time > 5:
                                silence = b'\x00' * 1280
                                await send_audio_chunk(silence)
                                last_send_time = time.time()
                    else:
                        await asyncio.sleep(0.05)
            
            sender_task = asyncio.create_task(audio_sender())
            
            is_awake = False
            
            while self._should_continue(run_id):
                try:
                    text = await asyncio.wait_for(recognized_sentences.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                
                if not text:
                    continue
                
                if text in ["嗯", "嗯嗯", "嗯嗯嗯", "哦", "呃", "啊"]:
                    continue
                
                if len(text) < 2 and not text.isascii():
                    continue
                
                # 1) 唤醒模式
                if not is_awake:
                    self.gui_queue.put(("heard", f"[唤醒监听] {text}"))
                    
                    if is_wakeup(text):
                        is_awake = True
                        self.gui_queue.put(("wake_ok", None))
                        try:
                            recorder.pause()
                            play_wav("awake.mp3")
                            recorder.resume()
                        except Exception:
                            recorder.resume()
                    continue
                
                # 2) 问答模式
                t0 = time.time()
                
                self.gui_queue.put(("user", text))
                
                if any(x in text for x in ["退出", "再见", "谢谢"]):
                    try:
                        threading.Thread(target=play_wav, args=("bye.mp3",), daemon=True).start()
                    except Exception:
                        pass
                    self.voice_running = False
                    self.gui_queue.put(("stop", None))
                    break
                
                # 回答问题
                self.gui_queue.put(("status", "Brain is thinking..."))
                reply = answer_question(text)
                t_llm = time.time()
                
                perf_stats = {
                    "t_asr": t0,
                    "t_llm": t_llm
                }
                
                self.gui_queue.put(("bot", reply))
                
                try:
                    recorder.pause()
                    await tts_synthesize_and_play(reply, perf_stats=perf_stats)
                    recorder.resume()
                except Exception:
                    recorder.resume()
            
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass
                
        finally:
            if recorder and recorder.stream:
                try:
                    recorder.stream.stop()
                    recorder.stream.close()
                except Exception:
                    pass

            await stop_persistent_asr()

    # -------------------- Queue polling --------------------
    def _start_polling_queue(self):
        self.after(100, self._poll_queue)

    def _poll_queue(self):
        try:
            while True:
                item = self.gui_queue.get_nowait()
                kind, payload = item

                if kind == "user":
                    self.append_chat("你", payload)
                
                elif kind == "bot":
                    self.append_chat("小蓝", payload)

                elif kind == "status":
                    # 状态更新显示在标题栏或者简单的print
                    self.status_bar.configure(text=payload)

                elif kind == "heard":
                    pass # 不显示监听杂音

                elif kind == "wake_ok":
                    self.append_chat("小蓝", "✅ 已唤醒！请吩咐...")

                elif kind == "stop":
                    self.voice_btn.configure(text="开启语音问答", fg_color="#1F6AA5", hover_color="#144870")
                    self.append_chat("系统", "语音已关闭")

        except queue.Empty:
            pass

        self.after(100, self._poll_queue)


if __name__ == "__main__":
    app = VoiceGUI()
    app.mainloop()
