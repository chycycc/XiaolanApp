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
# ============================================================
# 1) 大模型配置 (从 config.py 加载)
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
                # ngram_range=(2, 4), # 连词匹配
                ngram_range=(1, 4), # 单字匹配
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
# 5) 统一问答入口（GUI 文本和语音都用这个）
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

        # 绑定关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.voice_running = False
        self.voice_thread = None
        self.gui_queue = queue.Queue()

        self._build_ui()
        self._start_polling_queue()

        # 启动欢迎
        # 启动欢迎
        self.append_chat("小蓝", "您好，我是迪尔空分公司的问答机器人小蓝，支持文本和语音两种问答方式。")

    def on_closing(self):
        # 停止播放和语音循环
        self.voice_running = False
        stop_playback()
        self.destroy()

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
            
            # 立即停止播放
            stop_playback()
            
            # 立即播放关闭提示音
            def play_off_sound():
                try:
                    play_wav("voice_off.mp3")
                except Exception:
                    pass
            threading.Thread(target=play_off_sound, daemon=True).start()
    
    # async def _play_voice_off(self): -> Removed
    #    ...

    def _voice_loop_thread(self, run_id):
        """单独线程跑 asyncio 语音循环"""
        try:
            asyncio.run(self._voice_loop_async(run_id))
        finally:
            # 清理 TTS 客户端（避免 Event loop is closed 错误）
            volc_tts_close()
            logger.info(f"[语音循环] 结束清理完成 (Session {run_id})")

    # check helper
    def _should_continue(self, run_id):
        return self.voice_running and getattr(self, 'voice_run_id', 0) == run_id

    async def _voice_loop_async(self, run_id):
        """
        持久连接模式的语音循环
        - 开启时建立一次WebSocket连接
        - 循环中持续发送音频
        - 关闭时才断开连接
        """
        recorder = AudioRecorder()
        
        # 存储识别到的完整句子
        recognized_sentences = asyncio.Queue()
        
        # 句子回调：当识别到完整句子时放入队列
        async def on_sentence(text, data):
            if text:
                await recognized_sentences.put(text)
        
        # 并行启动：1. 开始联网 2. 播放欢迎语
        # 联网任务（后台）
        connect_task = asyncio.create_task(start_persistent_asr(on_sentence_callback=on_sentence))
        
        # 播放欢迎语（阻塞，避免自录音）
        try:
            await asyncio.to_thread(play_wav, "voice_on.mp3")
        except Exception:
            pass
        
        # 等待连接完成
        try:
            await connect_task
            logger.info("[语音循环] 持久连接已建立 (并行启动成功)")
        except Exception as e:
            logger.error(f"[语音循环] 建立连接失败: {e}")
            self.gui_queue.put(("voice_error", f"连接失败: {e}"))
            return
        
        try:
            # 启动常驻麦克风 (此时欢迎语已播完)
            recorder.start_background_recording()
            recorder.resume()
            
            # 后台发送音频的任务
            async def audio_sender():
                """持续发送音频到ASR服务"""
                while self._should_continue(run_id) and rtasr_client.is_connected:
                    # 从麦克风缓冲读取音频块
                    if recorder.stream_queue:
                        try:
                            chunk = await asyncio.wait_for(recorder.stream_queue.get(), timeout=0.1)
                            if chunk:
                                await send_audio_chunk(chunk)
                                last_send_time = time.time()
                        except asyncio.TimeoutError:
                            # 没数据时（例如被pause了），每隔5秒发一次静音帧保活
                            if time.time() - last_send_time > 5:
                                # 发送 1280 字节的静音数据 (约40ms)
                                silence = b'\x00' * 1280
                                await send_audio_chunk(silence)
                                last_send_time = time.time()
                    else:
                        await asyncio.sleep(0.05)
            
            # 启动音频发送任务
            sender_task = asyncio.create_task(audio_sender())
            
            # 主循环：处理识别结果
            is_awake = False
            
            while self._should_continue(run_id):
                try:
                    # 等待识别结果（带超时以便检查should_continue）
                    text = await asyncio.wait_for(recognized_sentences.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                
                if not text:
                    continue
                
                # 过滤无效识别结果（幻听）
                # 1. 过滤单纯的语气词
                if text in ["嗯", "嗯嗯", "嗯嗯嗯", "哦", "呃", "啊"]:
                    print(f"忽略语气词: {text}")
                    continue
                
                # 2. 过滤过短的非唤醒词（少于2个字且非英文）
                if len(text) < 2 and not text.isascii():
                    print(f"忽略短文本: {text}")
                    continue
                
                # 1) 唤醒模式
                if not is_awake:
                    self.gui_queue.put(("voice_heard", f"[唤醒监听] {text}"))
                    
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
                
                self.gui_queue.put(("user_voice", text))
                
                # 退出检测
                if any(x in text for x in ["退出", "再见", "谢谢"]):
                    try:
                        threading.Thread(target=play_wav, args=("bye.mp3",), daemon=True).start()
                    except Exception:
                        pass
                    self.voice_running = False
                    self.gui_queue.put(("voice_stop", None))
                    print("语音问答已通过指令关闭")
                    break
                
                # 回答问题
                reply = answer_question(text)
                t_llm = time.time()
                
                perf_stats = {
                    "t_asr": t0,
                    "t_llm": t_llm
                }
                
                self.gui_queue.put(("bot_reply", reply))
                
                try:
                    # TTS播放前暂停录音（防止回声干扰）
                    recorder.pause()
                    
                    # 暂时使用 HTTP TTS（流式TTS待调试）
                    await tts_synthesize_and_play(reply, perf_stats=perf_stats)
                        
                    # TTS播放完成后恢复录音
                    recorder.resume()
                except Exception:
                    # 确保即使出错也恢复录音
                    recorder.resume()
            
            # 取消发送任务
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass
                
        finally:
            # 停止录音流
            if recorder and recorder.stream:
                try:
                    recorder.stream.stop()
                    recorder.stream.close()
                    logger.info("[语音循环] 麦克风流已关闭")
                except Exception as e:
                    logger.error(f"[语音循环] 关闭麦克风流失败: {e}")

            # 断开持久连接
            await stop_persistent_asr()
            logger.info("[语音循环] 持久连接已断开")

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
