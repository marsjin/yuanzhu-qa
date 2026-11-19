"""全局配置：语料路径、检索参数、模型配置、客服与防循环策略。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

# ---- 语料与产物 ----
BOOK_NAME = "遥远的救世主"
CORPUS_PATH = Path(
    os.getenv("CS_CORPUS_PATH", BASE_DIR / "data" / "corpus" / "遥远的救世主.txt")
)
FAQ_PATH = Path(os.getenv("CS_FAQ_PATH", BASE_DIR / "data" / "faq.json"))
INDEX_DIR = Path(os.getenv("CS_INDEX_DIR", BASE_DIR / ".index"))
INDEX_PATH = INDEX_DIR / "index.pkl"
INDEX_META_PATH = INDEX_DIR / "index_meta.json"
SESSION_DB_PATH = Path(os.getenv("CS_SESSION_DB", BASE_DIR / ".sessions.db"))

# ---- 切分与检索 ----
CHUNK_SIZE = 600          # 片段最大字符数；改动后需 --rebuild
CHUNK_OVERLAP = 80        # 相邻片段重叠字符数
TOP_K = 5                 # 单次检索返回片段数
PLOT_SPINE_N = 10         # 情节梳理的章节采样数（越大越慢，10 约每 5 章 1 点）
COVERAGE_THRESHOLD = 0.5  # 覆盖率 ≥ 此值才认为原文能回答
OUT_OF_CORPUS_THRESHOLD = 0.2  # 覆盖率低于此值直接判为超出语料
MIN_TOKEN_LEN = 2         # 参与检索的最短词长度

# ---- 大模型（默认 DeepSeek，OpenAI 兼容协议）----
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
LLM_TEMPERATURE = 0.2
LLM_TIMEOUT = 60
PLOT_MAX_TOKENS = 800    #1200 情节梳理的输出上限，防止长文拖慢响应

# ---- 客服策略 ----
# 0：原文无据时直接拒答并建议转人工（客服默认）
# 1：允许模型用通用知识补充，但强制打【非原文内容】标注
GENERAL_FALLBACK = os.getenv("CS_GENERAL_FALLBACK", "0") == "1"
HISTORY_WINDOW = 6        # 每个会话保留最近 3 轮消息

# ---- 防循环上限 ----
MAX_RETRY = 2             # 环 A（补证据）最大重试次数
MAX_REPAIR = 1            # 环 B（修答案）最大修复次数
MAX_LLM_CALLS = 5         # 单次请求 LLM 调用上限
RECURSION_LIMIT = 15      # LangGraph 步数上限
