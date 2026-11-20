"""全局配置：语料路径、检索参数、模型配置、客服与防循环策略。"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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
QUERY_REWRITE = os.getenv("CS_QUERY_REWRITE", "0") == "1"  # 向量检索可用时无需改写，默认关闭
REWRITE_MAX_TOKENS = 64    # 改写输出的 token 上限，只要几个关键词
MIN_TOKEN_LEN = 2         # 参与检索的最短词长度

# ---- 向量检索（本地小模型推理，不消耗大模型调用）----
EMBED_ENABLED = os.getenv("CS_EMBED", "1") == "1"
EMBED_MODEL = os.getenv("CS_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
EMBED_CACHE_DIR = Path(os.getenv("CS_EMBED_CACHE", str(Path.home() / ".cache" / "cs_embeddings")))
VECTORS_PATH = INDEX_DIR / "vectors.npy"
SEMANTIC_THRESHOLD = 0.55   # 余弦相似度 ≥ 此值视为语义有据
RRF_K = 60                  # RRF 融合常数，越大越平缓
VECTOR_TOP_N = 10           # 向量路召回的候选数

# ---- 大模型（默认 DeepSeek，OpenAI 兼容协议）----
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
LLM_TEMPERATURE = 0.2
LLM_TIMEOUT = 60
PLOT_MAX_TOKENS = 800    #1200 情节梳理的输出上限，防止长文拖慢响应

# ---- 结构导航（第X章 / 本书简介）----
CHAPTER_SUMMARY_CHUNKS = 6   # 章节摘要送入的片段数（章内均匀采样）
CHAPTER_EXCERPT_CHUNKS = 2   # 原文摘录返回的片段数
CHAPTER_MAX_TOKENS = 600     # 章节摘要输出上限（留足余量，避免长句被截断）

# ---- 书目信息（页面展示与购买引导）----
BOOK_DOUBAN_URL = os.getenv("CS_DOUBAN_URL", "https://book.douban.com/subject/1322455/")
BOOK_COVER_PATH = "/static/cover.jpg"
BOOK_COVER_FILE = BASE_DIR / "app" / "static" / "cover.jpg"
BOOK_SOURCE_NAME = "豆瓣读书"
BUY_LINK_LABEL = f"来源：{BOOK_SOURCE_NAME}"
BOOK_REFERRAL_TAG = os.getenv("CS_REFERRAL_TAG", "cs.brandex.top")


def douban_url() -> str:
    """在外链上追加来源标识：Douban 的 from 参数 + UTM，便于对方区分来路。"""
    parts = urlsplit(BOOK_DOUBAN_URL)
    query = dict(parse_qsl(parts.query))
    query.update({
        "from": BOOK_REFERRAL_TAG,
        "utm_source": BOOK_REFERRAL_TAG,
        "utm_medium": "referral",
        "utm_campaign": "book_purchase",
    })
    return urlunsplit(parts._replace(query=urlencode(query)))

# ---- HTTP 鉴权 ----
AUTH_ADMIN_USERNAME = os.getenv("CS_AUTH_USERNAME", "admin")
AUTH_ADMIN_PASSWORD = os.getenv("CS_AUTH_PASSWORD", "change-me-admin")
AUTH_USER_USERNAME = os.getenv("CS_AUTH_USER_USERNAME", "user")
AUTH_USER_PASSWORD = os.getenv("CS_AUTH_USER_PASSWORD", "change-me-user")
# 额外的 user 角色账号，格式 name:password，多个用英文逗号分隔
AUTH_EXTRA_USERS = os.getenv("CS_AUTH_EXTRA_USERS", "jerry:change-me-extra")
AUTH_ENABLED = bool(AUTH_ADMIN_PASSWORD)  # 配置管理员密码后，页面与全部接口均要求鉴权
AUTH_SESSION_MINUTES = int(os.getenv("CS_AUTH_SESSION_MINUTES", "60"))  # 登录有效期
AUTH_COOKIE_NAME = "cs_web_session"


def _parse_extra_users(raw: str) -> dict[str, str]:
    """把 name:password,name:password 解析成字典；忽略格式不完整的条目。"""
    users: dict[str, str] = {}
    for item in raw.split(","):
        name, separator, password = item.partition(":")
        name = name.strip()
        if separator and name and password:
            users[name] = password
    return users


def auth_users() -> dict[str, tuple[str, str]]:
    """账号 → (密码, 角色)。admin 可访问全部，user 仅可对话与查看健康。"""
    users: dict[str, tuple[str, str]] = {}
    if AUTH_ADMIN_PASSWORD:
        users[AUTH_ADMIN_USERNAME] = (AUTH_ADMIN_PASSWORD, "admin")
    if AUTH_USER_PASSWORD:
        users[AUTH_USER_USERNAME] = (AUTH_USER_PASSWORD, "user")
    for name, password in _parse_extra_users(AUTH_EXTRA_USERS).items():
        users.setdefault(name, (password, "user"))
    return users

# ---- 客服策略 ----
# 0：原文无据时直接拒答并建议转人工（客服默认）
# 1：允许模型用通用知识补充，但强制打【非原文内容】标注
GENERAL_FALLBACK = os.getenv("CS_GENERAL_FALLBACK", "0") == "1"
HISTORY_WINDOW = 6        # 每个会话保留最近 3 轮消息

# ---- LLM 调用控制 ----
# rules：按问题类型用规则模板生成工具计划（Agent 路径少 1 次调用）
# llm：让模型生成计划（更灵活，多 1 次调用）
PLAN_MODE = os.getenv("CS_PLAN_MODE", "rules")

# ---- 防循环上限 ----
MAX_RETRY = 2             # 环 A（补证据）最大重试次数
MAX_REPAIR = 1            # 环 B（修答案）最大修复次数
MAX_LLM_CALLS = 5         # 单次请求 LLM 调用上限
RECURSION_LIMIT = 15      # LangGraph 步数上限
