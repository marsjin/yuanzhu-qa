"""检索不足时的查询改写：把口语说法换成原著可能出现的用词。

只产出检索关键词，不参与作答，因此不会引入原文没有的事实。
"""

from __future__ import annotations

import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from customer_service.core import config as cfg
from customer_service.core import obs as log

SYSTEM = (
    "你是检索词改写器。把用户问题改写成原著中可能出现的说法，"
    "只输出 2-6 个关键词，用空格分隔；不要解释，不要编造人名或事实。"
)

_cache: dict[str, str] = {}


def expand(question: str, llm: ChatOpenAI) -> str:
    """返回追加到检索式后的关键词串；失败或空结果返回空串。"""
    if question in _cache:
        return _cache[question]
    chain = (
        ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "{q}")])
        | llm.bind(max_tokens=cfg.REWRITE_MAX_TOKENS, temperature=0.0)
        | StrOutputParser()
    )
    try:
        raw = chain.invoke({"q": question}).strip()
    except Exception as exc:
        log.recovery(f"rewrite｜失败 {type(exc).__name__}: {exc}")
        return ""

    words = [w for w in re.split(r"[，,、\s]+", raw) if 1 < len(w) <= 8]
    extra = " ".join(words[:6])
    _cache[question] = extra
    if extra:
        log.recovery(f"rewrite｜{question} → {extra}")
    return extra
