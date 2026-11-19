"""大模型客户端与流式收集工具。"""

from __future__ import annotations

import sys
from typing import Callable

from langchain_openai import ChatOpenAI

from customer_service.core import config as cfg


def build_llm() -> ChatOpenAI:
    """构造 DeepSeek（OpenAI 兼容）客户端。"""
    if not cfg.LLM_API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在 .env 中设置。")
    return ChatOpenAI(
        model=cfg.LLM_MODEL,
        api_key=cfg.LLM_API_KEY,
        base_url=cfg.LLM_BASE_URL,
        temperature=cfg.LLM_TEMPERATURE,
        timeout=cfg.LLM_TIMEOUT,
        max_retries=2,
    )


def stream_collect(
    chain,
    payload: dict,
    on_token: Callable[[str], None] | None = None,
    echo: bool = True,
) -> str:
    """流式收集链输出。

    on_token 用于 SSE 推送；echo=True 时同时打印到 stderr（CLI 模式）。
    """
    buf: list[str] = []
    for chunk in chain.stream(payload):
        if not chunk:
            continue
        buf.append(chunk)
        if on_token is not None:
            on_token(chunk)
        if echo:
            print(chunk, end="", file=sys.stderr, flush=True)
    if echo:
        print(file=sys.stderr)
    return "".join(buf)
