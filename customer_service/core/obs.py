"""可观测性：FLOW / LLM / RECOVERY 三类日志，带 trace_id 前缀。

环境变量：
    CS_LOG=0        静默
    CS_LOG_COLOR=0  强制去色（非 tty 自动去色）
"""

from __future__ import annotations

import contextvars
import os
import sys
import time
from contextlib import contextmanager

_ENABLED = os.getenv("CS_LOG", "1") == "1"
_COLOR = os.getenv("CS_LOG_COLOR", "1") == "1" and sys.stderr.isatty()
_C = {"FLOW": "\033[36m", "LLM": "\033[33m", "RECOVERY": "\033[31m", "RESET": "\033[0m"}

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("cs_trace_id", default="-")


def set_trace(trace_id: str) -> None:
    """绑定当前请求的 trace_id，后续日志自动带前缀。"""
    _trace_id.set(trace_id)


def get_trace() -> str:
    return _trace_id.get()


def _emit(tag: str, msg: str) -> None:
    if not _ENABLED:
        return
    prefix = f"[{_trace_id.get()}] " if _trace_id.get() != "-" else ""
    line = f"{prefix}[{tag}] {msg}"
    if _COLOR:
        print(f"{_C[tag]}{line}{_C['RESET']}", file=sys.stderr)
    else:
        print(line, file=sys.stderr)


def flow(msg: str) -> None:
    _emit("FLOW", msg)


def recovery(msg: str) -> None:
    _emit("RECOVERY", msg)


def llm(node: str, elapsed: float, calls: int, note: str = "") -> None:
    suffix = f"｜{note}" if note else ""
    _emit("LLM", f"{node} 调用大模型，耗时 {elapsed:.2f}s，累计第 {calls} 次{suffix}")


@contextmanager
def timed(node: str, calls: int, note: str = ""):
    """计时一次 LLM 调用，退出时自动打 [LLM] 日志。"""
    t0 = time.time()
    try:
        yield
    finally:
        llm(node, time.time() - t0, calls, note)
