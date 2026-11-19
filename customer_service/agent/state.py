"""Agent 状态：Evidence、ToolStep、State 与防循环计数器。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

from customer_service.core import config as cfg

MAX_RETRY = cfg.MAX_RETRY
MAX_REPAIR = cfg.MAX_REPAIR
MAX_LLM_CALLS = cfg.MAX_LLM_CALLS
RECURSION_LIMIT = cfg.RECURSION_LIMIT

QuestionType = Literal["qa", "plot", "compare", "aggregate", "multi_hop"]
Verdict = Literal["sufficient", "insufficient", "out_of_corpus"]


@dataclass
class Evidence:
    """一条可引用的原文证据。"""

    eid: str
    book: str
    chapter: str
    chapter_seq: int
    chunk_index: int
    text: str
    score: float
    coverage: float
    source_tool: str

    @property
    def dedup_key(self) -> tuple:
        return (self.book, self.chapter_seq, self.chunk_index)


@dataclass
class ToolStep:
    tool: str
    args: dict
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        payload = (self.tool, tuple(sorted(self.args.items())))
        self.fingerprint = hashlib.md5(repr(payload).encode()).hexdigest()[:12]


def merge_evidence(existing: list[Evidence] | None, new: list[Evidence] | None) -> list[Evidence]:
    """追加 + 按 (book, chapter_seq, chunk_index) 去重。"""
    merged = list(existing or [])
    seen = {e.dedup_key for e in merged}
    for e in new or []:
        if e.dedup_key not in seen:
            merged.append(e)
            seen.add(e.dedup_key)
    return merged


class CustomerState(TypedDict, total=False):
    """Agent 全流程状态；计数器只增不减。"""

    question: str
    history: Annotated[list, add_messages]
    emit: Any                # 流式输出回调（面向 SSE）；不参与序列化

    entities: list[str]
    question_type: QuestionType
    aspects: list[str]
    goal: str

    plan: list[ToolStep]
    completed: list[str]
    evidence: Annotated[list[Evidence], merge_evidence]
    evidence_count_hist: list[int]

    retry: int
    repair_count: int
    llm_calls: int

    verdict: Verdict
    draft: str
    verify_result: dict
    final: str
    label: str
    handoff: bool


def inc_retry(s: CustomerState) -> int:
    return s.get("retry", 0) + 1


def inc_repair(s: CustomerState) -> int:
    return s.get("repair_count", 0) + 1


def inc_llm(s: CustomerState) -> int:
    return s.get("llm_calls", 0) + 1
