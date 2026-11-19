"""快路径：单次检索 + 单次生成，覆盖 book_qa / plot / compare。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from customer_service.core import config as cfg
from customer_service.retrieval import index as ridx
from customer_service.retrieval import knowledge as kb
from customer_service.policy import prompts as P
from customer_service.generation.llm import stream_collect

_CITE_FMT = re.compile(r"《([^》]+)》第([一二三四五六七八九十百零两0-9]+)章")


@dataclass
class FastResult:
    text: str
    label: str
    mode: str                     # document / llm / refusal
    coverage: float
    citations: list[str] = field(default_factory=list)
    handoff: bool = False


def format_context(hits: list[ridx.Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, 1):
        chapter = hit.doc.metadata.get("chapter", "未知章节")
        blocks.append(f"[片段{i}｜《{cfg.BOOK_NAME}》{chapter}]\n{hit.doc.page_content}")
    return "\n\n".join(blocks)


def retrieve(question: str, retrieval_query: str, intent: str,
             entities: list[str], index: ridx.RagIndex) -> list[ridx.Hit]:
    if intent == "compare":
        hits: list[ridx.Hit] = []
        for name in entities[:2]:
            hits += ridx.search(f"{name} {question}", index, k=3, coverage_query=question)
        return hits
    if intent == "plot":
        return ridx.sample_plot_spine(index, n=cfg.PLOT_SPINE_N)
    return ridx.search(retrieval_query, index, coverage_query=question)


def _build_chain(llm: ChatOpenAI, system: str, with_profiles: bool = False):
    human = (
        "【人物档案】\n{profiles}\n\n【原文片段】\n{context}\n\n【问题】\n{question}"
        if with_profiles
        else "【原文片段】\n{context}\n\n【问题】\n{question}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder("history", optional=True),
        ("human", human),
    ])
    return prompt | llm | StrOutputParser()


def answer(question: str, index: ridx.RagIndex, llm: ChatOpenAI,
           history: list[BaseMessage], intent: str, entities: list[str],
           retrieval_query: str | None = None,
           on_token: Callable[[str], None] | None = None) -> FastResult:
    """执行快路径并返回结构化结果。"""
    hits = retrieve(question, retrieval_query or question, intent, entities, index)

    if intent == "book_qa":
        mode, coverage = ridx.decide_mode(question, hits)
    else:
        mode, coverage = ("document", 1.0) if hits else ("llm", 0.0)

    # 原文无据：客服默认拒答并建议转人工；可配置为标注式通用回答
    if mode == "llm":
        if not cfg.GENERAL_FALLBACK:
            return FastResult(P.REFUSAL_TEXT, "来源：无（未在原文中找到）", "refusal", coverage,
                              handoff=True)
        system = P.GENERAL_FALLBACK_SYSTEM
        text = stream_collect(
            _build_chain(llm, system),
            {"context": "（无）", "question": question, "history": history},
            on_token=on_token,
        ).strip()
        return FastResult(text, "来源：大模型回复（非原文内容）", "llm", coverage)

    system = {"book_qa": P.QA_SYSTEM, "plot": P.PLOT_SYSTEM, "compare": P.COMPARE_SYSTEM}[intent]
    payload = {
        "context": format_context(hits),
        "question": question,
        "history": history,
    }
    if intent == "compare":
        payload["profiles"] = "\n".join(kb.profile_text(n) for n in entities[:2])

    text = stream_collect(_build_chain(llm, system, with_profiles=(intent == "compare")),
                          payload, on_token=on_token).strip()

    if "【原文未提及】" in text:
        label = "来源：原文未覆盖 + 大模型补充（非原文内容）"
    else:
        label = f"来源：《{cfg.BOOK_NAME}》原文"
    citations = [f"《{b}》第{c}章" for b, c in _CITE_FMT.findall(text)]
    return FastResult(text, label, "document", coverage, citations)

