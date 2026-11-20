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
_EXCERPT_RE = re.compile(r"原文|摘录|原话|原句|怎么写的")


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
    chapter_range = ridx.chapter_range_of(index, question)
    if intent == "compare":
        hits: list[ridx.Hit] = []
        for name in entities[:2]:
            hits += ridx.search(f"{name} {question}", index, k=3,
                                coverage_query=question, chapter_range=chapter_range)
        return hits
    if intent == "plot":
        return ridx.sample_plot_spine(index, n=cfg.PLOT_SPINE_N)
    return ridx.search(retrieval_query, index, coverage_query=question,
                       chapter_range=chapter_range)


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


def _spread(docs: list, n: int) -> list:
    """章内均匀采样，避免只看开头导致漏掉本章后半段。"""
    if len(docs) <= n:
        return docs
    step = (len(docs) - 1) / (n - 1)
    picked = sorted({round(i * step) for i in range(n)})
    return [docs[i] for i in picked]


def _book_intro(index: ridx.RagIndex) -> FastResult:
    """书目导览：直接由档案与目录拼装，不检索、不调用模型。"""
    meta = kb.BOOK_META
    titles = ridx.chapter_titles(index)
    toc = "\n".join(title for _, title in titles)
    text = (
        f"《{meta['name']}》是{meta['author']}创作的长篇小说。\n\n"
        f"{meta['summary']}\n\n"
        f"全书共收录 {len(titles)} 个章节单元：\n{toc}"
    )
    return FastResult(text, f"来源：《{meta['name']}》书目信息", "document", 1.0)


def _chapter_answer(question: str, index: ridx.RagIndex, llm: ChatOpenAI,
                    history: list[BaseMessage],
                    on_token: Callable[[str], None] | None) -> FastResult:
    """单章问答：按 metadata 定位章节，证据限定在本章内。"""
    titles = dict(ridx.chapter_titles(index))
    seq = ridx.chapter_seq_of(index, question)
    if seq is None:
        text = (
            f"本书共收录 {len(titles)} 个章节单元，未找到您提到的章节。"
            "可以按「第X章讲了什么」提问，或直接询问书中人物与情节。"
        )
        return FastResult(text, "来源：服务范围提示", "refusal", 0.0)

    chapter = titles[seq]
    docs = ridx.chapter_chunks(index, seq)
    if not docs:
        return FastResult(P.REFUSAL_TEXT, "来源：无（未在原文中找到）", "refusal", 0.0,
                          handoff=True)

    citation = f"《{cfg.BOOK_NAME}》{chapter}"

    # 明确要原文 → 直接摘录，0 次模型调用
    if _EXCERPT_RE.search(question):
        excerpt = "\n".join(d.page_content for d in docs[:cfg.CHAPTER_EXCERPT_CHUNKS])
        text = f"《{cfg.BOOK_NAME}》{chapter}原文摘录：\n\n{excerpt}"
        return FastResult(text, f"来源：《{cfg.BOOK_NAME}》{chapter}（原文）",
                          "document", 1.0, [citation])

    picked = _spread(docs, cfg.CHAPTER_SUMMARY_CHUNKS)
    hits = [ridx.Hit(d, 1.0, 1.0) for d in picked]
    text = stream_collect(
        _build_chain(llm.bind(max_tokens=cfg.CHAPTER_MAX_TOKENS), P.CHAPTER_SYSTEM),
        {"context": format_context(hits), "question": question, "history": history},
        on_token=on_token,
    ).strip()
    return FastResult(text, f"来源：《{cfg.BOOK_NAME}》{chapter}", "document", 1.0, [citation])


def answer(question: str, index: ridx.RagIndex, llm: ChatOpenAI,
           history: list[BaseMessage], intent: str, entities: list[str],
           retrieval_query: str | None = None,
           on_token: Callable[[str], None] | None = None) -> FastResult:
    """执行快路径并返回结构化结果。"""
    if intent == "book_intro":
        return _book_intro(index)
    if intent == "chapter":
        return _chapter_answer(question, index, llm, history, on_token)

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

    if intent == "plot":
        chain = _build_chain(llm.bind(max_tokens=cfg.PLOT_MAX_TOKENS), system)
    else:
        chain = _build_chain(llm, system, with_profiles=(intent == "compare"))
    text = stream_collect(chain, payload, on_token=on_token).strip()

    if "【原文未提及】" in text:
        label = "来源：原文未覆盖 + 大模型补充（非原文内容）"
    else:
        label = f"来源：《{cfg.BOOK_NAME}》原文"
    citations = [f"《{b}》第{c}章" for b, c in _CITE_FMT.findall(text)]
    return FastResult(text, label, "document", coverage, citations)

