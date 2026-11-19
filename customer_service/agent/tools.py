"""Agent 工具层：8 个原子工具，统一返回 list[Evidence]。

原则：
- 工具是纯函数，不调模型、不存状态；
- 索引通过 set_index() 注入；
- 工具失败返回空列表，由 grade 决定重试或拒答。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from customer_service.core import config as cfg
from customer_service.retrieval import knowledge as kb
from customer_service.retrieval import index as ridx
from customer_service.agent.state import Evidence

_INDEX: ridx.RagIndex | None = None
_TERM_INDEX: dict[str, list[tuple[int, int]]] | None = None


def set_index(idx: ridx.RagIndex) -> None:
    global _INDEX, _TERM_INDEX
    _INDEX = idx
    _TERM_INDEX = None


def _get_index() -> ridx.RagIndex:
    if _INDEX is None:
        raise RuntimeError("索引未注入，请先调用 set_index()")
    return _INDEX


def _get_term_index() -> dict[str, list[tuple[int, int]]]:
    global _TERM_INDEX
    if _TERM_INDEX is not None:
        return _TERM_INDEX
    ti: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for doc, tokens in zip(_get_index().docs, _get_index().tokens):
        meta = doc.metadata
        loc = (int(meta.get("chapter_seq", 0)), int(meta.get("chunk_index", 0)))
        for t in set(tokens):
            ti[t].append(loc)
    _TERM_INDEX = dict(ti)
    return _TERM_INDEX


def _to_evidence(hits: list[ridx.Hit], tool: str) -> list[Evidence]:
    out: list[Evidence] = []
    for i, h in enumerate(hits, 1):
        m = h.doc.metadata
        out.append(Evidence(
            eid=f"E{i}",
            book=m.get("book", cfg.BOOK_NAME),
            chapter=m.get("chapter", "未知章节"),
            chapter_seq=int(m.get("chapter_seq", 0)),
            chunk_index=int(m.get("chunk_index", 0)),
            text=h.doc.page_content,
            score=h.score,
            coverage=h.coverage,
            source_tool=tool,
        ))
    return out


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------
def search_book(query: str, chapter_range: list[int] | tuple[int, int] | None = None,
                k: int = 5) -> list[Evidence]:
    """BM25 检索，可限定章节范围。"""
    cr = None
    if chapter_range:
        lo, hi = int(chapter_range[0]), int(chapter_range[1])
        cr = (lo, hi)
    hits = ridx.search(query, _get_index(), k=k, coverage_query=query, chapter_range=cr)
    return _to_evidence(hits, "search_book")


def plot_spine(n: int = 24) -> list[Evidence]:
    return _to_evidence(ridx.sample_plot_spine(_get_index(), n=n), "plot_spine")


def get_profile(name: str) -> list[Evidence]:
    if name not in kb.CHARACTERS:
        return []
    return [Evidence(
        eid="P0", book=cfg.BOOK_NAME, chapter="档案", chapter_seq=-1, chunk_index=-1,
        text=kb.profile_text(name), score=0.0, coverage=1.0, source_tool="get_profile",
    )]


def list_characters() -> list[Evidence]:
    text = "、".join(kb.CHARACTERS.keys()) or "（无）"
    return [Evidence(
        eid="L0", book=cfg.BOOK_NAME, chapter="人物列表", chapter_seq=-1, chunk_index=-1,
        text=text, score=0.0, coverage=1.0, source_tool="list_characters",
    )]


def find_by_alias(text: str) -> list[Evidence]:
    names = kb.detect_entities(text)
    body = "、".join(names) if names else "（未识别到人物）"
    return [Evidence(
        eid="A0", book=cfg.BOOK_NAME, chapter="别名解析", chapter_seq=-1, chunk_index=-1,
        text=body, score=0.0, coverage=1.0, source_tool="find_by_alias",
    )]


def term_stats(term: str) -> list[Evidence]:
    locs = _get_term_index().get(term.lower(), [])
    if not locs:
        body = f"「{term}」未在原文中出现"
        cov = 0.0
    else:
        first_seq = min(seq for seq, _ in locs)
        body = f"「{term}」在全书出现 {len(locs)} 次，首次出现在第 {first_seq + 1} 章"
        cov = 1.0
    return [Evidence(
        eid="T0", book=cfg.BOOK_NAME, chapter="术语统计", chapter_seq=-1, chunk_index=-1,
        text=body, score=0.0, coverage=cov, source_tool="term_stats",
    )]


def book_meta() -> list[Evidence]:
    m = kb.BOOK_META
    body = f"《{m['name']}》：作者 {m['author']}；主角 {m['protagonist']}；主题：{m['theme']}"
    return [Evidence(
        eid="M0", book=cfg.BOOK_NAME, chapter="书目信息", chapter_seq=-1, chunk_index=-1,
        text=body, score=0.0, coverage=1.0, source_tool="book_meta",
    )]


def faq_search(query: str) -> list[Evidence]:
    entry = kb.match_faq(query)
    if entry is None:
        return []
    return [Evidence(
        eid="F0", book=cfg.BOOK_NAME, chapter="FAQ", chapter_seq=-1, chunk_index=-1,
        text=entry["answer"], score=0.0, coverage=1.0, source_tool="faq_search",
    )]


TOOL_REGISTRY: dict[str, tuple[Any, list[str]]] = {
    "search_book": (search_book, ["query", "chapter_range", "k"]),
    "plot_spine": (plot_spine, ["n"]),
    "get_profile": (get_profile, ["name"]),
    "list_characters": (list_characters, []),
    "find_by_alias": (find_by_alias, ["text"]),
    "term_stats": (term_stats, ["term"]),
    "book_meta": (book_meta, []),
    "faq_search": (faq_search, ["query"]),
}


def execute_step(step) -> list[Evidence]:
    """执行一个计划步骤；忽略模型多给的参数，异常返回空列表。"""
    try:
        fn, allowed = TOOL_REGISTRY[step.tool]
        args = {k: v for k, v in step.args.items() if k in allowed}
        result = fn(**args)
        return result if isinstance(result, list) else []
    except Exception:
        return []
