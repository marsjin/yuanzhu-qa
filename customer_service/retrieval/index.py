"""检索层：BM25 索引构建 / 持久化 / 检索 / 覆盖率判定 / 情节采样。"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from customer_service.core import config as cfg
from customer_service.retrieval.corpus import (
    CHAPTER_QUERY_RE,
    cn_to_int,
    split_corpus,
    tokenize,
)


@dataclass
class Hit:
    doc: Document
    score: float
    coverage: float


@dataclass
class RagIndex:
    docs: list[Document]
    tokens: list[list[str]]
    bm25: BM25Okapi
    token_sets: list[set[str]]

    @classmethod
    def create(cls, docs: list[Document], tokens: list[list[str]]) -> "RagIndex":
        return cls(docs=docs, tokens=tokens, bm25=BM25Okapi(tokens),
                   token_sets=[set(t) for t in tokens])


def build_index(force: bool = False) -> RagIndex:
    """构建并持久化索引；已存在时直接加载。"""
    if cfg.INDEX_PATH.exists() and not force:
        return load_index()

    docs, meta = split_corpus()
    tokens = [tokenize(d.page_content) for d in docs]

    cfg.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with cfg.INDEX_PATH.open("wb") as f:
        pickle.dump(
            {"docs": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
             "tokens": tokens},
            f,
        )
    with cfg.INDEX_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {**meta, "chunk_size": cfg.CHUNK_SIZE, "chunk_overlap": cfg.CHUNK_OVERLAP,
             "vocab": len({t for ts in tokens for t in ts})},
            f, ensure_ascii=False, indent=2,
        )
    return RagIndex.create(docs, tokens)


def load_index() -> RagIndex:
    if not cfg.INDEX_PATH.exists():
        return build_index(force=True)
    with cfg.INDEX_PATH.open("rb") as f:
        payload = pickle.load(f)
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in payload["docs"]]
    return RagIndex.create(docs, payload["tokens"])


def search(query: str, index: RagIndex, k: int = cfg.TOP_K,
           coverage_query: str | None = None,
           chapter_range: tuple[int, int] | None = None) -> list[Hit]:
    """BM25 召回 Top-K 并计算关键词覆盖率。"""
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    cov_set = set(tokenize(coverage_query or query))
    if not cov_set:
        return []

    scores = index.bm25.get_scores(q_tokens)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    hits: list[Hit] = []
    for i in order:
        if scores[i] <= 0 or len(hits) >= k:
            break
        if chapter_range:
            seq = index.docs[i].metadata.get("chapter_seq", 0)
            if not (chapter_range[0] <= seq <= chapter_range[1]):
                continue
        coverage = len(cov_set & index.token_sets[i]) / len(cov_set)
        hits.append(Hit(index.docs[i], float(scores[i]), coverage))
    return hits


def sort_hits_by_plot(hits: list[Hit]) -> list[Hit]:
    return sorted(hits, key=lambda h: (h.doc.metadata.get("chapter_seq", 0),
                                       h.doc.metadata.get("chunk_index", 0)))


def chapter_titles(index: RagIndex) -> list[tuple[int, str]]:
    """[(chapter_seq, 章节标题), ...]，按 seq 升序。"""
    titles: dict[int, str] = {}
    for doc in index.docs:
        seq = doc.metadata.get("chapter_seq", 0)
        titles.setdefault(seq, doc.metadata.get("chapter", f"第{seq + 1}章"))
    return sorted(titles.items())


def chapter_seq_of(index: RagIndex, question: str) -> int | None:
    """把问题里的「第X章」解析为 chapter_seq；无法定位返回 None。

    不假设 seq == 章号 - 1：语料开头的「前言」会让两者错位，因此按标题匹配。
    """
    numbers = [cn_to_int(raw) for raw in CHAPTER_QUERY_RE.findall(question)]
    targets = [n for n in numbers if n is not None]
    if not targets:
        return None
    target = targets[0]
    for seq, title in chapter_titles(index):
        match = CHAPTER_QUERY_RE.search(title)
        if match and cn_to_int(match.group(1)) == target:
            return seq
    return None


def chapter_chunks(index: RagIndex, seq: int, limit: int | None = None) -> list[Document]:
    """某章的全部片段，按 chunk_index 升序；limit 取前 N 段。"""
    docs = [d for d in index.docs if d.metadata.get("chapter_seq") == seq]
    docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))
    return docs[:limit] if limit else docs


def sample_plot_spine(index: RagIndex, n: int = cfg.PLOT_SPINE_N) -> list[Hit]:
    """每章取开头片段，按时间线等间隔抽样，拼出故事脊柱。"""
    first_of_chapter: dict[int, Document] = {}
    for doc in index.docs:
        seq = doc.metadata.get("chapter_seq", 0)
        cur = first_of_chapter.get(seq)
        if cur is None or doc.metadata.get("chunk_index", 0) < cur.metadata.get("chunk_index", 0):
            first_of_chapter[seq] = doc
    spine = sort_hits_by_plot([Hit(d, 0.0, 1.0) for d in first_of_chapter.values()])
    if len(spine) <= n:
        return spine
    step = len(spine) / n
    return [spine[int(i * step)] for i in range(n)]


def decide_mode(query: str, hits: list[Hit]) -> tuple[str, float]:
    """覆盖率 ≥ 阈值 → document；否则 llm。"""
    q_tokens = tokenize(query)
    if not q_tokens or not hits:
        return "llm", 0.0
    best = max(h.coverage for h in hits)
    return ("document" if best >= cfg.COVERAGE_THRESHOLD else "llm"), best


def stats(index: RagIndex) -> dict:
    return {
        "book": cfg.BOOK_NAME,
        "chunks": len(index.docs),
        "chapters": len({d.metadata.get("chapter_seq", 0) for d in index.docs}),
        "vocab": len({t for ts in index.tokens for t in ts}),
    }
