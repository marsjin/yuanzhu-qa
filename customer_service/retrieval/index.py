"""检索层：BM25 索引构建 / 持久化 / 检索 / 覆盖率判定 / 情节采样。"""

from __future__ import annotations

import json
import math
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


def doc_tokens(doc: Document) -> list[str]:
    """片段分词：把章节标题并入词表，让「第X章」成为可检索的 token。

    只影响检索用的 tokens，不改 page_content，因此回答与摘录不会出现前缀。
    """
    return tokenize(f"{doc.metadata.get('chapter', '')} {doc.page_content}")


def build_index(force: bool = False) -> RagIndex:
    """构建并持久化索引；已存在时直接加载。"""
    if cfg.INDEX_PATH.exists() and not force:
        return load_index()

    docs, meta = split_corpus()
    tokens = [doc_tokens(d) for d in docs]

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


def query_weights(index: RagIndex, tokens: list[str]) -> dict[str, float]:
    """查询词权重：语料内用 BM25 的 idf，未登录词取上限值。

    未登录词意味着全书都没有这个词，属于最难命中的信号，按上限计权，
    避免「内容/本书」这类泛词独力把覆盖率抬过阈值。
    """
    ceiling = math.log(len(index.docs) + 1)
    return {t: max(index.bm25.idf.get(t, ceiling), 1e-6) for t in set(tokens)}


def search(query: str, index: RagIndex, k: int = cfg.TOP_K,
           coverage_query: str | None = None,
           chapter_range: tuple[int, int] | None = None) -> list[Hit]:
    """BM25 召回 Top-K，并按 IDF 加权计算关键词覆盖率。"""
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    weights = query_weights(index, tokenize(coverage_query or query))
    total = sum(weights.values())
    if not total:
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
        matched = sum(w for t, w in weights.items() if t in index.token_sets[i])
        hits.append(Hit(index.docs[i], float(scores[i]), matched / total))
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


def chapter_range_of(index: RagIndex, question: str) -> tuple[int, int] | None:
    """把问题里的章号解析成 (最小 seq, 最大 seq)，支持「从第X章到第Y章」。"""
    numbers = {cn_to_int(raw) for raw in CHAPTER_QUERY_RE.findall(question)}
    targets = {n for n in numbers if n is not None}
    if not targets:
        return None
    seqs: list[int] = []
    for seq, title in chapter_titles(index):
        match = CHAPTER_QUERY_RE.search(title)
        if match and cn_to_int(match.group(1)) in targets:
            seqs.append(seq)
    return (min(seqs), max(seqs)) if seqs else None


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
    """IDF 加权覆盖率 ≥ 阈值 → document；否则 llm。"""
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
