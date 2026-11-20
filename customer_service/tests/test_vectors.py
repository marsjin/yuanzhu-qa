"""P2-3 向量版：RRF 融合与语义门控（用合成向量，不依赖真实模型）。"""

from __future__ import annotations

import numpy as np
from langchain_core.documents import Document

from customer_service.retrieval import embeddings
from customer_service.retrieval.index import (
    RagIndex,
    _rrf,
    decide_mode,
    doc_tokens,
    search,
)

QUERY = "丁元英怎么撤出公司"


def _index() -> RagIndex:
    rows = [
        "丁元英在这里说话。",
        "芮小丹办案。",
        "欧阳雪经营饭店。",
        "韩楚风在公司。",
        "私募基金宣布解散并进入清算。",
    ]
    docs = [
        Document(page_content=text,
                 metadata={"book": "测试", "chapter": f"第{i + 1}章",
                           "chapter_seq": i, "chunk_index": 0})
        for i, text in enumerate(rows)
    ]
    raw = np.array([
        [0.20, 0.98, 0.0],
        [0.10, 0.10, 0.99],
        [0.10, 0.20, 0.97],
        [0.30, 0.95, 0.0],
        [0.99, 0.14, 0.0],
    ], dtype="float32")
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return RagIndex.create(docs, [doc_tokens(d) for d in docs], raw)


def test_rrf_ranks_items_appearing_in_both_lists_first():
    fused = _rrf([[0, 1, 2], [1, 9, 2]])
    assert fused[0] == 1
    assert set(fused[:2]) == {1, 2}


def test_semantic_gate_unlocks_when_lexical_coverage_low(monkeypatch):
    """原句「撤出」在语料里不存在，但向量能命中讲解散的片段。"""
    index = _index()
    monkeypatch.setattr(embeddings, "available", lambda: True)
    monkeypatch.setattr(embeddings, "encode", lambda texts: np.array([[1.0, 0.0, 0.0]], dtype="float32"))

    hits = search(QUERY, index, coverage_query=QUERY)
    assert hits, "向量路应返回候选"
    top = hits[0]
    assert top.doc.metadata["chapter_seq"] == 4
    assert top.semantic > 0.9
    assert top.coverage < 0.5

    mode, _ = decide_mode(QUERY, hits)
    assert mode == "document"


def test_without_vectors_semantic_gate_stays_closed(monkeypatch):
    """向量不可用时退回纯 BM25，行为与改动前一致。"""
    index = _index()
    monkeypatch.setattr(embeddings, "available", lambda: False)
    hits = search(QUERY, index, coverage_query=QUERY)
    assert all(h.semantic == 0.0 for h in hits)
    mode, cov = decide_mode(QUERY, hits)
    assert mode == "llm"
    assert cov < 0.5
