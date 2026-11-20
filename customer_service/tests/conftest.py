"""测试共享夹具：项目路径、微型索引。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from customer_service.retrieval.index import RagIndex, doc_tokens  # noqa: E402

_SAMPLE = [
    ("第一章", 0, "丁元英是私募基金的操盘手，后来隐居古城。"),
    ("第二章", 1, "芮小丹是古城的刑警，她和丁元英相识。"),
    ("第三章", 2, "欧阳雪经营饭店，是芮小丹的好朋友。"),
]


@pytest.fixture
def tiny_index() -> RagIndex:
    docs = [
        Document(
            page_content=text,
            metadata={"book": "遥远的救世主", "chapter": chapter,
                      "chapter_seq": seq, "chunk_index": 0},
        )
        for chapter, seq, text in _SAMPLE
    ]
    return RagIndex.create(docs, [doc_tokens(d) for d in docs])
