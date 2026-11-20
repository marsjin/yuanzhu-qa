"""P1 结构导航测试：中文数字、章节定位、简介直出、章节证据封闭。"""

from __future__ import annotations

import pytest

from customer_service.generation import fast_path
from customer_service.policy.routing import detect_intent
from customer_service.retrieval.corpus import cn_to_int
from customer_service.retrieval.index import (
    chapter_chunks,
    chapter_seq_of,
    chapter_titles,
)


class _NoLLM:
    """任何模型调用都视为失败，用于验证 0 调用路径。"""

    def __getattr__(self, name):
        raise AssertionError(f"不应调用模型（{name}）")


@pytest.mark.parametrize(
    "text,want",
    [("26", 26), ("一", 1), ("十", 10), ("十一", 11), ("二十", 20),
     ("二十六", 26), ("四十六", 46), ("一百零三", 103)],
)
def test_cn_to_int(text, want):
    assert cn_to_int(text) == want


def test_cn_to_int_invalid():
    assert cn_to_int("abc") is None
    assert cn_to_int("") is None


def test_routing_chapter_and_intro():
    assert detect_intent("第一章内容", []) == "chapter"
    assert detect_intent("第二十六章讲了什么", []) == "chapter"
    assert detect_intent("本书简介", []) == "book_intro"
    assert detect_intent("介绍一下这本书", []) == "book_intro"


def test_routing_chapter_range_is_not_navigation():
    """跨章范围交给 Agent，不按单章处理。"""
    assert detect_intent("从第一章到第三章发生了什么", []) != "chapter"


def test_chapter_titles_order(tiny_index):
    assert [title for _, title in chapter_titles(tiny_index)] == ["第一章", "第二章", "第三章"]


def test_chapter_seq_of(tiny_index):
    assert chapter_seq_of(tiny_index, "第一章内容") == 0
    assert chapter_seq_of(tiny_index, "第三章讲了什么") == 2
    assert chapter_seq_of(tiny_index, "第99章内容") is None


def test_chapter_chunks_are_scoped(tiny_index):
    docs = chapter_chunks(tiny_index, 1)
    assert len(docs) == 1
    assert "芮小丹" in docs[0].page_content
    assert all(d.metadata["chapter_seq"] == 1 for d in docs)


def test_book_intro_without_llm(tiny_index):
    result = fast_path.answer("本书简介", tiny_index, _NoLLM(), [], "book_intro", [])
    assert "遥远的救世主" in result.text
    assert "第一章" in result.text
    assert result.mode == "document"


def test_chapter_excerpt_without_llm(tiny_index):
    result = fast_path.answer("第一章原文", tiny_index, _NoLLM(), [], "chapter", [])
    assert "私募基金" in result.text
    assert result.citations == ["《遥远的救世主》第一章"]


def test_chapter_out_of_range(tiny_index):
    result = fast_path.answer("第99章内容", tiny_index, _NoLLM(), [], "chapter", [])
    assert "未找到" in result.text
    assert result.mode == "refusal"
