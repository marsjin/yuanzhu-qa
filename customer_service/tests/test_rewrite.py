"""P2-3（LLM 查询改写）：覆盖率不足时用改写词二次检索。"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from customer_service.core import config as cfg
from customer_service.generation import fast_path
from customer_service.retrieval.index import RagIndex, doc_tokens


class _ScriptedLLM(BaseChatModel):
    """按顺序返回预设回复；第一次用于改写，第二次用于作答。"""

    replies: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = self.replies.pop(0) if self.replies else "（无回复）"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(text))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        text = self.replies.pop(0) if self.replies else "（无回复）"
        for char in text:
            yield ChatGenerationChunk(message=AIMessageChunk(content=char))


def _index() -> RagIndex:
    rows = [
        "丁元英操盘的私募基金在柏林宣布解散，随后进入清算程序。",
        "芮小丹在古城当刑警，与丁元英相识。",
        "欧阳雪在古城经营一家饭店。",
        "韩楚风在北京主持公司事务。",
        "王庙村的农户开始生产音箱。",
    ]
    docs = [
        Document(
            page_content=text,
            metadata={"book": "测试", "chapter": f"第{i + 1}章",
                      "chapter_seq": i, "chunk_index": 0},
        )
        for i, text in enumerate(rows)
    ]
    return RagIndex.create(docs, [doc_tokens(d) for d in docs])


def test_rewrite_unlocks_insufficient_coverage(monkeypatch):
    """「退出」在原文里没有，改写为「解散 清算」后应命中。"""
    monkeypatch.setattr(cfg, "QUERY_REWRITE", True)
    llm = _ScriptedLLM(replies=["解散 清算", "丁元英的私募基金解散了（《测试》第一章）"])
    result = fast_path.answer("丁元英退出基金公司", _index(), llm, [], "book_qa", ["丁元英"])
    assert result.mode == "document"
    assert "解散" in result.text


def test_rewrite_disabled_keeps_refusal(monkeypatch):
    """关闭改写后，同一问题应维持拒答。"""
    monkeypatch.setattr(cfg, "QUERY_REWRITE", False)
    llm = _ScriptedLLM(replies=["不应被调用"])
    result = fast_path.answer("丁元英退出基金公司", _index(), llm, [], "book_qa", ["丁元英"])
    assert result.mode == "refusal"
    assert result.handoff


def test_compact_hits_dedupes_by_chunk():
    """合并两轮检索时按章节+片段去重。"""
    index = _index()
    hits = fast_path.retrieve("丁元英", "丁元英", "book_qa", ["丁元英"], index)
    merged = fast_path._merge_hits(hits, hits)
    assert len(merged) == len(hits)
