"""购买引导：FAQ 命中购买意图时附带豆瓣链接，其他 FAQ 不带。"""

from __future__ import annotations

import pytest

from customer_service.app.pipeline import ChatPipeline
from customer_service.core import config as cfg
from customer_service.core.session import SessionStore

# 公开 CI 不配置真实密钥：依赖大模型的用例在无 Key 时跳过
LLM_AVAILABLE = bool(cfg.LLM_API_KEY)
requires_llm = pytest.mark.skipif(not LLM_AVAILABLE, reason="DEEPSEEK_API_KEY 未配置，跳过依赖大模型的用例")


@pytest.fixture
def pipeline(tiny_index, tmp_path) -> ChatPipeline:
    return ChatPipeline(tiny_index, SessionStore(tmp_path / "sessions.db"))


def test_buy_question_returns_douban_link(pipeline):
    r = pipeline.answer("s1", "这本书在哪里可以买到？")
    assert r.route == "faq"
    assert r.intent == "faq"
    assert len(r.links) == 1
    assert r.links[0]["label"] == cfg.BUY_LINK_LABEL
    url = r.links[0]["url"]
    assert url.startswith("https://book.douban.com/subject/1322455/?")
    assert f"from={cfg.BOOK_REFERRAL_TAG}" in url
    assert f"utm_source={cfg.BOOK_REFERRAL_TAG}" in url


def test_other_faq_has_no_link(pipeline):
    r = pipeline.answer("s1", "作者是谁？")
    assert r.route == "faq"
    assert r.links == []


@requires_llm
def test_book_question_has_no_link(pipeline):
    r = pipeline.answer("s1", "丁元英是谁")
    assert r.links == []
