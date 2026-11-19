"""路由测试：五类意图与 Agent 升级条件。"""

from __future__ import annotations

from customer_service.policy.routing import detect_intent, has_pronoun, needs_agent


def test_handoff_intent():
    assert detect_intent("我要转人工", []) == "handoff"
    assert detect_intent("怎么申请退款", []) == "handoff"


def test_faq_intent():
    assert detect_intent("这本书在哪里买？", []) == "faq"


def test_plot_intent():
    assert detect_intent("这本书的情节主线是什么？", []) == "plot"


def test_compare_intent_requires_two_entities():
    assert detect_intent("对比丁元英和芮小丹", ["丁元英", "芮小丹"]) == "compare"
    assert detect_intent("对比丁元英", ["丁元英"]) == "book_qa"


def test_default_book_qa():
    assert detect_intent("丁元英是谁？", ["丁元英"]) == "book_qa"


def test_needs_agent_on_aggregate_words():
    assert needs_agent("分别介绍书中的几个主要人物", [])
    assert needs_agent("从第一章到第十章发生了什么", [])


def test_needs_agent_on_three_entities():
    assert needs_agent("丁元英和芮小丹和欧阳雪有什么关系", ["丁元英", "芮小丹", "欧阳雪"])


def test_needs_agent_false_for_simple_question():
    assert not needs_agent("丁元英是谁？", ["丁元英"])


def test_pronoun_detection():
    assert has_pronoun("她后来怎么样了？")
    assert not has_pronoun("芮小丹后来怎么样了？")
