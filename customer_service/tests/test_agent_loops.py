"""Agent 防循环测试：指纹去重、空转判定、条件边走向。"""

from __future__ import annotations

from customer_service.agent import graph as G
from customer_service.agent.state import Evidence, ToolStep, merge_evidence
from customer_service.agent.tools import set_index


def test_tool_step_fingerprint_stable():
    a = ToolStep("search_book", {"query": "丁元英", "k": 3})
    b = ToolStep("search_book", {"k": 3, "query": "丁元英"})
    c = ToolStep("search_book", {"query": "芮小丹", "k": 3})
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint


def test_merge_evidence_dedupes():
    e1 = Evidence("E1", "遥远的救世主", "第一章", 0, 0, "a", 1.0, 1.0, "search_book")
    e2 = Evidence("E2", "遥远的救世主", "第一章", 0, 0, "a", 1.0, 1.0, "search_book")
    merged = merge_evidence([e1], [e2])
    assert len(merged) == 1


def test_act_dedup_forces_out_of_corpus(tiny_index):
    set_index(tiny_index)
    step = ToolStep("search_book", {"query": "丁元英", "k": 2})
    out = G.act_node({
        "question": "丁元英是谁", "plan": [step], "completed": [step.fingerprint],
        "evidence": [], "evidence_count_hist": [],
    })
    assert out["verdict"] == "out_of_corpus"
    assert out["evidence"] == []


def test_act_executes_new_plan(tiny_index):
    set_index(tiny_index)
    step = ToolStep("search_book", {"query": "丁元英", "k": 2})
    out = G.act_node({
        "question": "丁元英是谁", "plan": [step], "completed": [],
        "evidence": [], "evidence_count_hist": [],
    })
    assert out["evidence"]
    assert out["evidence"][0].eid == "E1"
    assert "verdict" not in out


def test_route_after_grade():
    assert G.route_after_grade({"verdict": "sufficient"}) == "synthesize"
    assert G.route_after_grade({"verdict": "out_of_corpus"}) == "refuse"
    assert G.route_after_grade({"verdict": "insufficient", "retry": 0}) == "rewrite"


def test_route_after_verify():
    assert G.route_after_verify({"verify_result": {"ok": True}}) == "respond"
    assert G.route_after_verify(
        {"verify_result": {"ok": False}, "repair_count": 0}) == "repair"
    assert G.route_after_verify(
        {"verify_result": {"ok": False}, "repair_count": 1}) == "respond"


def test_understand_classifies_compare(tiny_index):
    out = G.understand_node({"question": "对比丁元英和芮小丹"})
    assert out["question_type"] == "compare"
    assert out["entities"] == ["丁元英", "芮小丹"]


def test_understand_classifies_plot(tiny_index):
    out = G.understand_node({"question": "这本书的情节主线是什么"})
    assert out["question_type"] == "plot"
