"""可信度测试：引文校验、专名拦截、降级修复与标注豁免。"""

from __future__ import annotations

from customer_service.agent.state import Evidence
from customer_service.policy.guard import repair_draft, verify


def _ev(text: str = "丁元英隐居古城。", chapter: str = "第一章", seq: int = 0,
        eid: str = "E1", tool: str = "search_book") -> Evidence:
    return Evidence(eid=eid, book="遥远的救世主", chapter=chapter, chapter_seq=seq,
                    chunk_index=0, text=text, score=1.0, coverage=1.0, source_tool=tool)


def test_valid_citation_passes():
    ev = [_ev()]
    vr = verify("丁元英隐居古城（《遥远的救世主》第一章）。", ev)
    assert vr.ok


def test_fake_citation_is_flagged():
    ev = [_ev()]
    vr = verify("丁元英隐居古城（《遥远的救世主》第九章）。", ev)
    assert not vr.ok
    assert vr.bad_citations


def test_unsupported_known_entity_is_flagged():
    ev = [_ev()]
    vr = verify("芮小丹也认识丁元英。", ev)
    assert "芮小丹" in vr.unsupported_entities


def test_repair_then_verify_passes():
    ev = [_ev()]
    draft = "丁元英住在123号。"
    vr = verify(draft, ev)
    assert not vr.ok
    repaired = repair_draft(draft, vr)
    assert "【大模型补充" in repaired
    assert verify(repaired, ev).ok


def test_exemption_targets_only_labeled_sentence():
    ev = [_ev()]
    draft = "丁元英隐居古城。他住在123号。"
    repaired = repair_draft(draft, verify(draft, ev))
    # 前一句有原文支撑，不被打标；后一句被打标
    assert repaired.startswith("丁元英隐居古城。")
    assert "他住在123号。【大模型补充" in repaired
    assert verify(repaired, ev).ok


def test_profile_evidence_supports_entity():
    ev = [_ev(text="【芮小丹】女主角，古城刑警", chapter="档案", seq=-1, tool="get_profile")]
    vr = verify("芮小丹是刑警。", ev)
    assert vr.ok
