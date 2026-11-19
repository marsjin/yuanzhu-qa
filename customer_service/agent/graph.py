"""Agent 图：9 节点 + 双环 + 四层防循环。

understand → plan → act → grade ─┬─ rewrite（环 A 补证据）
                                  ├─ synthesize → verify ─┬─ repair（环 B 修答案）
                                  └─ refuse → respond
"""

from __future__ import annotations

import re
from typing import Any, Callable, Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from customer_service.core import config as cfg
from customer_service.retrieval import index as ridx
from customer_service.retrieval import knowledge as kb
from customer_service.core import obs as log
from customer_service.policy import prompts as P
from customer_service.agent.state import (
    MAX_LLM_CALLS,
    MAX_REPAIR,
    MAX_RETRY,
    RECURSION_LIMIT,
    CustomerState,
    Evidence,
    ToolStep,
    inc_llm,
    inc_repair,
    inc_retry,
)
from customer_service.agent.tools import execute_step, set_index
from customer_service.policy.guard import VerifyResult, repair_draft, verify as verify_draft
from customer_service.generation.llm import build_llm, stream_collect


class PlanStep(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)


class PlanOut(BaseModel):
    steps: list[PlanStep]


def _budget_ok(s: CustomerState) -> bool:
    return s.get("llm_calls", 0) < MAX_LLM_CALLS


def _coverage_of(evidence: list[Evidence]) -> float:
    return max((e.coverage for e in evidence
                if e.source_tool in ("search_book", "plot_spine")), default=0.0)


def _aspect_coverage(evidence: list[Evidence], aspects: list[str]) -> tuple[float, list[str]]:
    if not aspects:
        return 1.0, []
    corpus = "\n".join(e.text for e in evidence)
    corpus_tokens = set(ridx.tokenize(corpus))
    missing: list[str] = []
    for a in aspects:
        tokens = ridx.tokenize(a)
        hit = any(t in corpus_tokens for t in tokens) if tokens else (a in corpus)
        if not hit:
            missing.append(a)
    return (len(aspects) - len(missing)) / len(aspects), missing


def _aspects_covered(evidence: list[Evidence], aspects: list[str], qtype: str) -> bool:
    ratio, _ = _aspect_coverage(evidence, aspects)
    if qtype in ("aggregate", "multi_hop"):
        return not aspects or ratio > 0
    return ratio >= 0.5


def _format_evidence(evidence: list[Evidence]) -> str:
    return "\n\n".join(f"[{e.eid}｜《{e.book}》{e.chapter}]\n{e.text}" for e in evidence)


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------
def understand_node(s: CustomerState) -> dict:
    q = s["question"]
    entities = s.get("entities") or kb.detect_entities(q)
    log.flow(f"→ understand｜实体={entities or '无'}")

    has_compare = bool(re.search(r"对比|比较|区别|差异|异同|相比", q)) and len(entities) >= 2
    has_plot = bool(re.search(r"情节|主线|脉络|梗概|讲了什么|故事线", q))
    has_agg = bool(re.search(r"分别|所有|哪些|各自|列举|统计|一共|几次|汇总", q))

    if has_compare:
        qtype: Literal["qa", "plot", "compare", "aggregate", "multi_hop"] = "compare"
        aspects = ["角色定位", "核心思想", "结局"]
    elif has_plot:
        qtype, aspects = "plot", []
    elif has_agg and entities:
        qtype, aspects = "aggregate", list(entities)
    elif has_agg:
        qtype, aspects = "aggregate", []
    elif len(entities) >= 2:
        qtype, aspects = "multi_hop", list(entities)
    else:
        qtype, aspects = "qa", []

    log.flow(f"  understand｜类型={qtype} 维度={aspects or '无'}")
    return {"entities": entities, "question_type": qtype, "aspects": aspects, "goal": q}


def plan_node(s: CustomerState) -> dict:
    q = s["question"]
    entities = s.get("entities", [])
    qtype = s.get("question_type", "qa")
    log.flow(f"→ plan｜类型={qtype}")

    def template_plan() -> list[ToolStep]:
        steps: list[ToolStep] = []
        if qtype == "compare" and len(entities) >= 2:
            for name in entities[:2]:
                steps.append(ToolStep("get_profile", {"name": name}))
                steps.append(ToolStep("search_book", {"query": f"{name} {q}", "k": 3}))
        elif qtype == "plot":
            steps.append(ToolStep("plot_spine", {"n": cfg.PLOT_SPINE_N}))
        elif qtype in ("aggregate", "multi_hop") and entities:
            for name in entities[:4]:
                steps.append(ToolStep("search_book", {"query": f"{name} {q}", "k": 3}))
        else:
            steps.append(ToolStep("search_book", {"query": q, "k": 5}))
        return steps

    if not _budget_ok(s):
        log.recovery(f"plan｜LLM 预算用尽，降级为模板计划")
        return {"plan": template_plan()}

    try:
        llm = build_llm().with_structured_output(PlanOut, method="json_mode")
        prompt = ChatPromptTemplate.from_messages([
            ("system", P.PLAN_SYSTEM),
            ("human", "目标：{goal}\n实体：{ents}\n类型：{qtype}\n维度：{aspects}\n只输出 JSON。"),
        ])
        calls = inc_llm(s)
        with log.timed("plan", calls, "生成工具计划"):
            out = (prompt | llm).invoke({
                "goal": s.get("goal", q), "ents": entities,
                "qtype": qtype, "aspects": s.get("aspects", []),
            })
        steps = [ToolStep(st.tool, st.args) for st in out.steps[:4]] or template_plan()
        log.flow(f"  plan｜计划={[st.tool for st in steps]}")
        return {"plan": steps, "llm_calls": calls}
    except Exception as e:
        log.recovery(f"plan｜LLM 失败（{type(e).__name__}），降级模板")
        return {"plan": template_plan()}


def act_node(s: CustomerState) -> dict:
    plan = s.get("plan", [])
    completed = set(s.get("completed", []))
    to_run = [st for st in plan if st.fingerprint not in completed]
    log.flow(f"→ act｜计划 {len(plan)} 步，执行 {len(to_run)} 步")

    new_evidence: list[Evidence] = []
    for st in to_run:
        new_evidence.extend(execute_step(st))
    log.flow(f"  act｜新增证据 {len(new_evidence)} 条")

    existing = s.get("evidence", [])
    offset = len(existing)
    for i, e in enumerate(new_evidence, 1):
        e.eid = f"E{offset + i}"

    update: dict[str, Any] = {
        "completed": sorted(completed | {st.fingerprint for st in to_run}),
        "evidence": new_evidence,
        "evidence_count_hist": s.get("evidence_count_hist", []) + [offset + len(new_evidence)],
    }
    if not to_run:
        log.recovery("act｜计划全部重复，强制 out_of_corpus")
        update["verdict"] = "out_of_corpus"
        return update

    hist = update["evidence_count_hist"]
    if len(hist) >= 2 and hist[-1] == hist[-2] and offset > 0:
        if any(st.tool == "search_book" for st in to_run):
            log.recovery("act｜证据零增长，强制 out_of_corpus")
            update["verdict"] = "out_of_corpus"
    return update


def grade_node(s: CustomerState) -> dict:
    if s.get("verdict") in ("out_of_corpus", "sufficient"):
        return {"verdict": s["verdict"]}

    evidence = s.get("evidence", [])
    aspects = s.get("aspects", [])
    qtype = s.get("question_type", "qa")
    retry = s.get("retry", 0)
    coverage = _coverage_of(evidence)
    ratio, missing = _aspect_coverage(evidence, aspects)

    if not evidence or coverage < cfg.OUT_OF_CORPUS_THRESHOLD:
        verdict = "out_of_corpus"
    elif coverage >= cfg.COVERAGE_THRESHOLD and _aspects_covered(evidence, aspects, qtype):
        verdict = "sufficient"
    else:
        verdict = "insufficient" if retry < MAX_RETRY else "out_of_corpus"

    aspect_info = f"，维度 {ratio:.0%}（缺 {'、'.join(missing)}）" if aspects else ""
    log.flow(f"→ grade｜证据 {len(evidence)} 条，覆盖率 {coverage:.2f}{aspect_info} "
             f"retry={retry}/{MAX_RETRY} → {verdict}")
    return {"verdict": verdict}


def rewrite_node(s: CustomerState) -> dict:
    retry = inc_retry(s)
    log.recovery(f"→ rewrite（环A 第 {retry}/{MAX_RETRY} 次）")

    if retry > MAX_RETRY:
        return {"retry": retry, "verdict": "out_of_corpus", "plan": []}

    q = s["question"]
    syn = {"自杀": "自尽 身亡", "死了": "去世 牺牲", "老板": "董事长 总经理"}
    expanded = q
    for k, v in syn.items():
        if k in expanded:
            expanded = expanded.replace(k, v)

    plan = [ToolStep("search_book", {"query": expanded, "k": 5})]
    for name in s.get("entities", [])[:4]:
        plan.append(ToolStep("search_book", {"query": f"{name} {expanded}", "k": 3}))

    completed = set(s.get("completed", []))
    if {st.fingerprint for st in plan}.issubset(completed):
        log.recovery("rewrite｜新计划全部重复，强制 out_of_corpus")
        return {"retry": retry, "verdict": "out_of_corpus", "plan": []}

    log.flow(f"  rewrite｜新计划={[st.tool for st in plan]}")
    return {"retry": retry, "plan": plan}


def synthesize_node(s: CustomerState) -> dict:
    evidence = s.get("evidence", [])
    profiles = "\n".join(e.text for e in evidence if e.chapter == "档案")
    context = _format_evidence([e for e in evidence if e.chapter != "档案"])
    if profiles:
        human_tpl = "【人物档案】\n{profiles}\n\n【证据片段】\n{context}\n\n【问题】\n{question}"
        payload = {"profiles": profiles, "context": context, "question": s["question"]}
    else:
        human_tpl = "【证据片段】\n{context}\n\n【问题】\n{question}"
        payload = {"context": context, "question": s["question"]}

    emit = s.get("emit")
    log.flow(f"→ synthesize｜证据 {len(evidence)} 条（流式）")
    with log.timed("synthesize", s.get("llm_calls", 0) + 1, "生成答案"):
        text = stream_collect(
            ChatPromptTemplate.from_messages([("system", P.SYNTHESIZE_SYSTEM),
                                              ("human", human_tpl)])
            | build_llm() | StrOutputParser(),
            payload,
            on_token=emit,
            echo=emit is None,
        )
    return {"draft": text.strip(), "llm_calls": inc_llm(s)}


def verify_node(s: CustomerState) -> dict:
    vr = verify_draft(s.get("draft", ""), s.get("evidence", []))
    if vr.ok:
        log.flow("→ verify｜通过")
    else:
        log.flow(f"→ verify｜未通过：{'、'.join(vr.reasons)}")
    return {"verify_result": {
        "ok": vr.ok, "bad_citations": vr.bad_citations, "missing_eids": vr.missing_eids,
        "unsupported_entities": vr.unsupported_entities,
        "unsupported_numbers": vr.unsupported_numbers, "reasons": vr.reasons,
    }}


def repair_node(s: CustomerState) -> dict:
    log.recovery(f"→ repair（环B 第 {s.get('repair_count', 0) + 1}/{MAX_REPAIR} 次）")
    data = s.get("verify_result", {})
    vr = VerifyResult(
        ok=data.get("ok", False),
        bad_citations=data.get("bad_citations", []),
        missing_eids=data.get("missing_eids", []),
        unsupported_entities=data.get("unsupported_entities", []),
        unsupported_numbers=data.get("unsupported_numbers", []),
    )
    return {"draft": repair_draft(s.get("draft", ""), vr), "repair_count": inc_repair(s)}


def refuse_node(s: CustomerState) -> dict:
    log.recovery("→ refuse｜原文无据，拒答并建议转人工")
    return {"draft": P.REFUSAL_TEXT, "handoff": True,
            "label": "来源：无（未在原文中找到）", "verdict": "out_of_corpus"}


def respond_node(s: CustomerState) -> dict:
    vr = s.get("verify_result", {})
    verdict = s.get("verdict", "out_of_corpus")
    if verdict == "out_of_corpus":
        label = "来源：无（未在原文中找到）"
    elif vr.get("ok", True):
        label = f"来源：《{cfg.BOOK_NAME}》原文"
    else:
        label = "来源：原文未覆盖 + 大模型补充（非原文内容）"
    log.flow(f"→ respond｜{label}｜LLM {s.get('llm_calls', 0)} 次")
    return {"final": s.get("draft", ""), "label": label,
            "handoff": s.get("handoff", False)}


# ---------------------------------------------------------------------------
# 图装配
# ---------------------------------------------------------------------------
def route_after_grade(s: CustomerState) -> str:
    verdict = s.get("verdict")
    if verdict == "sufficient":
        return "synthesize"
    if verdict == "out_of_corpus":
        return "refuse"
    return "rewrite"


def route_after_verify(s: CustomerState) -> str:
    if s.get("verify_result", {}).get("ok"):
        return "respond"
    if s.get("repair_count", 0) >= MAX_REPAIR:
        log.recovery("verify｜修复次数用尽，降级输出")
        return "respond"
    return "repair"


def build_graph() -> StateGraph:
    g = StateGraph(CustomerState)
    g.add_node("understand", understand_node)
    g.add_node("plan", plan_node)
    g.add_node("act", act_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("verify", verify_node)
    g.add_node("repair", repair_node)
    g.add_node("refuse", refuse_node)
    g.add_node("respond", respond_node)

    g.add_edge(START, "understand")
    g.add_edge("understand", "plan")
    g.add_edge("plan", "act")
    g.add_edge("act", "grade")
    g.add_conditional_edges("grade", route_after_grade,
                            {"rewrite": "rewrite", "synthesize": "synthesize", "refuse": "refuse"})
    g.add_edge("rewrite", "act")
    g.add_edge("synthesize", "verify")
    g.add_conditional_edges("verify", route_after_verify,
                            {"repair": "repair", "respond": "respond"})
    g.add_edge("repair", "verify")
    g.add_edge("refuse", "respond")
    g.add_edge("respond", END)
    return g


def run_agent(question: str, index: ridx.RagIndex, entities: list[str] | None = None,
              history: list | None = None,
              emit: Callable[[str], None] | None = None) -> dict:
    """运行 Agent 图；任何异常都降级为客服拒答，不返回 500。"""
    set_index(index)
    app = build_graph().compile()
    initial: CustomerState = {
        "question": question,
        "history": history or [],
        "entities": entities or kb.detect_entities(question),
        "evidence": [],
        "completed": [],
        "evidence_count_hist": [],
        "retry": 0,
        "repair_count": 0,
        "llm_calls": 0,
        "emit": emit,
    }
    try:
        result = app.invoke(initial, config={"recursion_limit": RECURSION_LIMIT})
        return {
            "text": result.get("final") or P.REFUSAL_TEXT,
            "label": result.get("label", "来源：无（未在原文中找到）"),
            "handoff": result.get("handoff", False),
            "verdict": result.get("verdict", "out_of_corpus"),
            "llm_calls": result.get("llm_calls", 0),
        }
    except Exception as e:
        log.recovery(f"run_agent｜图执行异常（{type(e).__name__}），降级拒答")
        return {"text": P.REFUSAL_TEXT, "label": "来源：无（未在原文中找到）",
                "handoff": True, "verdict": "out_of_corpus", "llm_calls": 0}
