"""意图路由：handoff / faq / plot / compare / book_qa，以及是否升级 Agent。"""

from __future__ import annotations

import re
from typing import Literal

from customer_service.retrieval import knowledge as kb

Intent = Literal["handoff", "faq", "plot", "compare", "book_qa"]

HANDOFF_RE = re.compile(r"转人工|人工客服|真人客服|投诉|退款|退货|订单|物流|发票|客服电话")
PLOT_RE = re.compile(r"情节|主线|脉络|梗概|故事发展|讲了什么|来龙去脉|故事线")
COMPARE_RE = re.compile(r"对比|比较|区别|差异|异同|相比|vs|VS")

# 需要多步推理的问题：交给 Agent
AGENT_TRIGGERS = re.compile(
    r"分别|所有|哪些|各自|列举|统计|一共|几次|排序|汇总|"
    r"从第[一二三四五六七八九十百零两0-9]+章到|"
    r"第[一二三四五六七八九十百零两0-9]+章到第[一二三四五六七八九十百零两0-9]+章|"
    r"为什么.*又|三者|三人"
)

# 指代词：命中且本轮没有实体时，用上一轮实体补全
PRONOUN_RE = re.compile(r"^(她|他|它|这个人|那个人|那本书|这本书|该人物)")


def detect_intent(question: str, entities: list[str]) -> Intent:
    """规则意图识别，顺序：转人工 → FAQ → 情节 → 对比 → 默认。"""
    if HANDOFF_RE.search(question):
        return "handoff"
    if kb.match_faq(question) is not None:
        return "faq"
    if COMPARE_RE.search(question) and len(entities) >= 2:
        return "compare"
    if PLOT_RE.search(question):
        return "plot"
    return "book_qa"


def needs_agent(question: str, entities: list[str]) -> bool:
    """聚合 / 多跳 / 章节范围 / 3 个以上实体 → Agent 路径。"""
    if AGENT_TRIGGERS.search(question):
        return True
    return len(entities) >= 3


def has_pronoun(question: str) -> bool:
    return bool(PRONOUN_RE.search(question.strip()))
