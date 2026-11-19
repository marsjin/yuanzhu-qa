"""可信度校验：纯代码比对引文 / 引用编号 / 专名 / 数字，并做降级修复。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from customer_service.core import config as cfg
from customer_service.retrieval import knowledge as kb
from customer_service.agent.state import Evidence

CITE_RE = re.compile(r"《([^》]+)》第([一二三四五六七八九十百零两0-9]+)章")
EID_RE = re.compile(r"\[E(\d+)\]")
NUMBER_RE = re.compile(r"\d+")

LABEL_MARKER = "【大模型补充"


@dataclass
class VerifyResult:
    ok: bool
    bad_citations: list[str] = field(default_factory=list)
    missing_eids: list[str] = field(default_factory=list)
    unsupported_entities: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        r: list[str] = []
        if self.bad_citations:
            r.append(f"假引文 {len(self.bad_citations)} 处")
        if self.missing_eids:
            r.append(f"无效引用 {len(self.missing_eids)} 处")
        if self.unsupported_entities:
            r.append(f"无支撑专名 {len(self.unsupported_entities)} 处")
        if self.unsupported_numbers:
            r.append(f"无支撑数字 {len(self.unsupported_numbers)} 处")
        return r


def _sources(evidence: list[Evidence]) -> set[tuple[str, str]]:
    return {(e.book, e.chapter) for e in evidence}


def _corpus(evidence: list[Evidence]) -> str:
    return "\n".join(e.text for e in evidence)


def _known_entities() -> set[str]:
    names = set(kb.CHARACTERS.keys())
    aliases = set(kb.NAME_ALIASES.keys())
    tags = {t for info in kb.CHARACTERS.values() for t in info["tags"]}
    return names | aliases | tags | {cfg.BOOK_NAME}


def _labeled_sentence_keys(draft: str) -> set[str]:
    """找【大模型补充】标注句里的实体/数字，作为二次校验的豁免名单。"""
    out: set[str] = set()
    pos = 0
    while True:
        idx = draft.find(LABEL_MARKER, pos)
        if idx == -1:
            break
        nearest = max(draft.rfind(p, 0, idx) for p in ("。", "！", "？"))
        if nearest == -1:
            sent_start = 0
        else:
            prev = max(draft.rfind(p, 0, nearest) for p in ("。", "！", "？"))
            sent_start = prev + 1 if prev != -1 else 0
        segment = draft[sent_start:idx]
        for ent in _known_entities():
            if len(ent) >= 2 and ent in segment:
                out.add(ent)
        out.update(NUMBER_RE.findall(segment))
        pos = idx + len(LABEL_MARKER)
    return out


def verify(draft: str, evidence: list[Evidence]) -> VerifyResult:
    """校验草稿；返回问题清单。"""
    if not draft.strip():
        return VerifyResult(ok=False, bad_citations=["空回答"])

    sources = _sources(evidence)
    corpus = _corpus(evidence)
    eids = {e.eid for e in evidence}
    labeled = _labeled_sentence_keys(draft)
    result = VerifyResult(ok=True)

    for book, chapter in CITE_RE.findall(draft):
        matched = any(
            b == book and (chapter in ch or chapter == ch.replace("第", "").replace("章", ""))
            for b, ch in sources
        )
        if not matched:
            result.bad_citations.append(f"《{book}》第{chapter}章")
            result.ok = False

    for eid in EID_RE.findall(draft):
        if f"E{eid}" not in eids:
            result.missing_eids.append(f"E{eid}")
            result.ok = False

    draft_text = CITE_RE.sub("", draft)
    for ent in _known_entities():
        if len(ent) < 2 or ent not in draft_text or ent in corpus:
            continue
        profile_supported = any(e.chapter == "档案" and ent in e.text for e in evidence)
        if not profile_supported and ent not in labeled:
            result.unsupported_entities.append(ent)
            result.ok = False

    for num in NUMBER_RE.findall(draft):
        if num not in corpus and num not in labeled:
            result.unsupported_numbers.append(num)
            result.ok = False

    return result


def repair_draft(draft: str, vr: VerifyResult) -> str:
    """降级修复：假引文改「出处存疑」，无支撑句尾加标注。只做减法。"""
    text = draft
    for cite in vr.bad_citations:
        text = text.replace(cite, "（出处存疑）")

    if vr.unsupported_entities or vr.unsupported_numbers:
        bad_tokens = set(vr.unsupported_entities) | set(vr.unsupported_numbers)
        sentences = re.split(r"(?<=[。！？])", text)
        repaired: list[str] = []
        for s in sentences:
            if any(tok in s for tok in bad_tokens) and LABEL_MARKER not in s:
                repaired.append(s.rstrip() + "【大模型补充，非原文内容】")
            else:
                repaired.append(s)
        text = "".join(repaired)
    return text
