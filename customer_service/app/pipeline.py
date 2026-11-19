"""编排层：会话 → 意图路由 → 快路径 / Agent → 持久化。"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from customer_service.retrieval import index as ridx
from customer_service.retrieval import knowledge as kb
from customer_service.core import obs as log
from customer_service.policy import prompts as P
from customer_service.agent.graph import run_agent
from customer_service.generation.fast_path import answer as fast_answer
from customer_service.generation.llm import build_llm
from customer_service.policy.routing import detect_intent, has_pronoun, needs_agent
from customer_service.core.session import SessionStore

_CITE_FMT = re.compile(r"《([^》]+)》第([一二三四五六七八九十百零两0-9]+)章")


@dataclass
class AnswerResult:
    text: str
    label: str
    route: str                 # fast / agent / faq / handoff
    intent: str
    handoff: bool = False
    citations: list[str] = field(default_factory=list)
    latency_ms: int = 0
    trace_id: str = ""
    session_id: str = ""


class ChatPipeline:
    def __init__(self, index: ridx.RagIndex, store: SessionStore | None = None,
                 llm=None) -> None:
        self.index = index
        self.store = store or SessionStore()
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            self._llm = build_llm()
        return self._llm

    def answer(self, session_id: str, message: str,
               on_token: Callable[[str], None] | None = None) -> AnswerResult:
        t0 = time.time()
        trace_id = uuid.uuid4().hex[:8]
        log.set_trace(trace_id)
        log.flow(f"chat｜session={session_id}｜问题「{message[:40]}」")

        history_dicts = self.store.history(session_id)
        history: list[BaseMessage] = [
            HumanMessage(content=h["content"]) if h["role"] == "user"
            else AIMessage(content=h["content"])
            for h in history_dicts
        ]
        prev_question, last_entities = self.store.last_state(session_id)

        entities = kb.detect_entities(message)
        if not entities and has_pronoun(message) and last_entities:
            entities = last_entities
            log.flow(f"  resolve｜指代补全实体={entities}")

        intent = detect_intent(message, entities)
        log.flow(f"  route｜intent={intent}")

        if intent == "handoff":
            result = AnswerResult(P.HANDOFF_TEXT, "来源：人工客服", "handoff", intent,
                                  handoff=True, trace_id=trace_id, session_id=session_id)
            self._persist(session_id, message, result.text, entities)
            return self._finish(result, t0)

        if intent == "faq":
            entry = kb.match_faq(message) or {}
            text = entry.get("answer", P.REFUSAL_TEXT)
            result = AnswerResult(text, "来源：客服知识库", "faq", intent,
                                  trace_id=trace_id, session_id=session_id)
            self._persist(session_id, message, result.text, entities)
            return self._finish(result, t0)

        if needs_agent(message, entities):
            log.flow("  route｜命中 Agent 触发条件")
            emitted = [0]

            def emit(tok: str) -> None:
                emitted[0] += 1
                if on_token is not None:
                    on_token(tok)

            out = run_agent(message, self.index, entities=entities,
                            history=history, emit=emit)
            if on_token is not None and emitted[0] == 0:
                on_token(out["text"])
            result = AnswerResult(
                text=out["text"], label=out["label"], route="agent", intent=intent,
                handoff=out.get("handoff", False),
                citations=_extract_citations(out["text"]),
                trace_id=trace_id, session_id=session_id,
            )
        else:
            retrieval_query = f"{prev_question} {message}" if prev_question else message
            fast = fast_answer(message, self.index, self.llm, history, intent, entities,
                               retrieval_query=retrieval_query, on_token=on_token)
            result = AnswerResult(
                text=fast.text, label=fast.label, route="fast", intent=intent,
                handoff=fast.handoff, citations=fast.citations,
                trace_id=trace_id, session_id=session_id,
            )

        self._persist(session_id, message, result.text, entities)
        return self._finish(result, t0)

    def _persist(self, session_id: str, question: str, answer: str, entities: list[str]) -> None:
        self.store.append(session_id, "user", question)
        self.store.append(session_id, "assistant", answer)
        self.store.save_state(session_id, question, entities)

    def _finish(self, result: AnswerResult, t0: float) -> AnswerResult:
        result.latency_ms = int((time.time() - t0) * 1000)
        log.flow(f"done｜route={result.route}｜{result.latency_ms}ms｜handoff={result.handoff}")
        return result


def _extract_citations(text: str) -> list[str]:
    return [f"《{b}》第{c}章" for b, c in _CITE_FMT.findall(text)]
