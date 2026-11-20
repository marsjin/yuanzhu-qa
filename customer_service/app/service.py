"""HTTP 服务：POST /chat（JSON 或 SSE 流式）、GET /health、POST /session/{id}/reset。

启动：
    uvicorn customer_service.app.service:app --reload
"""

from __future__ import annotations

import base64
import binascii
import json
import queue
import secrets
import threading
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from customer_service.core import config as cfg
from customer_service.retrieval import index as ridx
from customer_service.app.pipeline import AnswerResult, ChatPipeline
from customer_service.core.session import SessionStore

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    idx = ridx.build_index()
    _state["pipeline"] = ChatPipeline(idx, SessionStore())
    _state["stats"] = ridx.stats(idx)
    yield
    _state.clear()


app = FastAPI(title=f"《{cfg.BOOK_NAME}》智能客服", version="0.1.0", lifespan=lifespan)


def _valid_basic_auth(header: str | None) -> bool:
    if not header:
        return False
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    username, separator, password = decoded.partition(":")
    return bool(separator) and secrets.compare_digest(
        username.encode("utf-8"), cfg.AUTH_USERNAME.encode("utf-8")
    ) and secrets.compare_digest(
        password.encode("utf-8"), cfg.AUTH_PASSWORD.encode("utf-8")
    )


@app.middleware("http")
async def require_auth(request: Request, call_next):
    if cfg.AUTH_ENABLED and not _valid_basic_auth(request.headers.get("Authorization")):
        return JSONResponse(
            status_code=401,
            content={"detail": "需要鉴权"},
            headers={"WWW-Authenticate": 'Basic realm="customer-service", charset="UTF-8"'},
        )
    return await call_next(request)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, description="会话 ID，用于多轮上下文")
    message: str = Field(min_length=1, description="用户问题")
    stream: bool = Field(default=False, description="true 时返回 SSE 事件流")


class ChatResponse(BaseModel):
    text: str
    label: str
    route: str
    intent: str
    handoff: bool
    citations: list[str]
    links: list[dict] = Field(default_factory=list)
    latency_ms: int
    trace_id: str
    session_id: str


class ResetResponse(BaseModel):
    ok: bool
    session_id: str


class EventRequest(BaseModel):
    type: str = Field(min_length=1, max_length=32, description="事件类型，如 buy_click")
    url: str = Field(default="", max_length=512, description="点击的外链")
    session_id: str = Field(default="", max_length=64, description="会话 ID")


def _pipeline() -> ChatPipeline:
    return _state["pipeline"]


def _to_response(r: AnswerResult) -> ChatResponse:
    return ChatResponse(
        text=r.text, label=r.label, route=r.route, intent=r.intent, handoff=r.handoff,
        citations=r.citations, links=r.links, latency_ms=r.latency_ms, trace_id=r.trace_id,
        session_id=r.session_id,
    )


def _event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse(pipeline: ChatPipeline, req: ChatRequest) -> Iterator[str]:
    """后台线程跑完整流程，主线程把 token 推给客户端。"""
    q: queue.Queue[tuple[str, str | None]] = queue.Queue()
    holder: dict = {}

    def emit(token: str) -> None:
        q.put(("delta", token))

    def run() -> None:
        try:
            holder["result"] = pipeline.answer(req.session_id, req.message, on_token=emit)
        except Exception as e:  # 兜底：不让单个请求打断事件流
            holder["error"] = f"{type(e).__name__}: {e}"
        finally:
            q.put(("end", None))

    threading.Thread(target=run, daemon=True).start()

    while True:
        kind, payload = q.get()
        if kind == "end":
            break
        yield _event({"type": "delta", "content": payload})

    result = holder.get("result")
    if result is None:
        yield _event({"type": "error", "message": holder.get("error", "unknown error")})
        return
    yield _event({
        "type": "done", "text": result.text, "label": result.label,
        "route": result.route, "intent": result.intent, "handoff": result.handoff,
        "citations": result.citations, "links": result.links, "latency_ms": result.latency_ms,
        "trace_id": result.trace_id, "session_id": result.session_id,
    })


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    pipeline = _pipeline()
    if req.stream:
        return StreamingResponse(_sse(pipeline, req), media_type="text/event-stream")
    return _to_response(pipeline.answer(req.session_id, req.message))


@app.post("/event")
def log_event(req: EventRequest) -> dict:
    _pipeline().store.log_event(req.type, req.session_id, req.url)
    return {"ok": True}


@app.get("/events/stats")
def event_stats(days: int = 7) -> dict:
    return {"days": days, "counts": _pipeline().store.event_stats(days)}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "book": cfg.BOOK_NAME,
        "index": _state.get("stats", {}),
        "llm_configured": bool(cfg.LLM_API_KEY),
        "cover": cfg.BOOK_COVER_PATH if cfg.BOOK_COVER_FILE.exists() else None,
        "douban": cfg.douban_url(),
    }


@app.post("/session/{session_id}/reset", response_model=ResetResponse)
def reset_session(session_id: str) -> ResetResponse:
    _pipeline().store.reset(session_id)
    return ResetResponse(ok=True, session_id=session_id)


STATIC_DIR = cfg.BASE_DIR / "app" / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def web_chat() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
