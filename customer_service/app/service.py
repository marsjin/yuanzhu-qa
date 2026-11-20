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
from urllib.parse import parse_qs, quote

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from customer_service.core import auth
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


ADMIN_ONLY_PREFIXES = ("/admin", "/events/stats")
PUBLIC_PATHS = ("/login", "/logout", "/favicon.ico")


def _auth_identity(header: str | None) -> tuple[str, str] | None:
    """解析 Basic 凭证，返回 (用户名, 角色)。账号与口令固定来自配置文件。"""
    if not header:
        return None
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic":
        return None
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None
    username, separator, password = decoded.partition(":")
    if not separator:
        return None

    user = cfg.auth_users().get(username)
    if not user:
        return None
    expected, role = user
    if secrets.compare_digest(password.encode("utf-8"), expected.encode("utf-8")):
        return (username, role)
    return None


def _auth_role(header: str | None) -> str | None:
    identity = _auth_identity(header)
    return identity[1] if identity else None


def _safe_next(path: str) -> str:
    """只允许站内相对路径，避免登录后被跳到外站。"""
    if path.startswith("/") and not path.startswith("//"):
        return path
    return "/"


@app.middleware("http")
async def require_auth(request: Request, call_next):
    path = request.url.path
    if not cfg.AUTH_ENABLED or path in PUBLIC_PATHS or path.startswith("/static/"):
        return await call_next(request)

    if "text/html" in request.headers.get("accept", ""):
        # 浏览器页面只认登录 Cookie：退出后即失效，不会被缓存的 Basic 凭证绕过
        identity = auth.verify(request.cookies.get(cfg.AUTH_COOKIE_NAME))
        if identity is None:
            target = "/login" + (f"?next={quote(path, safe='/')}" if path != "/" else "")
            return RedirectResponse(target, status_code=303)
    else:
        # 登录 Cookie 优先：避免浏览器缓存的 Basic 凭证覆盖当前登录身份
        identity = auth.verify(request.cookies.get(cfg.AUTH_COOKIE_NAME)) or _auth_identity(
            request.headers.get("Authorization")
        )
        if identity is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "需要鉴权"},
                headers={"WWW-Authenticate": 'Basic realm="customer-service", charset="UTF-8"'},
            )

    if identity[1] != "admin" and path.startswith(ADMIN_ONLY_PREFIXES):
        return JSONResponse(status_code=403, content={"detail": "需要管理员权限"})
    request.state.auth = identity
    return await call_next(request)


@app.get("/me")
def current_user(request: Request) -> dict:
    """当前登录用户，供页面展示；未启用鉴权时返回 open。"""
    identity = getattr(request.state, "auth", None)
    if not identity:
        return {"username": "", "role": "open"}
    username, role = identity
    return {"username": username, "role": role}


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
    type: str = Field(min_length=1, max_length=32, description="事件类型，如 impression_answer / click_answer")
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
    counts = _pipeline().store.event_stats(days)

    def _rate(click: str, impression: str) -> float:
        shown = counts.get(impression, 0)
        return round(counts.get(click, 0) / shown, 4) if shown else 0.0

    return {
        "days": days,
        "counts": counts,
        "impressions": {
            "default": counts.get("impression_default", 0),
            "answer": counts.get("impression_answer", 0),
        },
        "clicks": {
            "default": counts.get("click_default", 0),
            "answer": counts.get("click_answer", 0),
        },
        "ctr": {
            "default": _rate("click_default", "impression_default"),
            "answer": _rate("click_answer", "impression_answer"),
        },
    }


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


@app.get("/login", include_in_schema=False)
def login_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "login.html")


@app.post("/login", include_in_schema=False)
async def login_submit(request: Request):
    body = (await request.body()).decode("utf-8", "ignore")
    form = {key: values[0] for key, values in parse_qs(body).items()}
    username = form.get("username", "").strip()
    password = form.get("password", "")
    next_path = _safe_next(form.get("next", "/"))

    user = cfg.auth_users().get(username)
    if not user or not secrets.compare_digest(password.encode("utf-8"), user[0].encode("utf-8")):
        return RedirectResponse(
            f"/login?error=1&next={quote(next_path, safe='/')}", status_code=303
        )

    response = RedirectResponse(next_path, status_code=303)
    response.set_cookie(
        cfg.AUTH_COOKIE_NAME,
        auth.issue(username, user[1]),
        max_age=cfg.AUTH_SESSION_MINUTES * 60,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return response


@app.get("/logout", include_in_schema=False)
def logout() -> RedirectResponse:
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(cfg.AUTH_COOKIE_NAME, path="/")
    return response


@app.get("/", include_in_schema=False)
def web_chat() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin", include_in_schema=False)
def web_admin() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin.html")
