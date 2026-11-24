"""HTTP 服务：POST /chat（JSON 或 SSE 流式）、GET /health、POST /session/{id}/reset。

启动：
    uvicorn customer_service.app.service:app --reload
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import queue
import re
import secrets
import threading
import time
from contextlib import asynccontextmanager
from typing import Iterator
from urllib.parse import parse_qs, quote, urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from customer_service.core import auth
from customer_service.core import config as cfg
from customer_service.core import obs as log
from customer_service.retrieval import index as ridx
from customer_service.app.pipeline import AnswerResult, ChatPipeline
from customer_service.core.session import SessionStore, local_day

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    idx = ridx.build_index()
    _state["pipeline"] = ChatPipeline(idx, SessionStore())
    _state["stats"] = ridx.stats(idx)
    yield
    _state.clear()


app = FastAPI(title=f"《{cfg.BOOK_NAME}》智能客服", version="0.1.0", lifespan=lifespan)


ADMIN_ONLY_PREFIXES = ("/admin", "/events/stats", "/visits")
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
            # 保留 query，避免登录回来后丢失 ?from=... 等来源参数
            target_path = path + (f"?{request.url.query}" if request.url.query else "")
            target = "/login" + (
                f"?next={quote(target_path, safe='/')}" if target_path != "/" else ""
            )
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


# ---- 访问来源记录：外层中间件，必须注册在 require_auth 之后（Starlette 后注册先执行）----
VISIT_SKIP_PATHS = ("/login", "/logout", "/favicon.ico", "/health", "/event", "/admin")
VISIT_SKIP_PREFIXES = ("/static/", "/visits", "/docs", "/redoc", "/openapi.json")
_last_purge = {"ts": 0.0}


def _visit_store() -> SessionStore | None:
    """测试未进入 lifespan 时 _state 为空，此处必须容错。"""
    pipeline = _state.get("pipeline")
    return pipeline.store if pipeline is not None else None


def _is_visit_request(request: Request) -> bool:
    if request.method != "GET" or request.url.path != "/":
        return False
    if "text/html" not in request.headers.get("accept", ""):
        return False
    dest = request.headers.get("sec-fetch-dest")
    if dest and dest != "document":
        return False
    mode = request.headers.get("sec-fetch-mode")
    if mode and mode != "navigate":
        return False
    return True


def _slug(raw: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", (raw or "").strip().lower()).strip("-")
    return cleaned[: cfg.VISIT_SOURCE_MAX_LEN]


def _normalize_source(raw: str) -> str:
    text = (raw or "").strip().lower()
    slug = _slug(text)
    if not slug:
        return ""
    if slug == "direct":
        return "direct"
    if slug in ("github", "gh"):
        return "github_unknown"
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", text):
        return slug
    return f"other:{slug}"[:64]


def _visit_source(request: Request) -> tuple[str, str, str, str]:
    """from > utm_source > Referer 主机 > direct；只把 Referer 主机落库。"""
    query = parse_qs(request.url.query)

    def first(key: str) -> str:
        return (query.get(key) or [""])[0]

    medium = first("utm_medium")[:64]
    campaign = first("utm_campaign")[:64]
    source = _normalize_source(first("from") or first("utm_source"))
    referer = request.headers.get("referer", "")
    referer_host = (urlsplit(referer).hostname or "").lower()[:128]
    if not source:
        own_host = (request.url.hostname or "").lower()
        if not referer_host or referer_host == own_host:
            source = "direct"
        elif referer_host == "github.com" or referer_host.endswith(".github.com"):
            source = "github_unknown"
        else:
            source = f"other:{referer_host}"[:64]
    return source, medium, campaign, referer_host


def _ua_family(ua: str) -> str:
    low = ua.lower()
    if not low:
        return ""
    if "micromessenger" in low:
        return "WeChat"
    if "edg/" in low:
        return "Edge"
    if "chrome" in low or "crios" in low:
        return "Chrome"
    if "firefox" in low or "fxios" in low:
        return "Firefox"
    if "safari" in low:
        return "Safari"
    return "Other"


def _is_bot(ua: str) -> bool:
    low = ua.lower()
    return any(keyword in low for keyword in cfg.VISIT_BOT_UA)


def _visitor_hash(request: Request) -> str:
    ip = request.client.host if request.client else ""
    ua = request.headers.get("user-agent", "")
    digest = hmac.new(
        cfg.VISIT_HASH_KEY.encode("utf-8"), f"{ip}|{ua}".encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return digest[:32]


@app.middleware("http")
async def record_visit(request: Request, call_next):
    path = request.url.path
    if (
        not cfg.VISIT_ENABLED
        or path in VISIT_SKIP_PATHS
        or path.startswith(VISIT_SKIP_PREFIXES)
        or not _is_visit_request(request)
    ):
        return await call_next(request)

    store = _visit_store()
    if store is None:
        return await call_next(request)

    visitor_id = request.cookies.get(cfg.VISITOR_COOKIE_NAME, "")
    generated = not visitor_id
    if generated:
        visitor_id = secrets.token_urlsafe(16)

    response = await call_next(request)

    try:
        now = time.time()
        identity = auth.verify(request.cookies.get(cfg.AUTH_COOKIE_NAME))
        username, role = identity if identity else ("", "")
        source, medium, campaign, referer_host = _visit_source(request)
        ua = request.headers.get("user-agent", "")
        visit_id = store.log_visit(
            visited_at=now, path=path, source=source, medium=medium,
            campaign=campaign, referer_host=referer_host, visitor_id=visitor_id,
            visitor_hash=_visitor_hash(request), username=username, role=role,
            ua_family=_ua_family(ua), is_bot=_is_bot(ua), status=response.status_code,
        )
        store.upsert_attribution(visitor_id, source, medium, campaign, now, visit_id)
        if now - _last_purge["ts"] > 3600:
            _last_purge["ts"] = now
            store.purge_visits(now - cfg.VISIT_RETENTION_DAYS * 86400)
    except Exception as exc:  # 埋点失败不影响页面
        log.flow(f"visit｜记录失败 {type(exc).__name__}: {exc}")

    if generated:
        response.set_cookie(
            cfg.VISITOR_COOKIE_NAME, visitor_id,
            max_age=cfg.VISITOR_COOKIE_DAYS * 86400,
            httponly=True, samesite="lax", path="/",
            secure=request.url.scheme == "https",
        )
    return response


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
def chat(req: ChatRequest, request: Request):
    pipeline = _pipeline()
    visitor_id = request.cookies.get(cfg.VISITOR_COOKIE_NAME, "")
    if visitor_id:
        identity = getattr(request.state, "auth", None) or ("", "")
        pipeline.store.bind_visitor_session(
            req.session_id, visitor_id, identity[0], identity[1]
        )
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


def _visit_window(days: int) -> tuple[int, float, float, str, str]:
    days = max(1, min(int(days), 365))
    now = time.time()
    from_ts = now - days * 86400
    return days, from_ts, now, local_day(from_ts), local_day(now)


@app.get("/visits/stats")
def visit_stats(days: int = 7, include_bots: int = 0) -> dict:
    days, from_ts, to_ts, from_day, to_day = _visit_window(days)
    store = _pipeline().store
    totals = store.visit_totals(from_ts, to_ts)
    chat_uv = store.visit_chat_uv(from_ts, to_ts)
    sources = store.visit_by_source(from_day, to_day, bool(include_bots))
    for row in sources:
        uv = row.get("uv") or 0
        chatted = chat_uv.get(row["source"], 0)
        row["chat_uv"] = chatted
        row["chat_rate"] = round(chatted / uv, 4) if uv else 0.0
    return {
        "days": days,
        **totals,
        "by_source": sources,
        "by_day": store.visit_by_day(from_day, to_day, bool(include_bots)),
    }


@app.get("/visits")
def visit_list(days: int = 7, source: str = "", include_bots: int = 0,
               page: int = 1, page_size: int = 20) -> dict:
    days, _, _, from_day, to_day = _visit_window(days)
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 200))
    total, items = _pipeline().store.list_visits(
        from_day, to_day, source.strip()[:64], bool(include_bots), page, page_size
    )
    return {
        "days": days, "page": page, "page_size": page_size,
        "total": total, "items": items,
    }


@app.get("/visits/questions")
def visit_questions(days: int = 30, source: str = "", limit: int = 100) -> dict:
    days, from_ts, to_ts, _, _ = _visit_window(days)
    limit = max(1, min(int(limit), 500))
    items = _pipeline().store.visit_questions(
        from_ts, to_ts, source.strip()[:64], limit
    )
    return {"days": days, "source": source, "items": items}


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
