# 变更与部署记录

## 运行环境

本地统一使用 **Python 3.11**（`conda activate pythonPro311`），与生产镜像 `python:3.11-slim` 一致。

```bash
cd customer_service
python run.py -q "本书简介"                 # CLI
python run_service.py --port 8019          # HTTP 服务
PYTHONPATH=.. python -m pytest tests -q    # 单测
```

> 项目根目录的 `.venv`（Python 3.13）已删除：3.13 下没有 `onnxruntime` 轮子，装不了 `fastembed`，
> 跑起来会静默降级为纯 BM25。向量能力依赖 3.11 环境。

依赖已锁定：`requirements.txt` 固定直接依赖版本，`requirements.lock.txt` 记录 79 个包的完整闭包；
Docker 构建使用锁文件，保证每次镜像一致。

## v0.3.0 — 2026-11-20 登录会话、双角色与运营统计

### 功能变更

| 模块 | 内容 |
|---|---|
| 登录会话 | `/login` 登录页、`/logout` 退出；HMAC 签名 Cookie，有效期 60 分钟（`CS_AUTH_SESSION_MINUTES`），退出即失效 |
| 权限模型 | `admin` 全部权限；`user` 仅对话与 `/health`；`/admin`、`/events/stats` 仅管理员 |
| 账号 | 固定配置：`admin`、`user`、`jerry`（`CS_AUTH_EXTRA_USERS` 可扩展）；不支持在线改密 |
| 当前用户 | `GET /me`；首页右上角、管理页头部显示用户名与角色 |
| 管理页 | `/admin` 只读：事件统计（默认/查询分流、点击率）+ 服务健康 |
| 交互 | 「清空会话」移到消息区上方工具条，首条消息后出现；新增「退出」 |
| 埋点 | 展示/点击分 `default`（首页封面卡）与 `answer`（回答来源徽标） |
| 外链 | 购买来源链接带 `from` + UTM 标识（`CS_REFERRAL_TAG`） |
| 访问日志 | 信任反代 `X-Forwarded-For`，记录真实客户端 IP |

### 鉴权行为变化

- 浏览器页面只认登录 Cookie：未登录 303 跳 `/login`，退出后立即无权限
- API 仍兼容 HTTP Basic（curl/脚本）；两者同时存在时以 Cookie 身份为准
- 浏览器缓存的旧 Basic 凭证不会覆盖当前登录账号

### 验证

- 单元测试：97 passed
- 端到端：匿名访问 303 → 登录页；admin `/admin` 200；user/jerry `/admin` 403；退出后再次访问 303
- `/me`：Cookie 身份优先，未登录 401

### 部署与升级

环境变量新增：

```bash
CS_AUTH_USERNAME=admin
CS_AUTH_PASSWORD=***
CS_AUTH_USER_USERNAME=user
CS_AUTH_USER_PASSWORD=***
CS_AUTH_EXTRA_USERS=jerry:***
CS_AUTH_SESSION_MINUTES=60
```

升级：`git pull` → `docker compose build` → `docker compose up -d`。账号密码只存在于服务器环境变量/`.env`（不入库）。

## v0.1.0 — 2026-11-19 首次部署成功

服务已部署至服务器并通过公网验收。

### 部署形态

```text
浏览器 --HTTPS--> Caddy / Nginx --127.0.0.1:8000--> Docker 容器（FastAPI）
                                                    ├── /data/index        BM25 索引（Volume）
                                                    ├── /data/sessions.db  会话数据（Volume）
                                                    └── DeepSeek API       外部模型调用
```

- 镜像：`customer-service:latest`，多阶段无关，单容器单 worker
- 依赖安装走国内镜像（`PIP_INDEX_URL` 可构建参数覆盖）
- 容器以非 root 用户 `appuser` 运行

### 功能范围

| 里程碑 | 内容 |
|---|---|
| M0 | 语料解析 47 章 / 728 片段 / 词表 18946，BM25 索引持久化，人物与 FAQ 知识基线 |
| M1 | 五类意图规则路由、三态出口、纯代码可信度校验、域外话题门控 |
| M2 | LangGraph 九节点双环 Agent、四层防循环、规则计划（默认省一次模型调用） |
| M3 | SQLite 会话持久化、指代补全、历史窗口注入 |
| M4 | FastAPI 服务、SSE 流式、静态聊天页、HTTP Basic 鉴权 |
| M5 | golden_50 分层评测集、路由/拒答/引文/延迟四类指标 |

### 验收结果

| 项 | 结果 |
|---|---|
| 单元测试 | 38 passed |
| 路由命中 | 10/10（抽样） |
| 拒答与转人工 | 10/10（抽样） |
| 引文有效率 | 100%（抽样 61 条全部存在） |
| 快路径延迟 | P50 约 0.9s |
| 情节梳理延迟 | 约 6–7s |
| 鉴权 | 未登录 401、错误密码 401、正确登录 200 |

### 已知限制

1. 全量 50 条评测基线尚未跑完，M5 指标仍为抽样结论。
2. 抽象问法（如「结局」）召回不足，人物档案尚未作为证据注入。
3. 会话为 SQLite 单机存储，横向扩容前需迁移至 PostgreSQL/Redis。
4. Agent 工具层使用模块级索引注入，多书或多实例并发存在串数据风险。
5. 前端无「停止生成」「历史会话列表」。

### 部署文件

- `Dockerfile`、`docker-compose.yml`、`.dockerignore`
- `Caddyfile`（备选）、nginx 配置见部署说明
- 环境变量模板：`.env.example`
