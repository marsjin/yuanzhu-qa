"""HTTP 服务启动器：在 customer_service/ 内直接运行，无需 PYTHONPATH 或完整包名。

    python run_service.py                       # 127.0.0.1:8000
    python run_service.py --port 8017           # 指定端口
    python run_service.py --host 0.0.0.0 --port 8017 --reload
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PARENT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PARENT)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in [PARENT, os.environ.get("PYTHONPATH", "")] if p
)

import uvicorn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="《遥远的救世主》智能客服 HTTP 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="代码变更自动重载（开发用）")
    args = parser.parse_args()

    uvicorn.run(
        "customer_service.app.service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
