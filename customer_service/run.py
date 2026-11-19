"""本地启动器：在 customer_service/ 内直接运行，无需 PYTHONPATH 或完整包名。

    python run.py                       # 交互模式
    python run.py -q "丁元英是谁？"      # 单问单答
    python run.py --rebuild -q "你好"    # 强制重建索引
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from customer_service.app.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
