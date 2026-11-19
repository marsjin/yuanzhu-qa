"""命令行入口：交互模式或单问单答。

    python -m customer_service.app.cli
    python -m customer_service.app.cli -q "丁元英是谁？"
    python -m customer_service.app.cli --rebuild -q "你好"
"""

from __future__ import annotations

import argparse
import sys

from customer_service.core import obs as log
from customer_service.retrieval import index as ridx
from customer_service.app.pipeline import ChatPipeline


def _print_result(result) -> None:
    print(f"\nAI（{result.label}）:\n{result.text}")
    print(f"\n[路由] {result.route}｜意图 {result.intent}｜{result.latency_ms}ms"
          f"｜trace {result.trace_id}"
          + ("｜已建议转人工" if result.handoff else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="《遥远的救世主》智能客服")
    parser.add_argument("-q", "--question", help="只问一个问题后退出")
    parser.add_argument("--session", default="cli", help="会话 ID（默认 cli）")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引")
    args = parser.parse_args()

    index = ridx.build_index(force=args.rebuild)
    pipeline = ChatPipeline(index)

    if args.question:
        _print_result(pipeline.answer(args.session, args.question))
        return

    print("《遥远的救世主》智能客服已就绪。")
    print("支持：书本问答 / 情节梳理 / 人物对比 / 常见问题；输入 exit 退出，clear 清空会话。")
    while True:
        try:
            question = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if question.lower() in {"exit", "quit"}:
            print("再见！")
            break
        if question.lower() == "clear":
            pipeline.store.reset(args.session)
            print("（会话已清空）")
            continue
        if not question:
            continue
        try:
            _print_result(pipeline.answer(args.session, question))
        except RuntimeError as e:  # 典型情况：未配置 API Key
            log.recovery(str(e))
            print(f"\n[错误] {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
