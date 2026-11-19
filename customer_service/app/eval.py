"""评测入口：跑 golden_50，统计路由命中、拒答/转人工、引文有效性与延迟。

    PYTHONPATH=.. python -m customer_service.app.eval --limit 10
    PYTHONPATH=.. python -m customer_service.app.eval --limit 10 --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import statistics

from customer_service.app.pipeline import ChatPipeline
from customer_service.core import config as cfg
from customer_service.core.session import SessionStore
from customer_service.retrieval.index import build_index

GOLDEN = cfg.BASE_DIR / "data" / "eval" / "golden_50.jsonl"
CITE_RE = re.compile(r"《([^》]+)》第([一二三四五六七八九十百零两0-9]+)章")
CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def chapter_no(text: str) -> int:
    """把「四十七」「47」「十」统一成整数。"""
    if text.isdigit():
        return int(text)
    if "十" not in text:
        return CN_DIGITS.get(text[:1], -1)
    head, _, tail = text.partition("十")
    tens = CN_DIGITS.get(head, 1) if head else 1
    ones = CN_DIGITS.get(tail, 0) if tail else 0
    return tens * 10 + ones



def load_items(limit: int = 0) -> list[dict]:
    lines = GOLDEN.read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    return items[:limit] if limit > 0 else items


def quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def ratio(hits: list[bool]) -> str:
    return f"{sum(hits)}/{len(hits)}" if hits else "0/0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    ap.add_argument("--rebuild", action="store_true", help="强制重建索引")
    ap.add_argument("--verbose", action="store_true", help="打印每题回答片段")
    ap.add_argument("--ids", default="", help="只跑指定 id，逗号分隔")
    args = ap.parse_args()

    index = build_index(force=args.rebuild)
    chapters = {chapter_no(d.metadata["chapter"][1:-1]) for d in index.docs
                if d.metadata.get("chapter", "").startswith("第")}
    store = SessionStore(cfg.INDEX_DIR / "eval_sessions.db")
    pipeline = ChatPipeline(index, store)
    items = load_items(args.limit)
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        items = [it for it in items if it["id"] in wanted]

    route_ok: list[bool] = []
    no_answer_ok: list[bool] = []
    keyword_ok: list[bool] = []
    cite_total = cite_valid = 0
    latencies: list[int] = []
    rows: list[str] = []

    for it in items:
        sid = f"eval-{it['id']}"
        store.reset(sid)
        if it.get("pre"):
            pipeline.answer(sid, it["pre"])
        r = pipeline.answer(sid, it["question"])

        r_ok = r.route == it.get("expected_route")
        n_ok = bool(r.handoff) == bool(it.get("no_answer"))
        kws = it.get("keywords") or []
        k_ok = any(k in r.text for k in kws) if kws else None

        cites = CITE_RE.findall(r.text)
        bad = [c for c in cites if chapter_no(c[1]) not in chapters]
        cite_total += len(cites)
        cite_valid += len(cites) - len(bad)

        route_ok.append(r_ok)
        no_answer_ok.append(n_ok)
        if k_ok is not None:
            keyword_ok.append(k_ok)
        latencies.append(r.latency_ms)

        mark = lambda b: "✓" if b else "✗"
        rows.append(f"{it['id']:<8} {it['type']:<12} route{r_ok and '✓' or '✗'} "
                    f"no_answer{n_ok and '✓' or '✗'} "
                    f"cite={len(cites)}{('(bad%d)' % len(bad)) if bad else ''} "
                    f"{r.latency_ms:>5}ms")
        if args.verbose:
            rows.append(f"         Q: {it['question']}\n         A: {r.text[:120]}…")

    print("== 逐题 ==")
    for row in rows:
        print("  " + row)

    print("\n== 汇总 ==")
    print(f"  样本数        {len(items)}")
    print(f"  路由命中      {ratio(route_ok)}")
    print(f"  拒答/转人工   {ratio(no_answer_ok)}")
    print(f"  关键词命中    {ratio(keyword_ok)}")
    rate = (cite_valid / cite_total * 100) if cite_total else 0.0
    print(f"  引文有效率    {cite_valid}/{cite_total}（{rate:.1f}%）")
    print(f"  延迟 P50/P95  {quantile(latencies, 0.5)}ms / {quantile(latencies, 0.95)}ms")
    if latencies:
        print(f"  延迟均值      {int(statistics.mean(latencies))}ms")


if __name__ == "__main__":
    main()
