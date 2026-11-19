"""知识层：书目元数据、人物档案、别名归一化、FAQ 匹配。"""

from __future__ import annotations

import json
from functools import lru_cache

from customer_service.core import config as cfg

BOOK_META = {
    "name": cfg.BOOK_NAME,
    "author": "豆豆",
    "protagonist": "丁元英",
    "theme": "文化属性：强势文化与弱势文化，得救之道在于自己",
    "aliases": ["遥远的救世主", "救世主", "天道"],
}

# 只收录高置信度事实（定位 / 标签 / 结局），情节细节交给原文检索
CHARACTERS: dict[str, dict] = {
    "丁元英": {"aliases": ["元英"], "role": "男主角，前私募基金操盘手，隐居古城",
               "tags": ["文化属性", "强势文化", "杀富济贫", "格律诗"],
               "fate": "芮小丹牺牲后离开古城"},
    "芮小丹": {"aliases": ["小丹"], "role": "女主角，古城刑警",
               "tags": ["天国之女", "警察", "爱情"], "fate": "执行任务时牺牲"},
    "欧阳雪": {"aliases": ["阳雪"], "role": "芮小丹挚友，饭店老板，格律诗股东",
               "tags": ["友情", "格律诗"], "fate": ""},
    "肖亚文": {"aliases": ["亚文"], "role": "丁元英前助理，后执掌格律诗公司",
               "tags": ["职场", "格律诗"], "fate": ""},
    "韩楚风": {"aliases": ["楚风"], "role": "正天集团总裁，丁元英好友",
               "tags": ["商界", "友情"], "fate": ""},
    "林雨峰": {"aliases": ["雨峰"], "role": "乐圣公司董事长，格律诗商战对手",
               "tags": ["商战", "乐圣"], "fate": "败局后自杀"},
    "刘冰": {"aliases": [], "role": "王庙村音箱项目合伙人之一",
             "tags": ["王庙村", "格律诗"], "fate": "跳楼自杀"},
    "叶晓明": {"aliases": [], "role": "王庙村音箱项目合伙人",
               "tags": ["王庙村", "格律诗"], "fate": ""},
    "冯世杰": {"aliases": [], "role": "王庙村音箱项目合伙人",
               "tags": ["王庙村", "格律诗"], "fate": ""},
}

NAME_ALIASES: dict[str, str] = {}
for _name, _info in CHARACTERS.items():
    NAME_ALIASES[_name] = _name
    for _alias in _info["aliases"]:
        NAME_ALIASES[_alias] = _name

# 最长优先，避免「小丹」先于「芮小丹」命中
ALL_NAMES = sorted(NAME_ALIASES.keys(), key=len, reverse=True)


def detect_entities(text: str) -> list[str]:
    """识别人物并按出现顺序去重（别名归一化为标准名）。"""
    found: list[str] = []
    for word in ALL_NAMES:
        if word in text:
            canonical = NAME_ALIASES[word]
            if canonical not in found:
                found.append(canonical)
    return found


def mentions_book(text: str) -> bool:
    return any(kw in text for kw in BOOK_META["aliases"])


def profile_text(name: str) -> str:
    info = CHARACTERS[name]
    parts = [f"《{cfg.BOOK_NAME}》{info['role']}"]
    if info["tags"]:
        parts.append("标签：" + "、".join(info["tags"]))
    if info["fate"]:
        parts.append(f"结局：{info['fate']}")
    return f"【{name}】" + "；".join(parts)


@lru_cache(maxsize=1)
def faq_entries() -> tuple[dict, ...]:
    """读取 FAQ 知识库（JSON 数组）。文件缺失时返回空。"""
    if not cfg.FAQ_PATH.exists():
        return ()
    data = json.loads(cfg.FAQ_PATH.read_text(encoding="utf-8"))
    return tuple(data)


def match_faq(question: str) -> dict | None:
    """按关键词命中数匹配 FAQ，取命中最多的一条（至少 1 个）。"""
    best: dict | None = None
    best_hits = 0
    for entry in faq_entries():
        hits = sum(1 for kw in entry.get("keywords", []) if kw in question)
        if hits > best_hits:
            best, best_hits = entry, hits
    return best
