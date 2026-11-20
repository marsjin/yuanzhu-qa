"""语料层：章节切分 → 片段切分 → 中文分词。"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from customer_service.core import config as cfg
from customer_service.retrieval.knowledge import ALL_NAMES

import jieba
import logging

jieba.setLogLevel(logging.WARNING)
for _w in ALL_NAMES:  # 人名整词化，避免「芮小丹」被切成「芮/小丹」
    jieba.add_word(_w)

CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百零两0-9]+章")
# 不锚定行首：用于从用户问题里取出「第X章」
CHAPTER_QUERY_RE = re.compile(r"第([一二三四五六七八九十百零两0-9]+)章")
SECTION_TITLES = {"卷首语", "后记", "尾声", "内容简介", "作者简介"}

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3,
              "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def cn_to_int(text: str) -> int | None:
    """中文数字或阿拉伯数字转整数，无法解析返回 None。"""
    s = text.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    total = 0
    num = 0
    for ch in s:
        if ch in _CN_DIGITS:
            num = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            total += (num or 1) * _CN_UNITS[ch]
            num = 0
        else:
            return None
    return total + num

STOPWORDS = set(
    """
    的 了 是 在 有 和 与 也 就 都 而 及 或 被 把 让 给 对 从 向 于 为 以 由
    我 你 他 她 它 我们 你们 他们 她们 自己 大家
    这 那 这个 那个 这些 那些 这里 那里 这样 那样 这么 那么
    什么 怎么 怎样 为什么 如何 哪个 哪些 谁 多少 几 是否 能否 可否
    吗 呢 吧 啊 呀 哦 嗯 请 请问 告诉 介绍 说说 讲 一下 一个 一些 关于
    可以 不能 不会 没有 还有 已经 现在 时候 如果 因为 所以 但是 但 虽然
    很 太 更 最 非常 特别 比较 可能 应该 需要 想 要 会 能 说 看
    """
    .split()
)


def load_chapters(path: Path) -> list[tuple[str, str]]:
    """把小说正文切成 [(章节标题, 正文), ...]。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到语料：{path}")
    text = path.read_text(encoding="utf-8-sig")

    chapters: list[tuple[str, str]] = []
    title, buf = "前言", []
    for line in text.splitlines():
        s = line.strip()
        is_heading = (CHAPTER_RE.match(s) and len(s) <= 12) or s in SECTION_TITLES
        if is_heading:
            if buf:
                chapters.append((title, "\n".join(buf)))
            title, buf = s, []
        else:
            buf.append(line)
    if buf:
        chapters.append((title, "\n".join(buf)))
    return [(t, b) for t, b in chapters if b.strip()]


def split_corpus() -> tuple[list[Document], dict]:
    """切章后递归分块，metadata 带章节与片内序号。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""],
        keep_separator="end",
    )
    chapters = load_chapters(cfg.CORPUS_PATH)
    docs: list[Document] = []
    for seq, (chapter, body) in enumerate(chapters):
        for i, piece in enumerate(splitter.split_text(body)):
            piece = piece.strip()
            if piece:
                docs.append(
                    Document(
                        page_content=piece,
                        metadata={
                            "book": cfg.BOOK_NAME,
                            "chapter": chapter,
                            "chapter_seq": seq,
                            "chunk_index": i,
                        },
                    )
                )
    return docs, {"book": cfg.BOOK_NAME, "chapters": len(chapters), "chunks": len(docs)}


def tokenize(text: str) -> list[str]:
    """jieba 分词 + 停用词 / 单字 / 纯符号过滤。"""
    tokens: list[str] = []
    for raw in jieba.lcut(text):
        t = raw.strip().lower()
        if not t or t in STOPWORDS:
            continue
        if not re.search(r"[0-9a-zA-Z\u4e00-\u9fff]", t):
            continue
        if len(t) < cfg.MIN_TOKEN_LEN:
            continue
        tokens.append(t)
    return tokens
