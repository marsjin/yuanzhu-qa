"""P2-2 覆盖率测试：IDF 加权后，泛词不能独力支撑「有据」。"""

from __future__ import annotations

import pytest

from customer_service.core import config as cfg
from customer_service.retrieval.index import decide_mode, query_weights, search


def test_oov_terms_get_ceiling_weight(tiny_index):
    """语料里完全没有的词按上限计权，权重高于普通词。"""
    weights = query_weights(tiny_index, ["量子纠缠", "私募"])
    assert weights["量子纠缠"] > weights["私募"]


def test_generic_word_alone_cannot_pass(tiny_index):
    """命中「丁元英」但漏掉「内容」，覆盖率应低于阈值。"""
    hits = search("丁元英的内容", tiny_index)
    assert hits
    assert hits[0].coverage < cfg.COVERAGE_THRESHOLD


def test_all_terms_matched_is_full(tiny_index):
    hits = search("丁元英", tiny_index)
    assert hits[0].coverage == pytest.approx(1.0)


def test_decide_mode_rejects_generic_word_query(tiny_index):
    query = "丁元英的内容"
    mode, coverage = decide_mode(query, search(query, tiny_index))
    assert mode == "llm"
    assert coverage < cfg.COVERAGE_THRESHOLD
