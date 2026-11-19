"""检索层测试：命中、覆盖率判定、情节采样、统计。"""

from __future__ import annotations

from customer_service.retrieval import index as ridx


def test_search_returns_hit(tiny_index):
    hits = ridx.search("丁元英", tiny_index, k=2)
    assert hits
    assert hits[0].coverage > 0
    assert "丁元英" in hits[0].doc.page_content


def test_search_ignores_stopwords_only(tiny_index):
    assert ridx.search("的了吗", tiny_index) == []


def test_decide_mode_document(tiny_index):
    hits = ridx.search("丁元英是谁", tiny_index, coverage_query="丁元英是谁")
    mode, coverage = ridx.decide_mode("丁元英是谁", hits)
    assert mode == "document"
    assert coverage >= 0.5


def test_decide_mode_llm_without_hits(tiny_index):
    mode, coverage = ridx.decide_mode("量子纠缠的原理", [])
    assert mode == "llm"
    assert coverage == 0.0


def test_plot_spine_is_ordered(tiny_index):
    spine = ridx.sample_plot_spine(tiny_index, n=10)
    seqs = [h.doc.metadata["chapter_seq"] for h in spine]
    assert len(spine) == 3
    assert seqs == sorted(seqs)


def test_stats(tiny_index):
    assert ridx.stats(tiny_index)["chapters"] == 3
