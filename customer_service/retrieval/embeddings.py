"""本地向量编码：fastembed + bge-small-zh，纯 CPU 推理，不调用大模型。

依赖缺失或模型加载失败时 available() 返回 False，检索自动退回纯 BM25。
"""

from __future__ import annotations

import numpy as np

from customer_service.core import config as cfg
from customer_service.core import obs as log

_model = None
_disabled = False


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=cfg.EMBED_MODEL, cache_dir=str(cfg.EMBED_CACHE_DIR))
    return _model


def available() -> bool:
    """探测向量能力；失败只记一次日志，避免每次检索都重试。"""
    global _disabled
    if _disabled or not cfg.EMBED_ENABLED:
        return False
    try:
        _get_model()
        return True
    except Exception as exc:
        _disabled = True
        log.recovery(f"embed｜不可用（{type(exc).__name__}），退回纯 BM25")
        return False


def encode(texts: list[str]) -> np.ndarray:
    """批量编码并做 L2 归一化，返回 (n, dim) 的 float32 矩阵。"""
    matrix = np.array(list(_get_model().embed(list(texts), batch_size=32)), dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
