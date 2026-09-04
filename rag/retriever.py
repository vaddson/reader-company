"""Векторный поиск по индексированным чанкам (косинусное сходство)."""
from __future__ import annotations

import math

from .db import load_chunks
from .embedder import embed


def cos_sim(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def retrieve(
    conn,
    question: str,
    k: int = 6,
    min_score: float = 0.0,
    model: str | None = None,
) -> list[tuple[float, int, str, str]]:
    """Вернуть [(score, seq, heading_path, text), ...] — top-k по косинусу."""
    chunks = load_chunks(conn)
    if not chunks:
        raise RuntimeError("Индекс пуст — сначала выполните: python main.py index <книга.md>")
    qv = embed([question], model=model, progress=False)[0]

    dim = len(qv)
    scored = []
    for seq, path, text, vec in chunks:
        if len(vec) != dim:
            continue
        scored.append((cos_sim(qv, vec), seq, path, text))
    scored.sort(key=lambda item: (-item[0], item[1]))
    hits = scored[:k]
    if min_score > 0:
        hits = [h for h in hits if h[0] >= min_score]
    return hits
