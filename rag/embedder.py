"""Эмбэддинги через Ollama /api/embed."""
from __future__ import annotations

import json
import os
import sys
import time

import requests

DEFAULT_EMBED_MODEL = "qwen3-embedding:8b"


class OllamaError(RuntimeError):
    pass


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def embed_model() -> str:
    return os.environ.get("RAG_EMBED_MODEL") or DEFAULT_EMBED_MODEL


def embed(
    texts: list[str],
    model: str | None = None,
    batch_size: int = 8,
    timeout: int = 600,
    progress: bool = True,
) -> list[list[float]]:
    """Векторизовать список текстов батчами. Возвращает список векторов."""
    model = model or embed_model()
    out: list[list[float]] = []
    total = len(texts)
    if not total:
        return out
    for i in range(0, total, batch_size):
        part = texts[i : i + batch_size]
        started = time.monotonic()
        try:
            resp = requests.post(
                ollama_host() + "/api/embed",
                json={"model": model, "input": part},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.ConnectionError as exc:
            raise OllamaError(
                f"Ollama недоступна ({ollama_host()}): {exc}. Проверьте, запущен ли ollama serve."
            ) from exc
        except requests.RequestException as exc:
            raise OllamaError(f"Ошибка запроса к Ollama: {exc}") from exc

        embs = data.get("embeddings")
        if embs is None and "embedding" in data:  # старые версии API
            embs = [data["embedding"]]
        if embs is None:
            raise OllamaError(
                "Неожиданный ответ /api/embed: " + json.dumps(data)[:300]
            )
        out.extend(embs)
        if progress:
            print(
                f"\r  эмбэд: {min(i + batch_size, total)}/{total} "
                f"({time.monotonic() - started:.1f}s на батч)",
                end="",
                file=sys.stderr,
                flush=True,
            )
    if progress:
        print(file=sys.stderr)
    return out
