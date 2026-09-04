"""Генерация ответов через Ollama /api/chat (JSON-огиление ответ/вопрос)."""
from __future__ import annotations

import json
import os

import requests

from .embedder import OllamaError, ollama_host

DEFAULT_LLM_MODEL = "qwen3.8:27b"

SYSTEM_TMPL = """Ты — ассистент, отвечающий на вопросы по книге. Отвечай, опираясь ТОЛЬКО на приведённые ниже материалы (фрагменты книги).

{context}

Правила:
1. Отвечай на языке вопроса пользователя.
2. Не выдумывай факты, которых нет в материалах. Если ответа в материалах нет — так и скажи.
3. Ссылайся на названия разделов книги, откуда взяты факты.
4. Если вопрос слишком размыт или допускает несколько трактовок — не отвечай, а задай один уточняющий вопрос.
5. Ответь СТРОГО одним JSON-объектом, без пояснений вне JSON:
   {{"type": "answer", "text": "<ответ>"}}  — если отвечаешь,
   {{"type": "clarify", "text": "<уточняющий вопрос>"}} — если уточняешь."""


def llm_model() -> str:
    return os.environ.get("RAG_LLM_MODEL") or DEFAULT_LLM_MODEL


def system_prompt(context: str) -> str:
    return SYSTEM_TMPL.format(context=context)


def complete(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    timeout: int = 900,
) -> str:
    """Один ход чата. Возвращает текст ответа модели."""
    model = model or llm_model()
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(
            ollama_host() + "/api/chat", json=payload, timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.ConnectionError as exc:
        raise OllamaError(
            f"Ollama недоступна ({ollama_host()}): {exc}. Проверьте, запущен ли ollama serve."
        ) from exc
    except requests.RequestException as exc:
        raise OllamaError(f"Ошибка запроса к Ollama /api/chat: {exc}") from exc

    content = (data.get("message") or {}).get("content")
    if not content:
        raise OllamaError("Пустой ответ от модели: " + json.dumps(data)[:300])
    return content


def parse_reply(text: str) -> tuple[str, str]:
    """Разобрать JSON-огиление. Возвращает ("answer"|"clarify", текст)."""
    t = (text or "").strip()
    # на всякий случай: модель обернула в ```json-блок
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:].lstrip()
    try:
        obj = json.loads(t)
    except (ValueError, TypeError):
        return "answer", t

    if isinstance(obj, dict):
        kind = str(obj.get("type", "answer") or "answer").strip().lower()
        if kind not in ("answer", "clarify"):
            kind = "answer"
        body = obj.get("text")
        if not isinstance(body, str) or not body.strip():
            # модель вложила что-то другое — рендерим обратно в текст
            body = json.dumps(obj, ensure_ascii=False)
        return kind, body.strip()
    return "answer", t
