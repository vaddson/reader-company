#!/usr/bin/env python3
"""reader — консольный RAG-помощник по markdown-книге (Ollama + SQLite)."""
from __future__ import annotations

import argparse
import os
import sys
import time

from rag import db
from rag.embedder import OllamaError, embed, embed_model
from rag.llm import complete, llm_model, parse_reply, system_prompt
from rag.mdchunks import build_chunks
from rag.retriever import retrieve


def build_context(hits: list[tuple[float, int, str, str]]) -> str:
    blocks = []
    for i, (score, _seq, path, text) in enumerate(hits, 1):
        blocks.append(f"[Материал {i}] — {path}\n{text}")
    return "\n\n".join(blocks)


def truncate(s: str, n: int = 120) -> str:
    s = " ".join(s.split())
    return s[: n - 1] + "…" if len(s) > n else s


def cmd_index(args: argparse.Namespace) -> int:
    path = os.path.abspath(args.book)
    if not os.path.isfile(path):
        print(f"Не найден файл: {args.book}", file=sys.stderr)
        return 1
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError as exc:
        print(f"Не удалось прочитать файл как UTF-8: {exc}", file=sys.stderr)
        return 1
    if not text.strip():
        print("Файл пуст.", file=sys.stderr)
        return 1

    chunks = build_chunks(text, target_chars=args.target_chars)
    if not chunks:
        print("Не удалось выделить ни одного чанка.", file=sys.stderr)
        return 1
    print(f"Книга: {len(text):,} симв. → {len(chunks)} чанков (цель ≈ {args.target_chars} симв./чанк)")

    print(f"Векторизация моделью {embed_model()}…")
    started = time.monotonic()
    try:
        embeddings = embed([chunk_text for _p, chunk_text in chunks])
    except OllamaError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    dim = len(embeddings[0])

    doc_hash = db.sha256(text.encode("utf-8"))
    conn = db.connect(args.db)
    stored = db.replace_doc_chunks(conn, path, doc_hash, embed_model(), chunks, embeddings)
    db.set_meta(conn, "source", path)
    db.set_meta(conn, "source_hash", doc_hash)
    db.set_meta(conn, "embed_model", embed_model())
    db.set_meta(conn, "embed_dim", dim)
    db.set_meta(conn, "chunk_chars", args.target_chars)
    total = db.count_chunks(conn)
    print(
        f"Готово: {stored} чанков сохранено, {total} всего в базе.\n"
        f"Индекс: {os.path.abspath(args.db)} (dim={dim})\n"
        f"Время: {time.monotonic() - started:.1f}s"
    )
    return 0


def print_hits(hits: list[tuple[float, int, str, str]], args: argparse.Namespace) -> None:
    print("\n— Найденные фрагменты (top " + str(args.top_k) + "):")
    for i, (score, seq, path, text) in enumerate(hits, 1):
        print(f"  {i}. [{score:.3f}] § {truncate(path, 90)}")
        print(f"     {truncate(text, 160)}")
    print()


def run_ask(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.db):
        print(
            f"Не найден индекс: {args.db}\n"
            "Сначала выполните: python main.py index <книга.md> --db book.db",
            file=sys.stderr,
        )
        return 1
    conn = db.connect(args.db)
    total = db.count_chunks(conn)
    if total == 0:
        print("Индекс пуст.", file=sys.stderr)
        return 1

    model = args.model or llm_model()
    source = db.get_meta(conn, "source") or "?"
    print(f"Индекс: {os.path.abspath(args.db)} — {total} чанков (книга: {source})")
    print(f"LLM: {model} · эмбэд: {db.get_meta(conn, 'embed_model')} · top-k: {args.top_k}")
    print("Команды: /reset (сброс диалога), /hits (показать найденные фрагменты), /exit")
    print("Если отвечаю — следующий ввод это новый вопрос.\n")

    history: list[dict] = []      # сообщения после первичного вопроса (цепочка уточнений)
    last_kind: str | None = None  # "answer" | "clarify"
    root_question: str | None = None
    last_hits: list | None = None

    while True:
        prompt = "   ? " if last_kind == "clarify" else "*Вопрос>* "
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/exit", "/quit", "/quit"):
            break
        if line == "/reset":
            history, last_kind, root_question, last_hits = [], None, None, None
            print("Диалог сброшен.")
            continue
        if line == "/hits":
            if last_hits:
                print_hits(last_hits, args)
            else:
                print("Пока нет поиска.")
            continue
        if line.startswith("/"):
            print(f"Неизвестная команда: {line} (есть /reset, /hits, /exit)")
            continue

        # Новое или продолжение вопроса
        if root_question is None or last_kind == "answer":
            root_question = line
            history = []
            query_text = line
        else:  # ответ на уточняющий вопрос
            history.append({"role": "user", "content": line})
            query_text = (root_question or "") + " " + line

        print("Ищу по книге…")
        tried = time.monotonic()
        try:
            hits = retrieve(conn, query_text, k=args.top_k)
        except OllamaError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if not hits:
            print("Поиск ничего не дал — попробуйте сформулировать иначе.", file=sys.stderr)
            continue
        last_hits = hits
        print(f"  ({time.monotonic() - tried:.1f}s, лучший score: {hits[0][0]:.3f})")

        print("Думаю… ")
        sys.stdout.flush()
        tried = time.monotonic()
        messages = [
            {"role": "system", "content": system_prompt(build_context(hits))},
            {"role": "user", "content": root_question or line},
            *history,
        ]
        try:
            raw = complete(messages, model=model)
        except OllamaError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        kind, body = parse_reply(raw)
        # обновляем историю: сохраняем последний ход ассистента
        if kind == "clarify":
            history.append({"role": "assistant", "content": body})
        print(f"\n({time.monotonic() - tried:.1f}s)\n")
        if kind == "clarify":
            print(f"  → {body}\n")
            last_kind = "clarify"
            continue
        print(body)
        print()
        last_kind = "answer"

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reader",
        description="Консольный RAG по markdown-книге (Ollama + SQLite).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="прочитать markdown-книгу и построить индекс")
    pi.add_argument("book", help="путь к .md-файлу книги")
    pi.add_argument("--db", default="book.db", help="файл SQLite-индекса (по умолчанию book.db)")
    pi.add_argument("--target-chars", type=int, default=3000,
                    help="целевой размер чанка в символах (по умолчанию 3000)")
    pi.set_defaults(func=cmd_index)

    pa = sub.add_parser("ask", help="задавать вопросы по проиндексированной книге")
    pa.add_argument("--db", default="book.db", help="файл SQLite-индекса (по умолчанию book.db)")
    pa.add_argument("--model", default=None, help="LLM-модель (по умолчанию $RAG_LLM_MODEL или qwen3.8:27b)")
    pa.add_argument("--top-k", type=int, default=6, help="число фрагментов в контекст (по умолчанию 6)")
    pa.set_defaults(func=run_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
