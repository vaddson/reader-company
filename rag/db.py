"""Хранение чанков и эмбэддингов в SQLite."""
from __future__ import annotations

import hashlib
import sqlite3
import struct

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc TEXT NOT NULL,
    doc_hash TEXT NOT NULL,
    seq INTEGER NOT NULL,
    heading_path TEXT NOT NULL,
    text TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc, seq);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def blob_to_vec(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def replace_doc_chunks(
    conn: sqlite3.Connection,
    doc: str,
    doc_hash: str,
    embed_model: str,
    chunks: list[tuple[str, str]],
    embeddings: list[list[float]],
) -> int:
    """Заменить все чанки документа (полная пересборка индекса)."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks и embeddings разные по длине")
    conn.execute("DELETE FROM chunks WHERE doc=?", (doc,))
    rows = [
        (doc, doc_hash, i, path, text, embed_model, vec_to_blob(emb))
        for i, ((path, text), emb) in enumerate(zip(chunks, embeddings))
    ]
    conn.executemany(
        "INSERT INTO chunks(doc, doc_hash, seq, heading_path, text, embed_model, embedding) "
        "VALUES(?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def load_chunks(conn: sqlite3.Connection) -> list[tuple[int, str, str, list[float]]]:
    rows = conn.execute(
        "SELECT seq, heading_path, text, embedding FROM chunks ORDER BY seq"
    ).fetchall()
    return [(seq, path, text, blob_to_vec(blob)) for seq, path, text, blob in rows]


def count_chunks(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    return row[0]
