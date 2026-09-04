"""Разбиение книги в markdown на чанки вдоль заголовков.

Учитывает «нормальные» ATX-заголовки (`# ..`) и книги, переформатированные
из PDF (markitdown и т.п.): там заголовков нет, зато есть оглавление —
его пункты берутся как канонический список названий разделов, а в теле
к ним добавляются паттерны «Глава N …» и «1.2.3 Название».
"""
from __future__ import annotations

import re

HEADING_ATX_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*$")
FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
# строка оглавления: «1.1.1 Название . . . . . 14»
TOC_LEADER_RE = re.compile(r"(?:\s\.){2,}\s*\d{1,4}\s*$")
# «Глава 3 Инструменты» (короткая строка)
CHAPTER_HEADING_RE = re.compile(r"^\s*Глава\s+\d{1,3}\s+.{3,80}\s*$")
# «1.2.3 Название» (короткая строка)
NUM_HEADING_RE = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2}){1,}\s+\S.*\S\s*$")

DEFAULT_TARGET_CHARS = 3000
DEFAULT_OVERLAP = 0.15

BOOTSTRAP_PATH = "Начало документа"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


_QUOTE_START = set('"\'«„')


def _looks_like_code_title(title: str) -> bool:
    t = title.strip()
    if not t:
        return True
    return t[0] in _QUOTE_START


def _is_codeish(s: str) -> bool:
    """Похожа ли строка на код (таблица, комментарий, синтаксис)?"""
    s2 = s.strip()
    if not s2:
        return False
    if s2.startswith("|") or s2.startswith("#"):
        return True
    if re.search(r"\b(def|class|import|from|return|lambda|function|const|var|await)\b", s2):
        return True
    if any(ch in s2 for ch in "{;}:="):
        return True
    return False


def _is_prose(s: str) -> bool:
    s2 = s.strip()
    return len(s2) >= 25 and not _is_codeish(s2)


def _atx_is_real_heading(lines: list[str], i: int) -> bool:
    """`# строка` — заголовок, если сразу после идёт проза, а не код.

    Защита от фрагментов кода/комментариев, попавших в конвертацию без фенсов.
    """
    title = HEADING_ATX_RE.match(lines[i]).group(2).strip()
    if _looks_like_code_title(title):
        return False
    # предыдущая непустая строка выглядит как код → скорее всего это код-мусор
    j = i - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    if j >= 0 and _is_codeish(lines[j]):
        return False
    j = i + 1
    while j < len(lines):
        s = lines[j].strip()
        j += 1
        if not s:
            continue
        if _is_prose(s):
            return True
        if _is_codeish(s):
            return False
    return False


def extract_toc_titles(lines: list[str]) -> tuple[set[str], int, int]:
    """Найти оглавление (строки с точечными лидерами).

    Возвращает (названия, lo, hi) — границы зоны оглавления.
    """
    toc_idx = [i for i, l in enumerate(lines) if TOC_LEADER_RE.search(l)]
    if not toc_idx:
        return set(), -1, -1
    lo = max(0, min(toc_idx) - 4)
    hi = min(len(lines) - 1, max(toc_idx) + 4)
    canon: set[str] = set()
    plain_re = re.compile(r"^(.*?)\s\d{1,4}$")
    for i in range(lo, hi + 1):
        line = lines[i].strip()
        if not line:
            continue
        if TOC_LEADER_RE.search(line):
            title = line[: TOC_LEADER_RE.search(line).start()].strip()
        else:
            m = plain_re.match(line)
            if not m:
                continue
            title = m.group(1).strip()
        if 3 <= len(title) <= 130:
            canon.add(_norm(title))
    return canon, lo, hi


def _section_heading(
    line: str,
    canon: set[str],
    prev_norm: str | None,
) -> str | None:
    """Вернуть заголовок (исходную строку), если строка — заголовок раздела."""
    s = line.strip()
    if not s or s.startswith("|"):
        return None
    if TOC_LEADER_RE.search(line):
        return None  # строки оглавления не считаются заголовками
    n = _norm(s)
    if canon and n in canon:
        return None if n == prev_norm else s
    if len(s) <= 110 and CHAPTER_HEADING_RE.match(line):
        return None if (n == prev_norm or _looks_like_code_title(s)) else s
    if len(s) <= 120 and NUM_HEADING_RE.match(line) and not _is_codeish(s):
        return None if n == prev_norm else s
    return None


def parse_sections(md_text: str) -> list[tuple[str, str, str]]:
    """Разбить markdown на секции.

    Возвращает [(heading_path, heading_line, body), ...].
    Текст до первого заголовка — первая секция с путём «Начало документа».
    Повторяющиеся «бегущие строки» (running heads) не создают новых секций.
    """
    lines = md_text.splitlines()
    canon, toc_lo, toc_hi = extract_toc_titles(lines)

    sections: list[tuple[str, str, str]] = []
    stack = [""] * 7
    path = BOOTSTRAP_PATH
    heading_line = ""
    body: list[str] = []
    fence: str | None = None
    prev_norm: str | None = None

    def flush() -> None:
        nonlocal body
        text = "\n".join(body).strip()
        if text:
            sections.append((path, heading_line, text))
        body = []

    for i, line in enumerate(lines):
        f = FENCE_RE.match(line)
        if f:
            marker = f.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            body.append(line)
            continue

        if fence is None:
            in_toc = toc_lo <= i <= toc_hi
            m_atx = HEADING_ATX_RE.match(line) if not in_toc else None
            heading = None
            heading_level = 1
            if m_atx and _atx_is_real_heading(lines, i):
                heading = (line.strip(), _norm(m_atx.group(2)))
                heading_level = len(m_atx.group(1))
            elif not in_toc:
                h = _section_heading(line, canon, prev_norm)
                if h:
                    heading = (h, _norm(h))
                    mnum = re.match(r"^\s*(\d+(?:\.\d+)*)\s+", h)
                    heading_level = len(mnum.group(1).split(".")) if mnum else 1

            if heading is not None:
                hline, hnorm = heading
                flush()
                title = hnorm
                stack[heading_level - 1] = title
                for k in range(heading_level, 7):
                    stack[k] = ""
                path = " > ".join(t for t in stack if t)
                heading_line = hline
                prev_norm = hnorm
                continue

            # строки оглавления (с точечными лидерами) — мусор для поиска, не индексируем
            if TOC_LEADER_RE.search(line):
                continue
        body.append(line)

    flush()
    return sections


class _Flusher:
    """Собирает соседние секции в чанки целевого размера."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._parts: list[tuple[str, str]] = []
        self._size = 0
        self.out: list[tuple[str, str]] = []

    def _flush(self) -> None:
        if not self._parts:
            return
        text = "\n\n".join(part for _, part in self._parts)
        paths = list(dict.fromkeys(path for path, _ in self._parts))
        hp = " | ".join(p for p in paths if p != BOOTSTRAP_PATH) or BOOTSTRAP_PATH
        self.out.append((hp, text))
        self._parts = []
        self._size = 0

    def add(self, path: str, rendered: str) -> None:
        if self._parts and self._size + len(rendered) + 2 > self.limit:
            self._flush()
        self._parts.append((path, rendered))
        self._size += len(rendered) + 2

    def done(self) -> list[tuple[str, str]]:
        self._flush()
        return self.out


def split_long(text: str, limit: int, overlap: float) -> list[str]:
    """Разбить длинный текст на блоки ≤ limit с перекрытием ~overlap."""
    tail = max(1, int(limit * overlap))
    units: list[str] = []
    for para in [p for p in re.split(r"\n\s*\n", text) if p.strip()]:
        while len(para) > limit:
            units.append(para[:limit])
            para = para[limit - tail :]
        units.append(para)

    out: list[str] = []
    cur = ""
    for u in units:
        if len(u) > limit * 2:  # страховка от гигантского «абзаца»
            while len(u) > limit:
                out.append(u[:limit])
                u = u[limit - tail :]
        if cur and len(cur) + len(u) + 2 > limit:
            out.append(cur)
            carry = cur[limit - tail :] if len(cur) > tail else cur
            cur = carry + "\n\n" + u
        else:
            cur = cur + "\n\n" + u if cur else u
    if cur:
        out.append(cur)
    return out


def build_chunks(
    md_text: str,
    target_chars: int = DEFAULT_TARGET_CHARS,
    overlap: float = DEFAULT_OVERLAP,
) -> list[tuple[str, str]]:
    """Вернуть список (heading_path, chunk_text) для всей книги."""
    sections = parse_sections(md_text)
    fl = _Flusher(target_chars)
    for path, heading_line, body in sections:
        rendered = heading_line + "\n" + body if heading_line else body
        if len(rendered) > target_chars:
            fl._flush()
            fl.out.extend(
                (path, part) for part in split_long(rendered, target_chars, overlap)
            )
            continue
        fl.add(path, rendered)
    return fl.done()
