"""SQLite FTS5/BM25 lexical retrieval over RU canonical chunks (issue #115).

Local/keyless only: the standard-library ``sqlite3`` module with the
``FTS5`` extension and its ``bm25()`` ranker. No remote search service.

Persistence layout (generated artifact, never committed):

- ``lexical.db`` — persisted FTS5 table (startup/persistence only);
- hot-path search runs against an SQLite ``:memory:`` copy loaded at
  startup via the supported backup mechanism. Ordinary turns never
  open/read ``lexical.db`` from disk.

Index layout:

- ``chunks_fts`` virtual table: ``chunk_id`` (unindexed), ``section``
  (unindexed), ``norm`` (stemmed Russian token text).

Search converts each Russian query to stemmed tokens (see
:mod:`aa.retrieval.normalize`) and issues an OR ``MATCH`` query ordered
by ``bm25()``. FTS5 special characters in user text are stripped by the
tokenizer path, so raw user text can never break the MATCH syntax.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from aa.retrieval.normalize import stemmed_norm_text

FTS_TABLE = "chunks_fts"
LEXICAL_TOP_K = 40


class LexicalError(ValueError):
    """Raised when the lexical index cannot be built or searched."""


def build_lexical_db(
    db_path: str | Path, *, chunk_ids: list[str], sections: list[str], texts: list[str]
) -> None:
    """Build a fresh FTS5 lexical database over RU chunk texts."""
    if not chunk_ids or len(chunk_ids) != len(sections) or len(chunk_ids) != len(texts):
        raise LexicalError("chunk_ids, sections and texts must be non-empty parallel lists")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise LexicalError("chunk_ids must be unique")
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute(f"DROP TABLE IF EXISTS {FTS_TABLE}")
        connection.execute(
            f"CREATE VIRTUAL TABLE {FTS_TABLE} USING fts5("
            "chunk_id UNINDEXED, section UNINDEXED, norm, tokenize='unicode61')"
        )
        rows = [
            (chunk_id, section, stemmed_norm_text(text))
            for chunk_id, section, text in zip(chunk_ids, sections, texts, strict=True)
        ]
        connection.executemany(
            f"INSERT INTO {FTS_TABLE} (chunk_id, section, norm) VALUES (?, ?, ?)", rows
        )
        connection.commit()
    finally:
        connection.close()


def _match_query(query: str) -> str | None:
    """Convert a Russian query to a safe FTS5 OR expression, or None."""
    norm = stemmed_norm_text(query)
    if not norm.strip():
        return None
    terms = [f'"{token}"' for token in norm.split()]
    return " OR ".join(terms)


def load_lexical_into_memory(db_path: str | Path) -> sqlite3.Connection:
    """Load the persisted FTS5 database into a ``:memory:`` connection.

    Uses SQLite's supported backup mechanism. The returned connection is
    long-lived process memory; callers must not reopen the file per turn.
    """
    source_path = Path(db_path)
    if not source_path.is_file():
        raise LexicalError(f"lexical database is missing: {source_path}")
    try:
        source = sqlite3.connect(str(source_path))
    except sqlite3.Error as exc:
        raise LexicalError(f"lexical database is unreadable: {exc}") from exc
    try:
        target = sqlite3.connect(":memory:")
        try:
            source.backup(target)
        except sqlite3.Error as exc:
            target.close()
            raise LexicalError(f"lexical memory load failed: {exc}") from exc
        try:
            count = target.execute(f"SELECT COUNT(*) FROM {FTS_TABLE}").fetchone()
        except sqlite3.Error as exc:
            target.close()
            raise LexicalError(f"lexical memory index is unreadable: {exc}") from exc
        if count is None:
            target.close()
            raise LexicalError("lexical memory index is empty")
        return target
    finally:
        source.close()


def lexical_search_conn(
    connection: sqlite3.Connection, query: str, *, top_k: int = LEXICAL_TOP_K
) -> list[tuple[str, float]]:
    """Search an open (in-memory) FTS5 connection (hot path, no disk I/O)."""
    if top_k <= 0:
        raise LexicalError("top_k must be > 0")
    match = _match_query(query)
    if match is None:
        return []
    try:
        rows = connection.execute(
            f"SELECT chunk_id, bm25({FTS_TABLE}) AS rank "
            f"FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH ? "
            "ORDER BY rank LIMIT ?",
            (match, top_k),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        raise LexicalError(f"lexical search failed: {exc}") from exc
    scored = [(str(chunk_id), -float(rank)) for chunk_id, rank in rows]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def lexical_search(
    db_path: str | Path, query: str, *, top_k: int = LEXICAL_TOP_K
) -> list[tuple[str, float]]:
    """Search the persisted FTS5 index file (startup/tooling only).

    Ordinary turns must use :func:`lexical_search_conn` against the
    RAM-resident connection instead of reopening the file per query.
    """
    if top_k <= 0:
        raise LexicalError("top_k must be > 0")
    match = _match_query(query)
    if match is None:
        return []
    connection = sqlite3.connect(str(db_path))
    try:
        try:
            rows = connection.execute(
                f"SELECT chunk_id, bm25({FTS_TABLE}) AS rank "
                f"FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH ? "
                "ORDER BY rank LIMIT ?",
                (match, top_k),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            raise LexicalError(f"lexical search failed: {exc}") from exc
    finally:
        connection.close()
    scored = [(str(chunk_id), -float(rank)) for chunk_id, rank in rows]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def multi_lexical_search(
    db_path: str | Path, queries: list[str], *, top_k: int = LEXICAL_TOP_K
) -> dict[str, list[tuple[str, float]]]:
    """Run :func:`lexical_search` for each query string."""
    return {query: lexical_search(db_path, query, top_k=top_k) for query in queries}
