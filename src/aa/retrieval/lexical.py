"""SQLite FTS5/BM25 lexical retrieval over RU canonical chunks (issue #17).

Local/keyless only: the standard-library ``sqlite3`` module with the
``FTS5`` extension and its ``bm25()`` ranker. No remote search service.

Index layout (generated artifact, never committed):

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


def lexical_search(
    db_path: str | Path, query: str, *, top_k: int = LEXICAL_TOP_K
) -> list[tuple[str, float]]:
    """Search the FTS5 index; return ``[(chunk_id, bm25_score)]`` (best first).

    ``bm25()`` returns negative values where more negative is better; this
    function converts to a positive relevance (``-bm25``) so higher is
    better and ranks best-first. Empty/stopword-only queries return [].
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
