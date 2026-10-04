"""AA retrieval boundary (issue #17).

RU-first hybrid retrieval over the aligned corpus (#8):

- SQLite FTS5/BM25 lexical retrieval over RU canonical chunks;
- pinned local ``intfloat/multilingual-e5-base`` semantic retrieval over RU
  canonical chunks (normalized embeddings + exact ``IndexFlatIP`` semantics);
- RRF fusion with overlap dedup and section/chapter diversity.

Local/keyless only: no remote embeddings or search service is used here.
Plaintext text-bearing indexes are runtime/generated artifacts under
``corpus/generated/retrieval/`` and are never committed or cached.
"""

from __future__ import annotations
