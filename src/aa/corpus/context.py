"""AA corpus/context provider boundary.

Resolves which AA corpus snapshot (location + version) the worker should
use when building prompts. No corpus is loaded over the network here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusInfo:
    """Describes the resolved AA corpus snapshot."""

    path: str
    version: str


class CorpusContext:
    """Offline corpus pointer used during startup/shutdown wiring."""

    def __init__(self, path: str, version: str) -> None:
        self._info = CorpusInfo(path=path, version=version)
        self._loaded = False

    async def load(self) -> CorpusInfo:
        """Mark the corpus as loaded (no I/O beyond pointer checks)."""
        self._loaded = True
        return self._info

    async def unload(self) -> None:
        """Release corpus resources."""
        self._loaded = False

    @property
    def loaded(self) -> bool:
        """Whether the corpus context is loaded."""
        return self._loaded

    @property
    def info(self) -> CorpusInfo:
        """Return the corpus pointer."""
        return self._info
