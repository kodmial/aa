"""AA conversation package (issue #118 cutover).

The LangGraph runtime under :mod:`aa.conversation.graph`,
:mod:`aa.conversation.graph_runtime`, :mod:`aa.conversation.planner_node`,
:mod:`aa.conversation.retrieval_node` and
:mod:`aa.conversation.turn_pipeline` is the only ordinary conversational
path in production.

The retired ``aa.conversation.orchestrator`` / ``aa.conversation.meta``
modules remain on disk for offline qualification history only. They are
deliberately not imported here so production processes (``aa.app`` and
the LangGraph turn graph) never load the obsolete router, planner tables,
or fail-closed path. Import those modules directly only from offline
evaluation tooling, never from production code.
"""

from __future__ import annotations

__all__: list[str] = []
