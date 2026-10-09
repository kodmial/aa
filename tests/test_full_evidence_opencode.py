"""Regression: OpenCode must see each complete selected book passage.

This guards immutable Product Contract #110: generation and verification
may not silently discard lower-ranked passages or cut the source text.
"""

from aa.conversation.prompt_builder import EvidencePassage, render_turn_context
from aa.conversation.response_units import ResponseUnitDraft
from aa.conversation.verifier import build_single_unit_text


def _source_passages(count: int = 10) -> list[EvidencePassage]:
    return [
        EvidencePassage(
            passage_id=f"chapter-3#exp{i:04d}",
            source="canonical-ru",
            section="chapter-3",
            text=f"Начало фрагмента {i}. " + ("Подлинный книжный контекст. " * 35) + f" Конец фрагмента {i}.",
        )
        for i in range(count)
    ]


def test_answer_prompt_preserves_every_selected_full_passage() -> None:
    passages = _source_passages()
    prompt = render_turn_context(
        summary="",
        passages=passages,
        user_message="Что мне делать, чтобы бросить пить?",
    )
    for passage in passages:
        assert passage.text in prompt
        assert f'id="{passage.passage_id}"' in prompt


def test_opencode_verifier_sees_same_complete_source_pack() -> None:
    passages = _source_passages()
    data = [
        {
            "passage_id": passage.passage_id,
            "source_id": passage.source,
            "section_id": passage.section,
            "text": passage.text,
        }
        for passage in passages
    ]
    draft = "Я хочу разобраться с тем, как прекратить пить."
    unit = ResponseUnitDraft("u1", draft, 0, len(draft))
    verifier_input = build_single_unit_text(
        unit=unit,
        passages=data,
        resolved_intent="хочу прекратить пить",
        user_message="Что мне делать?",
    )
    for number, passage in enumerate(passages, start=1):
        assert passage.text in verifier_input
        assert f'id="p{number}"' in verifier_input
