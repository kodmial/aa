"""Russian query normalization for RU-first retrieval (issue #17).

Deterministic, local, keyless text handling for the lexical branch:

- casefold + ``ё`` → ``е`` unification;
- Cyrillic/Latin token extraction;
- light rule-based Russian stemming so inflected forms
  (``алкоголизма``/``алкоголизме``) meet the indexed base form.

The stemmer is intentionally conservative: it strips only well-known
inflectional endings and never reduces a stem below three characters.
It is a retrieval aid, never evidence, and never modifies stored text.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[а-яa-z0-9]+")

# Longest-first inflectional endings (nouns, adjectives, verbs).
# Applied at most once per token, keeping a minimum stem of 3 chars.
_SUFFIXES = sorted(
    {
        "ами",
        "ями",
        "ах",
        "ях",
        "ов",
        "ев",
        "ее",
        "ие",
        "ые",
        "ое",
        "ей",
        "ий",
        "ый",
        "ой",
        "ем",
        "им",
        "ом",
        "ам",
        "ям",
        "ью",
        "ю",
        "ать",
        "ять",
        "еть",
        "ут",
        "ют",
        "ат",
        "ят",
        "ил",
        "ыл",
        "ел",
        "ла",
        "ло",
        "ли",
        "ого",
        "его",
        "ому",
        "ему",
        "ыми",
        "ими",
        "ую",
        "юю",
        "ая",
        "яя",
        "ия",
        "ии",
        "ию",
        "а",
        "о",
        "у",
        "ы",
        "и",
        "е",
        "ь",
        "й",
        "я",
    },
    key=len,
    reverse=True,
)


_VOWEL_REPEAT_RE = re.compile(r"([аеиоуыэюяaeiou])\1+", re.IGNORECASE)

# Colloquial spoken forms mapped to their literary base for retrieval
# only (generic slang, never an exact-question list): ``че`` is the
# colloquial form of ``что`` and otherwise fragments BM25 recall on
# colloquial turns while the dense branch alone must carry them.
_COLLOQUIAL_TOKEN_MAP = {
    "че": "что",
}


def normalize_ru(text: str) -> str:
    """Normalize Russian text for retrieval (casefold, ``ё`` → ``е``).

    A repeated-vowel collapse (``ее`` → ``е``) absorbs generic
    doubled-vowel typos on both the index and query sides consistently;
    double consonants (``ссора``, ``поддержка``) are preserved so their
    stems stay distinct from single-consonant forms.
    """
    folded = text.casefold().replace("ё", "е")
    return _VOWEL_REPEAT_RE.sub(r"\1", folded)


def ru_tokens(text: str) -> list[str]:
    """Extract normalized alphanumeric tokens from ``text``."""
    tokens = _TOKEN_RE.findall(normalize_ru(text))
    return [_COLLOQUIAL_TOKEN_MAP.get(token, token) for token in tokens]


def ru_stem(token: str) -> str:
    """Return the light stem of one normalized ``token``.

    Suffixes apply iteratively (longest first) so conjugated forms such
    as ``бухаю`` reduce to the same stem as the infinitive ``бухать``
    (``бухаю`` → ``буха`` → ``бух``). Stems never shrink below 3 chars.

    Craving-root unification: verb inflections (``тянет``/``тяну``) and
    the noun family (``тяга``) share one retrieval root (``тяг``) so
    inflected and typo-doubled forms meet the same FTS term on both the
    index and query sides. Generic root only, never an exact question.
    """
    token = normalize_ru(token)
    if len(token) <= 3:
        if token.startswith("тян"):
            return "тяг"
        return token
    changed = True
    while changed and len(token) > 3:
        changed = False
        for suffix in _SUFFIXES:
            if len(suffix) >= len(token):
                continue
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                token = token[: len(token) - len(suffix)]
                changed = True
                break
    if token.startswith("тян"):
        return "тяг"
    return token


def stemmed_norm_text(text: str) -> str:
    """Return space-joined stemmed tokens for FTS5 indexing/search."""
    return " ".join(ru_stem(token) for token in ru_tokens(text))
