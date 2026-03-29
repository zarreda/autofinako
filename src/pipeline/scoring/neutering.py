"""Entity masking (neutering) for earnings call sentences.

Replaces named entities (PERSON, ORG, DATE, ...) with placeholder tokens
using spaCy NER. The spaCy model is loaded lazily so the import cost is
zero when neutering is not used.
"""

from __future__ import annotations

import re
from typing import Any

_ENTITY_MAP: dict[str, str] = {
    "PERSON": "[PERSON]",
    "ORG": "[ORG]",
    "DATE": "[DATE]",
    "PRODUCT": "[PRODUCT]",
    "EVENT": "[EVENT]",
    "LOC": "[LOC]",
    "NORP": "[NORP]",
    "LANGUAGE": "[LANGUAGE]",
    "LAW": "[LAW]",
    "FAC": "[FAC]",
    "TIME": "[TIME]",
    "WORK_OF_ART": "[WORK_OF_ART]",
}

_NEUTER_TOKENS: set[str] = set(_ENTITY_MAP.values())

_nlp: Any = None


def _get_nlp() -> Any:
    """Lazily load the spaCy model on first use."""
    global _nlp  # noqa: PLW0603
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
    return _nlp


def _apply_replacements(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Apply entity replacements to *text*, handling overlaps."""
    spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
    filtered: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, repl in spans:
        if start >= last_end:
            filtered.append((start, end, repl))
            last_end = end

    for start, end, repl in reversed(filtered):
        text = text[:start] + repl + text[end:]
    return text


def _drop_duplicated_tokens(text: str) -> str:
    """Remove consecutive duplicate neutering tokens."""
    for token in _NEUTER_TOKENS:
        escaped = re.escape(token)
        pattern = rf"({escaped})(?:\s*[,;]?\s*{escaped})+"
        text = re.sub(pattern, r"\1", text)
    return text


def neuter(sentence: str) -> str:
    """Replace named entities in *sentence* with placeholder tokens."""
    if not sentence or not sentence.strip():
        return sentence

    nlp = _get_nlp()
    doc = nlp(sentence)

    spans: list[tuple[int, int, str]] = []
    for ent in doc.ents:
        if ent.label_ in _ENTITY_MAP:
            spans.append((ent.start_char, ent.end_char, _ENTITY_MAP[ent.label_]))

    if not spans:
        return sentence

    result = _apply_replacements(sentence, spans)
    result = _drop_duplicated_tokens(result)
    return result
