"""Unicode-safe Marathi preprocessing and pair input construction."""
from __future__ import annotations

import unicodedata
from typing import Any

REQUIRED_COLUMNS = ["sentence", "target_word", "lemma", "pos", "sense_id", "gloss", "label"]


def normalize_text(value: Any) -> str:
    """Normalize canonical Unicode and whitespace without removing Marathi text."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return " ".join(text.split())


def matching_occurrences(sentence: str, target: str) -> list[tuple[int, int]]:
    """Return exact NFC, case-sensitive target spans (no stemming or fuzzy matching)."""
    sentence, target = normalize_text(sentence), normalize_text(target)
    if not target:
        return []
    spans, offset = [], 0
    while True:
        found = sentence.find(target, offset)
        if found < 0:
            return spans
        spans.append((found, found + len(target)))
        offset = found + max(1, len(target))


def mark_target(sentence: str, target: str, occurrence: int = 0) -> str:
    """Mark a selected exact occurrence for transparent model conditioning."""
    sentence, target = normalize_text(sentence), normalize_text(target)
    spans = matching_occurrences(sentence, target)
    if not spans:
        raise ValueError(f"Target word {target!r} does not occur in the sentence (exact NFC match).")
    if occurrence < 0 or occurrence >= len(spans):
        raise ValueError(f"Occurrence must be between 1 and {len(spans)}.")
    start, end = spans[occurrence]
    return sentence[:start] + f" [TGT] {sentence[start:end]} [/TGT] " + sentence[end:]


def mark_target_if_present(sentence: str, target: str, occurrence: int = 0) -> str:
    """For corpus training, retain context when inflection means exact marking is impossible."""
    try:
        return mark_target(sentence, target, occurrence)
    except ValueError:
        return normalize_text(sentence)


def pair_text(row: dict[str, Any]) -> str:
    """Stable baseline representation retaining sentence, target and candidate gloss."""
    sentence = mark_target_if_present(row["sentence"], row["target_word"], int(row.get("_occurrence", 0)))
    return f"sentence: {sentence} target: {normalize_text(row['target_word'])} lemma: {normalize_text(row.get('lemma',''))} pos: {normalize_text(row.get('pos',''))} definition: {normalize_text(row['gloss'])}"

