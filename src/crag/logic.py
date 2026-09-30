"""Pure, dependency-free decision logic for CRAG.

Everything in this module is deterministic and has no LLM / network calls,
which makes it the easiest part of the system to unit-test.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import Enum
from typing import TypeVar

T = TypeVar("T")


class Verdict(str, Enum):
    """The three retrieval actions from the CRAG paper (Section 4.3)."""

    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    AMBIGUOUS = "AMBIGUOUS"


def classify_retrieval(scores: Sequence[float], upper: float, lower: float) -> Verdict:
    """Turn per-document relevance scores into a single retrieval verdict.

    Rules (paper, Section 4.3):
      * CORRECT   -> at least one document scores above ``upper``
      * INCORRECT -> every document scores below ``lower``
      * AMBIGUOUS -> anything else

    An empty score list (the retriever returned nothing) is treated as
    INCORRECT: there is no internal knowledge to trust, so we must go to the web.
    """
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(f"Expected 0 <= lower < upper <= 1, got lower={lower}, upper={upper}")

    if not scores:
        return Verdict.INCORRECT
    if any(s > upper for s in scores):
        return Verdict.CORRECT
    if all(s < lower for s in scores):
        return Verdict.INCORRECT
    return Verdict.AMBIGUOUS


def select_knowledge(verdict: Verdict, internal: list[T], external: list[T]) -> list[T]:
    """Pick which knowledge sources feed refinement for a given verdict.

    CORRECT -> internal only, INCORRECT -> web only, AMBIGUOUS -> both.
    """
    if verdict is Verdict.CORRECT:
        return list(internal)
    if verdict is Verdict.INCORRECT:
        return list(external)
    return list(internal) + list(external)


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str, min_chars: int = 20) -> list[str]:
    """Naive regex sentence splitter (same behaviour as the original notebook).

    Known weakness: splits on abbreviations like "e.g." or "Fig. 3". Good enough
    for a demo; swap in a proper segmenter (e.g. ``pysbd``) for production.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > min_chars]


def decompose(text: str, sentences_per_strip: int = 1, min_chars: int = 20) -> list[str]:
    """Decompose text into "knowledge strips" (paper, Section 4.4).

    ``sentences_per_strip=1`` reproduces the notebook (one sentence per strip).
    The paper uses strips of "a few sentences"; larger strips keep pronouns
    attached to their antecedents, so filtering loses less context.
    """
    if sentences_per_strip < 1:
        raise ValueError("sentences_per_strip must be >= 1")
    sentences = split_sentences(text, min_chars=min_chars)
    return [
        " ".join(sentences[i : i + sentences_per_strip])
        for i in range(0, len(sentences), sentences_per_strip)
    ]
