"""The shared state object that flows through the LangGraph workflow."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class Strip(TypedDict):
    """A small unit of knowledge plus where it came from."""

    text: str
    source: str


class CRAGState(TypedDict, total=False):
    # Input
    question: str

    # Retrieval + evaluation
    docs: list[Document]
    scores: list[float]
    good_docs: list[Document]
    verdict: str
    reason: str

    # Web search (INCORRECT / AMBIGUOUS paths)
    web_query: str
    web_docs: list[Document]

    # Knowledge refinement (decompose -> filter -> recompose)
    strips: list[Strip]
    kept_strips: list[Strip]
    refined_context: str

    # Output
    answer: str
    sources: list[str]
