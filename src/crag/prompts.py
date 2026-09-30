"""Prompt templates and structured-output schemas used by each LLM step."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Retrieval evaluator: scores ONE chunk against the question
# ---------------------------------------------------------------------------
class DocEvalScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Relevance score in [0, 1]")
    reason: str = Field(description="One short sentence justifying the score")


DOC_EVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON only.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)


# ---------------------------------------------------------------------------
# 2. Strip filter: keep or drop one knowledge strip
# ---------------------------------------------------------------------------
class KeepOrDrop(BaseModel):
    keep: bool


FILTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the text directly helps answer the question.\n"
            "Use ONLY the text. Output JSON only.",
        ),
        ("human", "Question: {question}\n\nText:\n{sentence}"),
    ]
)


# ---------------------------------------------------------------------------
# 3. Query rewriter: question -> keyword web query
# ---------------------------------------------------------------------------
class WebQuery(BaseModel):
    query: str


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6-14 words).\n"
            "- Add the key technical terms a search engine would need.\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query",
        ),
        ("human", "Question: {question}"),
    ]
)


# ---------------------------------------------------------------------------
# 4. Generator: answers ONLY from the refined context
# ---------------------------------------------------------------------------
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
            "Each context line is prefixed with a source tag like [1].\n"
            "Cite the tags you rely on, e.g. 'LayerNorm is batch-independent [2].'\n"
            "If the context is empty or insufficient, say: 'I don't know.'",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)
