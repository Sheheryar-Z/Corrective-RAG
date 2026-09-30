"""LangGraph node implementations.

Each node is created by a factory that receives its dependencies (LLM chains,
retriever, search function). This keeps the nodes free of global state and lets
tests swap in fakes without API keys.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from .config import Settings
from .logic import Verdict, classify_retrieval, decompose, select_knowledge
from .state import CRAGState, Strip

Node = Callable[[CRAGState], dict[str, Any]]


def source_label(doc: Document) -> str:
    """Human-readable provenance for a document (file + page, or URL)."""
    meta = doc.metadata or {}
    if meta.get("url"):
        return meta["url"]
    src = meta.get("source")
    if src:
        name = Path(str(src)).name
        page = meta.get("page")
        return f"{name} p.{int(page) + 1}" if isinstance(page, int) else name
    return "unknown"


# ---------------------------------------------------------------------------
# 1. Retrieve
# ---------------------------------------------------------------------------
def make_retrieve_node(retriever: Runnable) -> Node:
    def retrieve(state: CRAGState) -> dict[str, Any]:
        return {"docs": retriever.invoke(state["question"])}

    return retrieve


# ---------------------------------------------------------------------------
# 2. Evaluate each retrieved chunk and decide CORRECT / INCORRECT / AMBIGUOUS
# ---------------------------------------------------------------------------
def make_evaluate_node(evaluator: Runnable, settings: Settings) -> Node:
    upper, lower = settings.upper_threshold, settings.lower_threshold

    def evaluate(state: CRAGState) -> dict[str, Any]:
        q = state["question"]
        docs = state.get("docs", [])

        # One LLM call per chunk, run in parallel (the notebook ran them sequentially).
        outputs = (
            evaluator.batch(
                [{"question": q, "chunk": d.page_content} for d in docs],
                config={"max_concurrency": settings.max_concurrency},
            )
            if docs
            else []
        )
        scores = [float(o.score) for o in outputs]

        verdict = classify_retrieval(scores, upper=upper, lower=lower)
        # Chunks above the lower threshold are "at least weakly relevant".
        good = [d for d, s in zip(docs, scores, strict=True) if s > lower]

        reasons = {
            Verdict.CORRECT: f"At least one retrieved chunk scored > {upper}.",
            Verdict.INCORRECT: (
                f"All retrieved chunks scored < {lower}."
                if docs
                else "Retriever returned no chunks."
            ),
            Verdict.AMBIGUOUS: f"No chunk scored > {upper}, but not all were < {lower}.",
        }
        return {
            "scores": scores,
            "good_docs": [] if verdict is Verdict.INCORRECT else good,
            "verdict": verdict.value,
            "reason": reasons[verdict],
        }

    return evaluate


def route_after_eval(state: CRAGState) -> str:
    """CORRECT skips the web entirely; the other two verdicts search first."""
    return "refine" if state["verdict"] == Verdict.CORRECT.value else "rewrite_query"


# ---------------------------------------------------------------------------
# 3. Rewrite the question into a keyword web query
# ---------------------------------------------------------------------------
def make_rewrite_node(rewriter: Runnable) -> Node:
    def rewrite_query(state: CRAGState) -> dict[str, Any]:
        out = rewriter.invoke({"question": state["question"]})
        return {"web_query": out.query}

    return rewrite_query


# ---------------------------------------------------------------------------
# 4. Web search
# ---------------------------------------------------------------------------
def make_web_search_node(search: Callable[[str], list[Document]]) -> Node:
    def web_search(state: CRAGState) -> dict[str, Any]:
        query = state.get("web_query") or state["question"]
        return {"web_docs": search(query)}

    return web_search


# ---------------------------------------------------------------------------
# 5. Knowledge refinement: decompose -> filter -> recompose
# ---------------------------------------------------------------------------
def make_refine_node(strip_filter: Runnable, settings: Settings) -> Node:
    def refine(state: CRAGState) -> dict[str, Any]:
        q = state["question"]
        verdict = Verdict(state["verdict"])
        docs = select_knowledge(verdict, state.get("good_docs", []), state.get("web_docs", []))

        # Decompose per document so every strip keeps its source.
        strips: list[Strip] = [
            {"text": text, "source": source_label(d)}
            for d in docs
            for text in decompose(d.page_content, settings.sentences_per_strip)
        ]

        # Filter: one keep/drop judgment per strip, in parallel.
        decisions = (
            strip_filter.batch(
                [{"question": q, "sentence": s["text"]} for s in strips],
                config={"max_concurrency": settings.max_concurrency},
            )
            if strips
            else []
        )
        kept = [s for s, d in zip(strips, decisions, strict=True) if d.keep]

        # Recompose: concatenate in original order, tagged with numbered sources.
        sources: list[str] = []
        lines: list[str] = []
        for s in kept:
            if s["source"] not in sources:
                sources.append(s["source"])
            lines.append(f"[{sources.index(s['source']) + 1}] {s['text']}")

        return {
            "strips": strips,
            "kept_strips": kept,
            "refined_context": "\n".join(lines),
            "sources": [f"[{i}] {src}" for i, src in enumerate(sources, 1)],
        }

    return refine


# ---------------------------------------------------------------------------
# 6. Generate
# ---------------------------------------------------------------------------
def make_generate_node(generator: Runnable) -> Node:
    def generate(state: CRAGState) -> dict[str, Any]:
        out = generator.invoke(
            {"question": state["question"], "context": state.get("refined_context", "")}
        )
        return {"answer": getattr(out, "content", out)}

    return generate
