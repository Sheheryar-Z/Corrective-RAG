"""Wiring: build real components (OpenAI, FAISS, Tavily) and compile the graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from .config import Settings
from .nodes import (
    make_evaluate_node,
    make_generate_node,
    make_refine_node,
    make_retrieve_node,
    make_rewrite_node,
    make_web_search_node,
    route_after_eval,
)
from .state import CRAGState


@dataclass
class Components:
    """Everything the graph depends on. Swap any field for a fake in tests."""

    retriever: Runnable  # str -> List[Document]
    evaluator: Runnable  # {question, chunk} -> obj with .score
    strip_filter: Runnable  # {question, sentence} -> obj with .keep
    rewriter: Runnable  # {question} -> obj with .query
    web_search: Callable[[str], list[Document]]
    generator: Runnable  # {question, context} -> message with .content


def build_graph(components: Components, settings: Settings | None = None):
    """Compile the CRAG LangGraph.

    START -> retrieve -> evaluate --CORRECT--------------------> refine -> generate -> END
                                \\--INCORRECT/AMBIGUOUS-> rewrite_query -> web_search -/
    """
    settings = settings or Settings()
    g = StateGraph(CRAGState)

    g.add_node("retrieve", make_retrieve_node(components.retriever))
    g.add_node("evaluate", make_evaluate_node(components.evaluator, settings))
    g.add_node("rewrite_query", make_rewrite_node(components.rewriter))
    g.add_node("web_search", make_web_search_node(components.web_search))
    g.add_node("refine", make_refine_node(components.strip_filter, settings))
    g.add_node("generate", make_generate_node(components.generator))

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "evaluate")
    g.add_conditional_edges(
        "evaluate",
        route_after_eval,
        {"refine": "refine", "rewrite_query": "rewrite_query"},
    )
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "refine")
    g.add_edge("refine", "generate")
    g.add_edge("generate", END)

    return g.compile()


def make_tavily_search(max_results: int) -> Callable[[str], list[Document]]:
    """Tavily web search returning LangChain Documents with URL metadata."""
    from langchain_tavily import TavilySearch

    tool = TavilySearch(max_results=max_results)

    def search(query: str) -> list[Document]:
        raw = tool.invoke({"query": query})
        results = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
        docs: list[Document] = []
        for r in results:
            content = r.get("content") or r.get("snippet") or ""
            if content.strip():
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"url": r.get("url", ""), "title": r.get("title", "")},
                    )
                )
        return docs

    return search


def build_default_components(settings: Settings, rebuild_index: bool = False) -> Components:
    """Real components: OpenAI chat + embeddings, FAISS over your PDFs, Tavily search."""
    from langchain_openai import ChatOpenAI

    from .indexing import load_or_build_retriever
    from .prompts import (
        ANSWER_PROMPT,
        DOC_EVAL_PROMPT,
        FILTER_PROMPT,
        REWRITE_PROMPT,
        DocEvalScore,
        KeepOrDrop,
        WebQuery,
    )

    llm = ChatOpenAI(model=settings.llm_model, temperature=0)
    return Components(
        retriever=load_or_build_retriever(settings, rebuild=rebuild_index),
        evaluator=DOC_EVAL_PROMPT | llm.with_structured_output(DocEvalScore),
        strip_filter=FILTER_PROMPT | llm.with_structured_output(KeepOrDrop),
        rewriter=REWRITE_PROMPT | llm.with_structured_output(WebQuery),
        web_search=make_tavily_search(settings.web_max_results),
        generator=ANSWER_PROMPT | llm,
    )
