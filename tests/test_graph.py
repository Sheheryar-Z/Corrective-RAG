"""End-to-end graph tests with fake components: no API keys, no network."""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from crag.config import Settings
from crag.graph import Components, build_graph
from crag.nodes import source_label

RELEVANT = "Layer normalization normalizes across features for each example."
IRRELEVANT = "The mitochondria is the powerhouse of the cell, as every student knows."
WEB = "Batch normalization depends on mini-batch statistics during training."


def make_components(chunk_scores, web_calls):
    """Build fakes. `chunk_scores` maps chunk text -> evaluator score."""
    docs = [
        Document(page_content=text, metadata={"source": "data/book1.pdf", "page": i})
        for i, text in enumerate(chunk_scores)
    ]

    def search(query):
        web_calls.append(query)
        return [Document(page_content=WEB, metadata={"url": "https://example.com/bn"})]

    return Components(
        retriever=RunnableLambda(lambda q: docs),
        evaluator=RunnableLambda(lambda x: SimpleNamespace(score=chunk_scores[x["chunk"]])),
        # Keep any strip mentioning "normalization".
        strip_filter=RunnableLambda(
            lambda x: SimpleNamespace(keep="normalization" in x["sentence"].lower())
        ),
        rewriter=RunnableLambda(lambda x: SimpleNamespace(query="bn vs ln keywords")),
        web_search=search,
        generator=RunnableLambda(lambda x: SimpleNamespace(content=f"ANSWER<{x['context']}>")),
    )


def run(chunk_scores):
    web_calls = []
    app = build_graph(make_components(chunk_scores, web_calls), Settings())
    return app.invoke({"question": "BN vs LN?"}), web_calls


def test_correct_path_skips_web():
    res, web_calls = run({RELEVANT: 0.9, IRRELEVANT: 0.1})
    assert res["verdict"] == "CORRECT"
    assert web_calls == []
    assert "web_query" not in res
    assert res["refined_context"] == f"[1] {RELEVANT}"
    assert res["sources"] == ["[1] book1.pdf p.1"]


def test_incorrect_path_uses_web_only():
    res, web_calls = run({RELEVANT: 0.1, IRRELEVANT: 0.0})
    assert res["verdict"] == "INCORRECT"
    assert web_calls == ["bn vs ln keywords"]
    assert res["good_docs"] == []
    assert res["refined_context"] == f"[1] {WEB}"
    assert res["sources"] == ["[1] https://example.com/bn"]


def test_ambiguous_path_combines_internal_and_web():
    res, web_calls = run({RELEVANT: 0.5, IRRELEVANT: 0.1})
    assert res["verdict"] == "AMBIGUOUS"
    assert len(web_calls) == 1
    # Internal strip first, then web strip, each with its own source tag.
    assert res["refined_context"].splitlines() == [f"[1] {RELEVANT}", f"[2] {WEB}"]
    assert len(res["strips"]) == 2  # irrelevant chunk (0.1 < lower) never reached refinement


def test_filter_drops_irrelevant_strips():
    res, _ = run({RELEVANT: 0.9, IRRELEVANT: 0.5})
    assert len(res["strips"]) == 2
    assert [s["text"] for s in res["kept_strips"]] == [RELEVANT]


def test_generator_receives_refined_context():
    res, _ = run({RELEVANT: 0.9})
    assert res["answer"] == f"ANSWER<[1] {RELEVANT}>"


@pytest.mark.parametrize(
    "meta,expected",
    [
        ({"url": "https://a.b"}, "https://a.b"),
        ({"source": "/x/y/book2.pdf", "page": 4}, "book2.pdf p.5"),
        ({"source": "notes.txt"}, "notes.txt"),
        ({}, "unknown"),
    ],
)
def test_source_label(meta, expected):
    assert source_label(Document(page_content="x", metadata=meta)) == expected
