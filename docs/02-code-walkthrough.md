# 2. Code Walkthrough

This page follows one question through the system and explains what each file does.
The notebook in [`notebooks/`](../notebooks/corrective_rag.ipynb) is the original
exploratory version; `src/crag/` is the same pipeline restructured as a package.

## Project layout

```
src/crag/
├── config.py     # Settings dataclass; every value overridable via env vars
├── logic.py      # Pure functions: verdict rules, knowledge selection, decomposition
├── prompts.py    # Prompt templates + Pydantic output schemas for each LLM step
├── state.py      # CRAGState: the dict that flows through the graph
├── nodes.py      # One factory per graph node (retrieve, evaluate, refine, ...)
├── graph.py      # Wires nodes into a LangGraph; builds real OpenAI/FAISS/Tavily parts
├── indexing.py   # PDF loading, chunking, FAISS build + persist/reload
└── cli.py        # `python -m crag "question"`
```

**Design choice:** nodes never create their own LLM clients. `graph.build_graph()`
receives a `Components` object (retriever, evaluator, filter, rewriter, search,
generator). Production code passes real OpenAI/Tavily objects; the tests pass tiny
fakes. That is why the test suite runs in under a second with no API keys.

## The state

Every node reads from and writes to one shared dictionary, `CRAGState`
([`state.py`](../src/crag/state.py)). LangGraph merges each node's returned dict into it.

| Field | Written by | Meaning |
|---|---|---|
| `question` | caller | the user's question |
| `docs` | retrieve | top-k chunks from FAISS |
| `scores` | evaluate | one relevance score per chunk |
| `good_docs` | evaluate | chunks scoring above the lower threshold |
| `verdict`, `reason` | evaluate | CORRECT / INCORRECT / AMBIGUOUS and why |
| `web_query` | rewrite_query | keyword query for the search engine |
| `web_docs` | web_search | search results as Documents |
| `strips`, `kept_strips` | refine | all knowledge strips, and the ones kept |
| `refined_context` | refine | kept strips, joined, tagged `[1]`, `[2]`... |
| `sources` | refine | what each tag refers to (PDF page or URL) |
| `answer` | generate | final answer |

## Step by step

Running example: `"Batch normalization vs layer normalization"` over three PDFs in `data/`.

### Step 0: Indexing (once) — `indexing.py`

1. Load every `data/*.pdf` with `PyPDFLoader` (one Document per page).
2. Split into ~900-character chunks with 150-character overlap.
3. Strip invalid Unicode left over from PDF extraction.
4. Embed with `text-embedding-3-large` and store in FAISS.
5. Save the index to `.index/` so the next run loads it instead of re-embedding.

### Step 1: Retrieve — `nodes.make_retrieve_node`

Cosine-similarity search returns the top 4 chunks. At this point nothing checks
whether they are *actually* relevant; that's the next step's job.

### Step 2: Evaluate — `nodes.make_evaluate_node` + `logic.classify_retrieval`

Each chunk goes to the LLM with the `DOC_EVAL_PROMPT`, which asks for a score where
1.0 means "this chunk alone answers the question". Calls run in parallel via
`.batch()`. The scores are then turned into a verdict by a pure function:

```python
if not scores:                     return INCORRECT   # nothing retrieved
if any(s > upper for s in scores): return CORRECT
if all(s < lower for s in scores): return INCORRECT
return AMBIGUOUS
```

In the example run, no chunk passed 0.7 but some passed 0.3, so the verdict was
**AMBIGUOUS**.

### Step 3: Route — `nodes.route_after_eval`

- CORRECT goes straight to `refine`.
- INCORRECT and AMBIGUOUS go to `rewrite_query` first.

### Step 4: Rewrite query — `nodes.make_rewrite_node`

The LLM turns the question into a short keyword query. (In the example it returned
the question unchanged, because it was already keyword-shaped.)

### Step 5: Web search — `graph.make_tavily_search`

Tavily returns up to 5 results. Each becomes a `Document` whose `page_content` is the
result text and whose metadata holds the URL, so provenance survives to the answer.

### Step 6: Refine — `nodes.make_refine_node`

1. **Select** knowledge by verdict (`logic.select_knowledge`): CORRECT uses internal
   docs, INCORRECT uses web docs, AMBIGUOUS uses both (internal first).
2. **Decompose** each document into strips (`logic.decompose`; one sentence per strip
   by default).
3. **Filter**: one LLM keep/drop call per strip, in parallel.
4. **Recompose** kept strips, tagging each with a numbered source:

```
[1] Layer normalization normalizes across the features of each example...
[2] Batch normalization depends on mini-batch statistics...
```

### Step 7: Generate — `nodes.make_generate_node`

The generator is instructed to answer **only** from the refined context, cite
source tags, and say "I don't know." if the context is insufficient.

## LLM calls per question

For the default settings (k=4, 5 web results), a single question costs roughly:

| Step | Calls |
|---|---|
| Evaluate | 4 (one per chunk) |
| Rewrite | 0 or 1 |
| Filter | one per strip, typically 20 to 60 |
| Generate | 1 |

The strip filter dominates cost and latency. The package runs these calls in
parallel (`CRAG_MAX_CONCURRENCY`), but the count itself is a design weakness; see
[Paper vs Implementation](03-paper-vs-implementation.md#known-weaknesses).

## Running and inspecting

```bash
python -m crag "Batch normalization vs layer normalization" --trace
```

`--trace` prints the verdict, per-chunk scores, the web query, and how many strips
survived filtering. It is the quickest way to understand *why* the system chose a path.
