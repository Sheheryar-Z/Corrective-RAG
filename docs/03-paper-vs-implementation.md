# 3. Paper vs Implementation

A reviewer who has read the CRAG paper will spot the differences below. They're
listed here openly because knowing the gap between a paper and your build is part
of understanding it.

## Where this project follows the paper

- The three-way action trigger (CORRECT / INCORRECT / AMBIGUOUS) with an upper and
  a lower threshold, using the paper's exact rules: "any above upper" and
  "all below lower".
- Knowledge per action: internal only, web only, or both.
- Decompose-filter-recompose refinement applied to internal **and** web knowledge.
- Keyword-style query rewriting before web search.

## Where it deliberately differs

| Aspect | Paper | This project | Consequence |
|---|---|---|---|
| Retrieval evaluator | Fine-tuned T5-large (0.77B), trained on PopQA relevance labels | `gpt-4o-mini` prompted for a 0 to 1 score | No training needed, but the paper found prompted ChatGPT judged relevance far less accurately (58 to 65% vs 84%). Scores are uncalibrated. |
| Score scale / thresholds | [-1, 1]; thresholds tuned per dataset, e.g. (0.59, -0.99) for PopQA | [0, 1]; thresholds fixed at 0.7 / 0.3 | Thresholds are a guess, not tuned on data. |
| Strip size | "A few sentences", depending on length | 1 sentence by default (configurable) | Sentences starting with "It" or "This" lose their subject and get filtered out. |
| Strip filter | Same T5 evaluator; keep top-5 strips above a threshold | Separate LLM keep/drop call per strip, no top-k cap | Many LLM calls per question; context size unbounded. |
| Web search | Google Search API, full page content, Wikipedia preferred | Tavily snippets, no source preference | Less text per result, no authority filtering. |
| Evaluation | 4 benchmarks with accuracy / FactScore | One example question | No evidence yet that this build beats plain RAG. |

## Changes from the original notebook to the package

| Change | Why |
|---|---|
| Nodes built from injected components | Testable without API keys. |
| Evaluator and filter calls use `.batch()` in parallel | The notebook ran 20 to 60 LLM calls sequentially. |
| Empty retrieval returns INCORRECT | The notebook returned AMBIGUOUS when zero chunks came back, even though there was no internal knowledge to use. |
| Strips decomposed per document and tagged with source | The notebook joined all docs into one string before splitting, losing provenance, and web results embedded `TITLE:` / `URL:` lines that were split into fake "sentences". |
| Answer cites `[n]` source tags; `sources` returned | Answers were uncitable. |
| FAISS index persisted to `.index/` | The notebook re-embedded every PDF on each run, costing time and money. |
| `langchain_tavily.TavilySearch` | `TavilySearchResults` is deprecated. |
| Rewrite prompt drops the "(last 30 days)" rule | Adding that text to a query string does not actually filter by date in Tavily; it only pollutes the query. |
| `CRAGState` uses `total=False` | No need to pre-fill every key with an empty value when invoking. |
| Settings via environment variables | Tune without editing code. |

## Known weaknesses

These are the most important issues, in order of impact.

### 1. There is no evaluation, so no quality claim can be made

The whole point of CRAG is "better answers than plain RAG". This project currently
has one example question. **Suggested next step:** write 30 to 50 questions over your
PDFs (some answerable from the books, some not, some comparisons), run plain RAG vs
CRAG, and report answer correctness plus the verdict distribution. Even a small,
hand-graded table is far stronger evidence than a demo.

### 2. The evaluator prompt is biased against comparison and multi-part questions

The prompt defines 1.0 as "this chunk **alone** is sufficient to answer". A question
like "BN vs LN" can almost never be answered by a single chunk, because the two
halves usually live in different sections of a book. So comparison questions will
rarely reach CORRECT and will be sent to the web even when the books contain
everything needed. The example run's AMBIGUOUS verdict is consistent with this.
**Fix options:** score "contains information useful for answering" instead of
"sufficient alone", or decompose multi-part questions before retrieval.

### 3. LLM-as-judge scores are uncalibrated

`0.7` from `gpt-4o-mini` has no stable meaning; it varies with prompt wording and
model version. **Fix:** label ~100 (question, chunk) pairs yourself, then choose
thresholds from that data, or use a cross-encoder reranker (for example a
`bge-reranker` model), whose scores are more consistent and far cheaper.

### 4. The strip filter is expensive

One LLM call per sentence. **Fix:** one call per document that returns the indices
of relevant sentences, or a reranker model, or embedding similarity with a cutoff.

### 5. Web content is trusted once it passes the filter

The filter checks relevance, not truth. The example answer includes a web-sourced
claim that BN is "generally faster for larger networks", which is vague at best.
Relevance filtering is not fact-checking.

### 6. The sentence splitter is naive

The regex breaks on "e.g.", "Fig. 3", "et al." and on equations, all common in ML
textbooks. `pysbd` or spaCy's sentencizer are drop-in improvements.
