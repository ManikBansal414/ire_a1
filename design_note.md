# Design Note — CS4.406 Assignment 1
## Lexical & Semantic Retrieval on EB-NeRD and MIND

**Author**: Manik Bansal (2024101084)  
**GitHub**: https://github.com/ManikBansal414/ire_a1

---

## 1. What I Built

`build_pipeline.py` is the single entry point that downloads MIND-small and
EB-NeRD demo, parses both datasets into a common schema, applies a temporal
split with an anti-gaming assertion, and writes a Parquet feature store under
`data/processed/{mind,ebnerd}/`.

**Unified schema** — both datasets are normalised to:
- *Articles*: `article_id, title, abstract, body, category, subcategory, entities, text`
  (where `text = title + " " + abstract` pre-built for lexical retrieval).
  List-type columns (`entities`, `history`, `candidates`, `labels`) are stored
  pipe-separated to avoid PyArrow list-column schema conflicts across OS and
  Python versions.
- *Impressions*: `impression_id, user_id, time, history, candidates, labels`
- *Users*: `user_id, history, recency_weights` — per-user recency weights
  (`decay^age`, most-recent = 1.0) are pre-computed and stored alongside
  the history so retrieval modules do not repeat the computation.

On top of the feature store, two retrieval modules run independently:

**BM25 (`src/retrieval/bm25_retriever.py`)** — wraps `rank_bm25.BM25Okapi`
inside a `BM25Index` class that stores the article-id ↔ corpus-index mapping.
It exposes two modes: *global retrieval* (top-K from the whole catalog,
for recall@K measurement) and *per-impression re-ranking*
(`rerank_candidates_bm25`) where each impression's own candidate set is
re-scored using `bm25.get_scores()` and sorted descending. The re-ranked
list is what feeds AUC/MRR/nDCG in the evaluation harness.

**Semantic (`src/retrieval/semantic_retriever.py`)** — uses
`sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalised) to
encode articles. A `FAISSIndex` class wraps `faiss.IndexFlatIP`, which gives
exact cosine similarity search (dot product on unit vectors). User vectors are
built on-the-fly via `_user_vector()`: exponentially decayed mean-pool of
clicked article embeddings, then re-normalised. A parallel `rerank_candidates_semantic`
function re-scores each impression's own candidates by direct dot-product
with the user vector, without calling FAISS search.

**Evaluation (`run_eval.py` + `src/evaluation/`)** reports:

| Metric | Implementation |
|--------|----------------|
| AUC | `roc_auc_score` on rank-position scores (`1/(rank+1)`) per impression, averaged |
| MRR | `1/rank` of first relevant item in the re-ranked list |
| nDCG@5, @10 | Standard DCG / ideal DCG over the re-ranked labels |
| Novelty | Mean `−log₂(pop(i)/N)` over top-10 recommended articles |
| ILD | Fraction of top-10 pairs from different categories (category-based proxy) |
| Coverage | Unique articles recommended / total catalog size |

Slicing (`src/evaluation/slicing.py`) uses a **dynamic cold/warm threshold**:
the 25th percentile of history lengths in the current split — this adapts
automatically because EB-NeRD has a mean history of ~281 clicks versus MIND's
~10 clicks, so a fixed threshold like 5 would label almost all EB-NeRD users
as "warm."

A LightGBM reranker (`generate_ebnerd_submission.py`) was used for the final
EB-NeRD Codabench submission, using 8 features: mean-user cosine sim, max
history cosine sim, recent-5 cosine sim, category/subcategory overlap
fractions, title Jaccard overlap, history length, and candidate position.
To handle the 11M test impressions, a `ebnerd_parallel_worker.py` splits
the file into three chunks across separate processes, batches LightGBM
predictions (`model.predict` on a large matrix at once), and merges outputs
with `merge_and_zip.py`.

---

## 2. Key Design Decisions

**Temporal split — never random (`src/pipeline/split.py`)**

The split cuts on wall-clock time: `test_start = t_max − 1 day`,
`val_start = test_start − 1 day`, everything before is train. After splitting,
`_assert_no_leakage()` raises immediately if `max(train.time) > min(val.time)`
or `max(val.time) > min(test.time)`. A random 80/20 split would allow training
on impressions from day 5 and testing on day 2, inflating all offline metrics
by letting the model "see the future." The assertion makes this impossible.

**Pipe-separated Parquet over PyArrow list columns**

Early iterations stored history/candidates/labels as native PyArrow list columns.
This caused schema-compatibility errors when loading files written on Linux in a
Windows Python environment (and vice-versa) — PyArrow serialises variable-length
list types differently across versions. Switching to pipe-separated strings
(`"|\n".join(...)` on write, `.split("|")` on load) is uglier but completely
portable and readable by any parquet reader.

**`BM25Index` with a `_id_to_idx` mapping**

`rank_bm25` internally stores term frequencies by corpus position, not by
article ID. The `BM25Index` class maintains a `Dict[str, int]` mapping from
`article_id` to its position in the corpus. This lets `rerank_candidates_bm25`
call `bm25.get_scores(query_tokens)` once and then look up individual candidate
scores by position — avoiding a full re-query per candidate.

**`FAISSIndex` with separate `.meta.pkl` file**

FAISS writes and reads binary index files natively via `faiss.write_index` /
`faiss.read_index`. But the article-id list and `_id_to_idx` dict are Python
objects, not stored in the FAISS binary. These are serialised separately to a
`.meta.pkl` file alongside the `.faiss.index` file. The `load()` classmethod
reconstructs both in one call. The `get_embedding()` method uses
`faiss.reconstruct()` to fetch stored vectors back from the index without
keeping a separate NumPy matrix in memory, which matters for the re-ranking step.

**Recency decay λ = 0.9, exponential, age computed from list tail**

Inside `_user_vector()`, age for article at position `i` in the history list
is `len(history) − 1 − i`, so the last element (most recent click) always gets
`0.9^0 = 1.0` and earlier clicks decay geometrically. With λ=0.9 an article
10 positions back has weight ~0.35, and 20 positions back ~0.12. Weights are
normalised to sum to 1 before the weighted average, then the result is
re-L2-normalised so the final dot product with article embeddings equals cosine
similarity. An alternative considered was linear weights (`1..N`), but exponential
decay better models how news interest fades; the specific λ was left at 0.9
because it is the standard default in recency-weighted retrieval literature.

**Dynamic cold-start threshold (25th percentile)**

`split_cold_warm()` computes `threshold = max(1, np.percentile(history_len, 25))`
on each split independently. For EB-NeRD this resolves to a much higher value
than for MIND, avoiding the situation where a fixed threshold of e.g. 5 clicks
would classify nearly all EB-NeRD users as "warm" and make the cold slice trivially
small and uninterpretable.

**LightGBM batch prediction for EB-NeRD scale**

The EB-NeRD test set contains ~11M impressions. Calling `model.predict(features)`
per impression individually incurs Python-level overhead on each call. The parallel
worker instead accumulates feature matrices from up to 15,000 impressions into a
single `np.vstack` and calls `model.predict` once per batch — this is 10–50× faster
because LightGBM's C++ backend amortises setup overhead across the batch.

---

## 3. Observations from Experiments

### recall@K — EB-NeRD val split

| Retriever | recall@50 | recall@100 | recall@200 |
|-----------|-----------|------------|------------|
| BM25      | 0.0066    | 0.0156     | 0.0288     |
| Semantic  | 0.0030    | 0.0063     | 0.0141     |

### recall@K — MIND val split

| Retriever | recall@50 | recall@100 | recall@200 |
|-----------|-----------|------------|------------|
| Semantic  | 0.0144    | 0.0238     | 0.0373     |

### Ranking metrics — EB-NeRD val split (re-ranked candidates)

| Retriever | Slice | AUC    | MRR    | nDCG@5 | nDCG@10 |
|-----------|-------|--------|--------|--------|---------|
| BM25      | all   | 0.5032 | 0.1185 | 0.1209 | 0.1209  |
| BM25      | cold  | 0.5021 | 0.1075 | 0.1088 | 0.1088  |
| BM25      | warm  | 0.5036 | 0.1224 | 0.1251 | 0.1251  |
| BM25      | tail  | 0.4881 | 0.000  | 0.000  | 0.000   |
| Semantic  | all   | 0.5105 | 0.3134 | 0.3484 | 0.4304  |
| Semantic  | cold  | 0.5224 | 0.3154 | 0.3567 | 0.4306  |
| Semantic  | warm  | 0.5066 | 0.3127 | 0.3456 | 0.4304  |
| Semantic  | tail  | 0.4454 | 0.2864 | 0.2999 | 0.3792  |

**BM25 wins on recall, semantic wins on ranking.**
BM25 recall@50 (0.0066) is more than double semantic (0.003) on EB-NeRD.
However, BM25's MRR of 0.12 is less than half semantic's 0.31. This
makes sense: BM25's inverted-index scores many articles as non-zero if
any query term matches anywhere in the catalog, casting a wide net but
placing the truly relevant item far down the list. The recency-weighted
cosine representation is more selective — it retrieves fewer candidates
but ranks them more accurately within the impression's candidate set.

**Semantic is more robust on cold-start users.**
Semantic AUC on cold users (0.522) is actually *higher* than on warm users
(0.507), while BM25 barely changes (0.502 vs 0.504). With only 1–2 clicked
articles, the BM25 query becomes a short 2–5 word string that either matches
or doesn't. The user vector built from even 1–2 embeddings still points in
a coherent topic direction in the 384-dim space, giving semantic retrieval a
relative advantage in the cold-start regime.

**BM25 completely fails on tail articles.**
BM25 AUC on tail impressions falls to 0.488 (below random) and MRR/nDCG drop
to exactly 0. Tail articles have rare or absent vocabulary in user histories —
the BM25 query tokens simply never appear in their `title + abstract` text, so
scores are uniformly near-zero and ranking is effectively random. Semantic
retrieval degrades on tail too (AUC 0.445) but retains non-zero MRR (0.286),
because embedding similarity is vocabulary-independent.

**EB-NeRD overall numbers are lower than expected for MIND.**
MIND semantic recall@50 (0.014) is nearly 5× higher than EB-NeRD (0.003).
Two structural reasons: the EB-NeRD catalog is far larger so the chance of
hitting a relevant article by cosine search is lower; and `all-MiniLM-L6-v2`
is an English model — EB-NeRD is Danish, so embeddings carry less semantic
signal, hurting both retrieval and re-ranking.

### Codabench Leaderboard Screenshots

**MIND Competition**

| My Submissions | Leaderboard |
|:-:|:-:|
| ![MIND submissions](mind_subs.png) | ![MIND leaderboard](mind_lead.png) |

**EB-NeRD Competition**

| My Submissions | Leaderboard |
|:-:|:-:|
| ![EB-NeRD submissions](ebnerd_subs.png) | ![EB-NeRD leaderboard](ebnerd_lead.png) |

---

## 4. Where the Pipeline Breaks at 10×

| Component | What it does now | Breaks because | Fix |
|-----------|-----------------|----------------|-----|
| `feature_store.py` — `pandas.read_parquet` | Loads full datasets into RAM | 10× EB-NeRD behaviors ≈ 30–50 GB → OOM | Replace with `polars.scan_parquet()` lazy evaluation; only materialise needed columns |
| `BM25Index` — `BM25Okapi` in RAM | Holds tokenised corpus + IDF in memory | 10× article vocab → sparse matrix can exceed 10 GB | Move index to Elasticsearch (sharded, persistent, REST-queryable) |
| `rerank_candidates_bm25` — `iterrows()` loop | Python loop over all impressions | 48M iterations at 10× → single-threaded, hours | Vectorise with `pandas.explode()` + `groupby()`, or push to Spark |
| `FAISSIndex` — `IndexFlatIP` | Exact O(N·d) cosine search | 700K articles × 384 dim × 650K users = ~100× today's cost | Switch to `IndexHNSWFlat` (graph-based ANN, sub-linear query time, <1% recall loss at `ef=64`) |
| `compute_embeddings` — sequential encode | Encodes articles one batch at a time | 10× more articles → 10× encode time on same GPU | Use larger batch size + multi-GPU `DataParallel`; or load pre-built EB-NeRD embeddings |
| `temporal_split` — `pandas` in-memory sort | Sorts all impressions by timestamp | 23M-row DataFrame sort → OOM | Stream-sort with DuckDB (`ORDER BY time`) or partition parquet by date at ingest |
| `_assert_no_leakage` | Checks `max/min` on full split DataFrames | Fine — O(N) scan, fast at any scale | No change needed |
