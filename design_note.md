# Design Note — CS4.406 Assignment 1
## Lexical & Semantic Retrieval on EB-NeRD and MIND

**Author**: [Your Name]  
**Date**: August 2026  

---

## 1. What I Built

I built a reproducible two-stage candidate retrieval pipeline for news recommendation on both the MIND-small and EB-NeRD demo datasets. The pipeline produces article rankings from user click histories using two complementary approaches:

1. **Lexical retrieval (BM25)** — an inverted index over article titles and abstracts, queried using the concatenated titles of a user's most recent clicked articles.
2. **Semantic retrieval** — a FAISS flat index over sentence-transformer embeddings (all-MiniLM-L6-v2), with user representation computed as a recency-weighted mean-pool of clicked article embeddings.

### Key design choices

| Choice | Rationale |
|--------|-----------|
| Temporal split (never random) | News data has strong recency bias; random splits cause future-click leakage |
| BM25Okapi (rank_bm25) | Well-calibrated k1/b defaults; faster than TF-IDF for large corpora |
| Query = last 5 clicked titles | Balances recency with enough signal; early titles are often topic-setting |
| all-MiniLM-L6-v2 (384-dim) | Fast on CPU; strong BEIR benchmarks; fits in memory for demo/small scale |
| FAISS IndexFlatIP (exact) | Demo/small scale → brute-force is fast enough; no approximation error |
| Recency decay (λ=0.9) | Most-recent click most predictive; exponential decay is a natural prior |
| Pipe-separated parquet storage | Avoids pyarrow list-column schema issues across OS/Python versions |

---

## 2. Alternatives Considered

### BM25 query construction
- **Considered**: TF-IDF weighted query, entity-only query, full history
- **Chose**: Last-5-titles concatenation — simple, effective, directly interpretable
- **Why not full history**: BM25 is sensitive to query length; long queries dilute signal

### Semantic model
- **Considered**: XLM-RoBERTa (for EB-NeRD Danish text), EB-NeRD's provided Word2Vec embeddings
- **Chose**: all-MiniLM-L6-v2 for MIND; note EB-NeRD provides pre-trained Danish embeddings which should be preferred for production
- **Limitation**: MiniLM is English-only; EB-NeRD results will degrade

### ANN index
- **Considered**: FAISS HNSW (approximate), ScaNN
- **Chose**: IndexFlatIP (exact) — justified at demo/small scale (<200K articles); switch to HNSW at 10× scale

---

## 3. Observations from Experiments

*(Fill in after running the pipeline — template below)*

### recall@K comparison

| Dataset | Retriever | recall@50 | recall@100 | recall@200 |
|---------|-----------|-----------|------------|------------|
| MIND    | BM25      | —         | —          | —          |
| MIND    | Semantic  | —         | —          | —          |
| EB-NeRD | BM25      | —         | —          | —          |
| EB-NeRD | Semantic  | —         | —          | —          |

### Ranking metrics (val split)

| Dataset | Retriever | Slice | AUC | MRR | nDCG@5 | nDCG@10 |
|---------|-----------|-------|-----|-----|--------|---------|
| MIND    | BM25      | all   | —   | —   | —      | —       |
| MIND    | Semantic  | all   | —   | —   | —      | —       |
| MIND    | BM25      | cold  | —   | —   | —      | —       |
| MIND    | Semantic  | warm  | —   | —   | —      | —       |

### Key observations (to fill in)
- **BM25 vs Semantic**: …
- **Cold-start**: …
- **Head vs Tail**: …
- **MIND vs EB-NeRD**: …

### Codabench leaderboard scores

*(Include screenshot here)*

---

## 4. Where the Pipeline Breaks at 10×

| Component | Current (demo/small) | At 10× | Bottleneck |
|-----------|----------------------|--------|------------|
| BM25 index | ~200K articles, RAM | ~2M articles | Index fits in RAM (~4GB); OK |
| BM25 query | 1 query/impression | — | Single-threaded; parallelize with joblib |
| Embedding compute | ~1 min (GPU) | ~10 min | Batch encode with larger GPU; use pre-built |
| FAISS IndexFlatIP | O(N·d) per query | 10× slower | Switch to HNSW (IndexHNSWFlat) or ScaNN |
| Parquet feature store | ~MB | ~GB | Use Dask or partition by date |
| Temporal split | Pandas in-memory | OOM risk | Stream-sort or use DuckDB/Polars |
| Eval bootstrap | 1000 resamples | — | Already fast; parallelize with joblib |

**Critical change at 10×**: Replace `IndexFlatIP` with `IndexHNSWFlat` (FAISS) — reduces query time from O(N) to O(log N) with <1% recall loss at ef=64.
