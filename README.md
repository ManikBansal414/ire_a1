# CS4.406 Assignment 1 — Lexical & Semantic Retrieval on EB-NeRD and MIND

**Course**: CS4.406 Information Retrieval & Extraction  
**Due**: August 27, 2026  

---

## One-Command Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download + parse + split + feature store (both datasets)
python build_pipeline.py

# 3. BM25 retrieval
python -m src.retrieval.bm25_retriever --dataset mind --split val
python -m src.retrieval.bm25_retriever --dataset ebnerd --split val

# 4. Semantic retrieval
python -m src.retrieval.semantic_retriever --dataset mind --split val
python -m src.retrieval.semantic_retriever --dataset ebnerd --split val

# 5. Full evaluation harness (all metrics, all slices, bootstrap CI)
python run_eval.py --dataset mind --retriever both
python run_eval.py --dataset ebnerd --retriever both

# 6. Generate Codabench submission files
python -m src.submission.generate_mind
python -m src.submission.generate_ebnerd
```

Or simply run all steps with:

```bash
make all
make submit
```

---

## Project Structure

```
ire_a1/
├── build_pipeline.py          # Q1: one-command pipeline entry point
├── run_eval.py                # Q4: offline evaluation harness
├── Makefile                   # convenience targets
├── requirements.txt
├── .gitignore                 # ignores data/, *.zip, *.pt, embeddings
│
├── src/
│   ├── pipeline/
│   │   ├── download.py        # download MIND-small + EB-NeRD demo
│   │   ├── parse.py           # parse → unified schema
│   │   ├── split.py           # temporal train/val/test split + leakage assertion
│   │   └── feature_store.py   # article + user feature store (parquet)
│   │
│   ├── retrieval/
│   │   ├── bm25_retriever.py  # BM25 inverted index, recall@K (Q2)
│   │   └── semantic_retriever.py  # FAISS + sentence-transformers (Q3)
│   │
│   ├── evaluation/
│   │   ├── metrics.py         # AUC, MRR, nDCG@5/10, diversity, novelty, coverage
│   │   ├── slicing.py         # cold-start/warm, head/tail article slices
│   │   └── bootstrap.py       # 95% bootstrap confidence intervals
│   │
│   └── submission/
│       ├── generate_mind.py   # MIND Codabench prediction file
│       └── generate_ebnerd.py # EB-NeRD Codabench prediction file
│
└── data/                      # ← gitignored; created by pipeline
    ├── raw/
    │   ├── mind/              # MINDsmall_train.zip, MINDsmall_dev.zip, extracted
    │   └── ebnerd/            # ebnerd_demo.zip, extracted
    └── processed/
        ├── mind/              # articles.parquet, *_impressions.parquet, *_users.parquet
        └── ebnerd/
```

---

## Datasets

| Dataset | Source | Size |
|---------|--------|------|
| MIND-small | [Microsoft Azure](https://mind201910small.blob.core.windows.net/release/) | ~160K articles, 15M impressions |
| EB-NeRD demo | [S3](https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip) | demo subset of 2.7M users |

---

## Pipeline Details

### Q1 — Data Pipeline

- **Download**: automatic from HuggingFace (MIND) and AWS S3 (EB-NeRD)
- **Unified schema**: `articles` (article_id, title, abstract, body, category, entities) + `impressions` (impression_id, user_id, time, history, candidates, labels)
- **Temporal split**: last 1 day = test, preceding 1 day = val, rest = train  
  ⚠️ Anti-gaming assertion: asserts `max(train.time) ≤ min(val.time) ≤ min(test.time)`

### Q2 — BM25 (Lexical)

- Tokenizer: lowercase whitespace split
- Index: `rank_bm25.BM25Okapi` over `title + abstract`
- Query: concatenated titles of 5 most recent clicked articles
- Reports **recall@{50, 100, 200}**

### Q3 — Semantic Retrieval

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized)
- Index: FAISS `IndexFlatIP` (exact cosine search)
- User rep: **recency-weighted mean-pool** of clicked article embeddings (decay=0.9)
- Reports **recall@{50, 100, 200}**

### Q4 — Evaluation Harness

| Metric | Description |
|--------|-------------|
| AUC | Per-impression ROC AUC, averaged |
| MRR | Mean Reciprocal Rank |
| nDCG@5 | Normalized DCG at rank 5 |
| nDCG@10 | Normalized DCG at rank 10 |
| Novelty | Mean −log₂(popularity) |
| ILD | Intra-list category diversity |
| Coverage | % of catalog recommended |

**Slices**: all, cold-start (<5 clicks), warm (≥5 clicks), head (top-20% popular), tail  
**CIs**: 95% bootstrap (1000 resamples)

### Q5 — Codabench Submission

- MIND: https://www.codabench.org/competitions/13967/
- EB-NeRD: https://www.codabench.org/competitions/2469/

---

## Git Policy

- No large files: `data/`, `*.zip`, `*.pt`, `*.ckpt`, `*.npy`, `*.faiss` are gitignored
- Commit frequently with meaningful messages
- No force-pushes after deadline
