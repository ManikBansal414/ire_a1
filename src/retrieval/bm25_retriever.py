"""
src/retrieval/bm25_retriever.py — Q2: BM25 Lexical Candidate Generation
=========================================================================

Pipeline:
  1. Build an inverted index over article text (title + abstract)
  2. Given a user's click history → concatenate recently-clicked article titles
     as a query string
  3. Retrieve top-K candidates using BM25 scoring (rank_bm25)
  4. Report recall@K for K ∈ {50, 100, 200}

Usage (as script):
    python -m src.retrieval.bm25_retriever --dataset mind --split val
    python -m src.retrieval.bm25_retriever --dataset ebnerd --split val
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.pipeline.feature_store import load_articles, load_impressions

log = logging.getLogger(__name__)


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


# ── Index building ─────────────────────────────────────────────────────────────

class BM25Index:
    """Wraps rank_bm25.BM25Okapi with article id ↔ index mapping."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.article_ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}

    def build(self, articles: "pd.DataFrame") -> None:
        """
        Build BM25 index from a DataFrame with columns [article_id, text].
        text = title + abstract (pre-built by feature_store).
        """
        log.info(f"  Building BM25 index over {len(articles):,} articles …")
        self.article_ids = articles["article_id"].tolist()
        self._id_to_idx = {aid: i for i, aid in enumerate(self.article_ids)}
        corpus = [_tokenize(t) for t in tqdm(articles["text"].fillna(""), desc="  Tokenising")]
        self.bm25 = BM25Okapi(corpus)
        log.info("  BM25 index built.")

    def retrieve(self, query_tokens: List[str], top_k: int) -> List[str]:
        """Return top-k article_ids by BM25 score."""
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [self.article_ids[i] for i in top_indices]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"  BM25 index saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Query construction ─────────────────────────────────────────────────────────

def _build_query(history: List[str], article_title_map: Dict[str, str], max_articles: int = 5) -> List[str]:
    """
    Construct a BM25 query from a user's click history.
    Uses up to `max_articles` most recent clicked article titles.
    """
    recent = history[-max_articles:]  # most recent last
    titles = [article_title_map.get(aid, "") for aid in recent]
    query_str = " ".join(t for t in titles if t)
    return _tokenize(query_str)


# ── Recall@K evaluation ────────────────────────────────────────────────────────

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Recall@K = |relevant ∩ top-k retrieved| / |relevant|"""
    if not relevant:
        return 0.0
    top_k_set = set(retrieved[:k])
    return len(top_k_set & set(relevant)) / len(relevant)


def evaluate_recall(
    impressions: "pd.DataFrame",
    index: BM25Index,
    article_title_map: Dict[str, str],
    k_values: Tuple[int, ...] = (50, 100, 200),
    max_query_articles: int = 5,
) -> Dict[str, float]:
    """
    Evaluate recall@K for each K over all impressions.

    'relevant' for each impression = candidates that were clicked (label==1).
    """
    recall_sums = {k: 0.0 for k in k_values}
    n = 0

    for _, row in tqdm(impressions.iterrows(), total=len(impressions), desc="  Evaluating BM25"):
        history = row["history"] if isinstance(row["history"], list) else []
        candidates = row["candidates"] if isinstance(row["candidates"], list) else []
        labels = row["labels"] if isinstance(row["labels"], list) else []

        # relevant = candidates that were actually clicked
        relevant = [c for c, l in zip(candidates, labels) if l == 1]
        if not relevant:
            continue  # skip impressions with no positive label

        query_tokens = _build_query(history, article_title_map, max_articles=max_query_articles)
        if not query_tokens:
            continue

        max_k = max(k_values)
        retrieved = index.retrieve(query_tokens, top_k=max_k)

        for k in k_values:
            recall_sums[k] += recall_at_k(retrieved, relevant, k)
        n += 1

    if n == 0:
        return {f"recall@{k}": 0.0 for k in k_values}

    return {f"recall@{k}": recall_sums[k] / n for k in k_values}


# ── Ranked results for evaluation harness ─────────────────────────────────────

def run_bm25_retrieval(
    impressions: "pd.DataFrame",
    index: BM25Index,
    article_title_map: Dict[str, str],
    top_k: int = 100,
    max_query_articles: int = 5,
) -> "pd.DataFrame":
    """
    Add 'bm25_ranked' column to impressions: list of top-K article_ids.
    Used by the evaluation harness (Q4) and submission generator (Q5).
    """
    import pandas as pd

    results = []
    for _, row in tqdm(impressions.iterrows(), total=len(impressions), desc="  BM25 retrieval"):
        history = row["history"] if isinstance(row["history"], list) else []
        query_tokens = _build_query(history, article_title_map, max_articles=max_query_articles)
        retrieved = index.retrieve(query_tokens, top_k=top_k) if query_tokens else []
        results.append(retrieved)

    impressions = impressions.copy()
    impressions["bm25_ranked"] = results
    return impressions


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Q2: BM25 Retrieval")
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--max_query_articles", type=int, default=5)
    parser.add_argument("--rebuild_index", action="store_true")
    args = parser.parse_args()

    processed_dir = Path(f"data/processed/{args.dataset}")
    index_path = Path(f"outputs/indexes/{args.dataset}_bm25.pkl")
    results_dir = Path(f"outputs/predictions/{args.dataset}/bm25")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load articles
    log.info(f"Loading articles from {processed_dir} …")
    articles = load_articles(processed_dir)
    article_title_map = dict(zip(articles["article_id"], articles["title"].fillna("")))

    # Build or load BM25 index
    if index_path.exists() and not args.rebuild_index:
        log.info(f"Loading BM25 index from {index_path} …")
        bm25_index = BM25Index.load(index_path)
    else:
        bm25_index = BM25Index()
        bm25_index.build(articles)
        bm25_index.save(index_path)

    # Load impressions
    log.info(f"Loading {args.split} impressions …")
    impressions = load_impressions(processed_dir, args.split)

    # Evaluate recall@K
    log.info(f"Evaluating recall@K on {args.dataset} {args.split} …")
    metrics = evaluate_recall(
        impressions,
        bm25_index,
        article_title_map,
        k_values=(50, 100, 200),
        max_query_articles=args.max_query_articles,
    )

    print(f"\n{'='*50}")
    print(f"BM25 Recall@K  |  dataset={args.dataset}  split={args.split}")
    print(f"{'='*50}")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")
    print()

    # Save per-impression ranked lists
    result_df = run_bm25_retrieval(
        impressions, bm25_index, article_title_map,
        top_k=args.top_k,
        max_query_articles=args.max_query_articles,
    )
    out_path = results_dir / f"{args.split}_ranked.parquet"
    result_df[["impression_id", "user_id", "candidates", "labels", "bm25_ranked"]].to_parquet(
        out_path, index=False
    )
    log.info(f"Ranked results saved → {out_path}")

    # Save metrics
    import json
    metrics_path = results_dir / f"{args.split}_recall.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Recall metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
