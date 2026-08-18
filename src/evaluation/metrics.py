"""
src/evaluation/metrics.py — Q4: Official evaluation metrics
============================================================

Implements:
  - AUC (per impression, averaged)
  - MRR (Mean Reciprocal Rank)
  - nDCG@5, nDCG@10
  - Beyond-accuracy:
      · Intra-list diversity (ILD) — avg pairwise cosine distance
      · Novelty               — mean −log2(popularity) of recommended items
      · Coverage              — fraction of article catalog covered
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)


# ── Accuracy metrics ──────────────────────────────────────────────────────────

def _dcg(gains: List[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def _ndcg_at_k(ranked_labels: List[int], k: int) -> float:
    """nDCG@k given a list of labels in ranked order (1=relevant, 0=not)."""
    if not ranked_labels:
        return 0.0
    dcg = _dcg(ranked_labels, k)
    ideal_labels = sorted(ranked_labels, reverse=True)
    idcg = _dcg(ideal_labels, k)
    return dcg / idcg if idcg > 0 else 0.0


def _mrr(ranked_labels: List[int]) -> float:
    """MRR = 1/rank of first relevant item."""
    for i, label in enumerate(ranked_labels):
        if label == 1:
            return 1.0 / (i + 1)
    return 0.0


def _auc(scores: List[float], labels: List[int]) -> float:
    """Per-impression AUC. Returns 0.5 if only one class present."""
    if len(set(labels)) < 2:
        return 0.5  # undefined, use random baseline
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.5


def compute_ranking_metrics(
    impressions: "pd.DataFrame",
    ranked_col: str,
    k_values: Tuple[int, ...] = (5, 10),
) -> Dict[str, float]:
    """
    Compute per-impression AUC, MRR, nDCG@5, nDCG@10.

    Parameters
    ----------
    impressions : DataFrame with 'candidates', 'labels', and `ranked_col`
        ranked_col : list of article_ids in ranked order (highest score first)
    """
    auc_scores, mrr_scores = [], []
    ndcg_scores = {k: [] for k in k_values}

    for _, row in impressions.iterrows():
        candidates = row["candidates"] if isinstance(row["candidates"], list) else []
        labels = row["labels"] if isinstance(row["labels"], list) else []
        ranked = row[ranked_col] if isinstance(row[ranked_col], list) else []

        if not candidates or not labels or not ranked:
            continue

        label_map = dict(zip(candidates, labels))

        # For AUC: use rank-based scores (position penalty)
        # rank position → score = 1/(rank+1) for ranked items, 0 for unranked
        rank_pos = {aid: i for i, aid in enumerate(ranked)}
        scores_for_auc = [
            1.0 / (rank_pos[c] + 1) if c in rank_pos else 0.0
            for c in candidates
        ]
        auc_scores.append(_auc(scores_for_auc, labels))

        # For nDCG/MRR: labels in ranked order (only candidates in candidates list)
        ranked_labels = [label_map.get(aid, 0) for aid in ranked if aid in label_map]
        if not ranked_labels:
            continue

        mrr_scores.append(_mrr(ranked_labels))
        for k in k_values:
            ndcg_scores[k].append(_ndcg_at_k(ranked_labels, k))

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    result = {
        "AUC": _mean(auc_scores),
        "MRR": _mean(mrr_scores),
    }
    for k in k_values:
        result[f"nDCG@{k}"] = _mean(ndcg_scores[k])

    return result


# ── Beyond-accuracy metrics ───────────────────────────────────────────────────

def compute_novelty(
    ranked_lists: List[List[str]],
    article_popularity: Dict[str, int],
    top_k: int = 10,
) -> float:
    """
    Novelty = mean −log2(pop(i)/N) over recommended items.
    pop(i) = number of times article i appears in all impressions' histories.
    """
    total_clicks = sum(article_popularity.values()) or 1
    scores = []
    for ranked in ranked_lists:
        for aid in ranked[:top_k]:
            pop = article_popularity.get(aid, 1) / total_clicks
            scores.append(-math.log2(pop + 1e-10))
    return float(np.mean(scores)) if scores else 0.0


def compute_intra_list_diversity(
    ranked_lists: List[List[str]],
    article_category_map: Dict[str, str],
    top_k: int = 10,
) -> float:
    """
    ILD (category-based) = fraction of top-K pairs with different categories.
    Simple proxy when article embeddings are not loaded in eval.
    """
    scores = []
    for ranked in ranked_lists:
        top = ranked[:top_k]
        if len(top) < 2:
            continue
        cats = [article_category_map.get(aid, "unknown") for aid in top]
        n_pairs = 0
        n_diff = 0
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                n_pairs += 1
                if cats[i] != cats[j]:
                    n_diff += 1
        scores.append(n_diff / n_pairs if n_pairs > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def compute_coverage(
    ranked_lists: List[List[str]],
    total_articles: int,
    top_k: int = 10,
) -> float:
    """Coverage = |unique recommended articles| / total_articles."""
    recommended = set()
    for ranked in ranked_lists:
        recommended.update(ranked[:top_k])
    return len(recommended) / total_articles if total_articles > 0 else 0.0


def compute_beyond_accuracy_metrics(
    impressions: "pd.DataFrame",
    ranked_col: str,
    articles: "pd.DataFrame",
    top_k: int = 10,
) -> Dict[str, float]:
    """Compute novelty, ILD, and coverage."""
    import pandas as pd

    # Article popularity = count of appearances in histories
    all_history = [
        aid
        for hist in impressions["history"]
        for aid in (hist if isinstance(hist, list) else [])
    ]
    pop_counter: Dict[str, int] = {}
    for aid in all_history:
        pop_counter[aid] = pop_counter.get(aid, 0) + 1

    cat_map = dict(zip(articles["article_id"], articles["category"].fillna("unknown")))
    n_articles = len(articles)

    ranked_lists = [
        row[ranked_col] if isinstance(row[ranked_col], list) else []
        for _, row in impressions.iterrows()
    ]

    return {
        "Novelty":   compute_novelty(ranked_lists, pop_counter, top_k),
        "ILD":       compute_intra_list_diversity(ranked_lists, cat_map, top_k),
        "Coverage":  compute_coverage(ranked_lists, n_articles, top_k),
    }
