"""
src/evaluation/slicing.py — Q4: User and article slicing
==========================================================

Slices:
  - Cold-start users : users with < COLD_START_THRESHOLD clicks in history
  - Warm users       : users with >= COLD_START_THRESHOLD clicks
  - Head articles    : top POPULARITY_HEAD_PCT% by number of appearances in histories
  - Tail articles    : remaining (bottom 1 - POPULARITY_HEAD_PCT)

These slices let us compare BM25 vs semantic retrieval on different sub-populations.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COLD_START_THRESHOLD = 5   # clicks in history
POPULARITY_HEAD_PCT = 0.20  # top 20% = "head"


def split_cold_warm(impressions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split impressions into cold-start (< threshold clicks) and warm.
    Uses the length of the 'history' list for each impression.
    """
    history_len = impressions["history"].apply(
        lambda h: len(h) if isinstance(h, list) else 0
    )
    cold = impressions[history_len < COLD_START_THRESHOLD].copy()
    warm = impressions[history_len >= COLD_START_THRESHOLD].copy()
    log.info(
        f"  Cold-start: {len(cold):,} impressions  |  Warm: {len(warm):,} impressions"
    )
    return cold, warm


def split_head_tail_articles(articles: pd.DataFrame, impressions: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Split article_ids into head (popular) and tail (long-tail).
    Popularity = number of times article_id appears in all impression histories.

    Returns (head_ids, tail_ids) as pd.Series of article_ids.
    """
    # Count appearances
    all_aids = [
        aid
        for hist in impressions["history"]
        for aid in (hist if isinstance(hist, list) else [])
    ]
    if not all_aids:
        # Fall back to candidate appearances
        all_aids = [
            aid
            for cands in impressions["candidates"]
            for aid in (cands if isinstance(cands, list) else [])
        ]

    pop = pd.Series(all_aids).value_counts()
    pop = pop.reindex(articles["article_id"], fill_value=0)

    n_head = max(1, int(len(pop) * POPULARITY_HEAD_PCT))
    head_ids = pop.nlargest(n_head).index
    tail_ids = pop.index.difference(head_ids)

    log.info(f"  Head articles: {len(head_ids):,}  |  Tail: {len(tail_ids):,}")
    return pd.Series(head_ids), pd.Series(tail_ids)


def filter_impressions_by_article_set(
    impressions: pd.DataFrame,
    article_set,
    ranked_col: str,
) -> pd.DataFrame:
    """
    Keep only impressions where the relevant candidates (label=1) are all in article_set.
    Used to slice evaluation by head/tail articles.
    """
    article_set = set(article_set)

    def _has_head_positive(row):
        cands = row["candidates"] if isinstance(row["candidates"], list) else []
        labels = row["labels"] if isinstance(row["labels"], list) else []
        for c, l in zip(cands, labels):
            if l == 1 and c in article_set:
                return True
        return False

    mask = impressions.apply(_has_head_positive, axis=1)
    return impressions[mask].copy()


def get_all_slices(
    impressions: pd.DataFrame,
    articles: pd.DataFrame,
    ranked_col: str,
) -> Dict[str, pd.DataFrame]:
    """
    Return dict of named impression slices for evaluation.
    Keys: 'all', 'cold', 'warm', 'head', 'tail'
    """
    cold, warm = split_cold_warm(impressions)
    head_ids, tail_ids = split_head_tail_articles(articles, impressions)

    head_imp = filter_impressions_by_article_set(impressions, head_ids, ranked_col)
    tail_imp = filter_impressions_by_article_set(impressions, tail_ids, ranked_col)

    return {
        "all":  impressions,
        "cold": cold,
        "warm": warm,
        "head": head_imp,
        "tail": tail_imp,
    }
