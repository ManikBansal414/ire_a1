
def _to_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return x.tolist()
    except Exception:
        pass
    return [] if x is None else []

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

# Cold-start threshold is computed dynamically per dataset:
# bottom 25th percentile of history length = cold-start users
# This adapts to EB-NeRD (avg 281 clicks) vs MIND (avg ~10 clicks)
COLD_START_THRESHOLD = 5   # fallback; overridden by split_cold_warm()
POPULARITY_HEAD_PCT = 0.20  # top 20% = "head"


def split_cold_warm(impressions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split impressions into cold-start and warm using dynamic threshold.
    Cold-start = bottom 25th percentile of history length in this split.
    This adapts automatically to EB-NeRD (mean 281 clicks) vs MIND (mean ~10).
    """
    import numpy as np
    history_len = impressions["history"].apply(lambda h: len(_to_list(h)))
    # Use 25th percentile as the cold/warm boundary
    threshold = max(1, int(np.percentile(history_len, 25)))
    log.info(f"  Cold-start threshold (25th pct): {threshold} clicks")
    cold = impressions[history_len <= threshold].copy()
    warm = impressions[history_len > threshold].copy()
    log.info(
        f"  Cold-start: {len(cold):,} impressions  |  Warm: {len(warm):,} impressions"
    )
    return cold, warm


def split_head_tail_articles(articles: pd.DataFrame, impressions: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Split article_ids into head (popular) and tail (long-tail).
    Popularity = total exposure count across candidate sets (in-view articles).
    This captures article exposure frequency, not just click frequency.

    Returns (head_ids, tail_ids) as pd.Index of string article_ids.
    """
    # Count candidate exposures (articles shown in impressions)
    all_aids = []
    for cands in impressions["candidates"]:
        for aid in _to_list(cands):
            all_aids.append(str(aid))

    if not all_aids:
        # Fall back to history
        for hist in impressions["history"]:
            for aid in _to_list(hist):
                all_aids.append(str(aid))

    if not all_aids:
        # No data: return all as tail
        all_ids = pd.Index(articles["article_id"].astype(str))
        return pd.Index([]), all_ids

    pop = pd.Series(all_aids).value_counts()

    n_head = max(1, int(len(pop) * POPULARITY_HEAD_PCT))
    head_ids = pd.Index(pop.nlargest(n_head).index.astype(str))
    tail_ids = pd.Index(pop.iloc[n_head:].index.astype(str))
    log.info(f"  Head articles: {len(head_ids):,}  |  Tail: {len(tail_ids):,}")

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
        cands = _to_list(row["candidates"])
        labels = _to_list(row["labels"])
        for c, l in zip(_to_list(cands), _to_list(labels)):
            if int(l) == 1 and str(c) in article_set:
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
