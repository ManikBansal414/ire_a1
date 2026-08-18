"""
src/pipeline/feature_store.py — Q1: Article & user feature store
=================================================================

Saves a compact, reusable feature store to disk so that retrieval and
evaluation modules can load features without re-running the full pipeline.

Article store  (data/processed/<dataset>/articles.parquet)
    article_id, title, abstract, body, category, subcategory, entities, text

User store     (data/processed/<dataset>/<split>_users.parquet)
    user_id, history (list of article_ids), recency_weights (list of floats)

Impression store  (data/processed/<dataset>/<split>_impressions.parquet)
    impression_id, user_id, time, history, candidates, labels
"""

import logging
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Article features ─────────────────────────────────────────────────────────

def _build_article_features(articles: pd.DataFrame) -> pd.DataFrame:
    """Add a 'text' column = title + ' ' + abstract for lexical retrieval."""
    df = articles.copy()
    df["text"] = (
        df["title"].fillna("").str.strip()
        + " "
        + df["abstract"].fillna("").str.strip()
    ).str.strip()
    return df


# ── User features ─────────────────────────────────────────────────────────────

def _recency_weights(history_len: int, decay: float = 0.9) -> list:
    """Exponentially decaying weights, most recent = 1.0 (last element)."""
    if history_len == 0:
        return []
    weights = [decay ** i for i in range(history_len - 1, -1, -1)]
    return weights


def _build_user_features(impressions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate user features from impressions.
    For each user take the click history from their most recent impression.
    """
    # Sort by time, take last impression per user → gives most up-to-date history
    latest = (
        impressions.sort_values("time")
        .groupby("user_id")
        .last()
        .reset_index()[["user_id", "history"]]
    )
    latest["recency_weights"] = latest["history"].apply(
        lambda h: _recency_weights(len(h))
    )
    return latest


# ── Main ──────────────────────────────────────────────────────────────────────

def build_feature_store(
    articles: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    out_dir: Path,
    dataset: str,
) -> None:
    """
    Save article features and per-split impression + user features.

    Parameters
    ----------
    articles : unified articles DataFrame
    splits   : dict with keys 'train','val','test' → impression DataFrames
    out_dir  : directory to write parquet files
    dataset  : 'mind' or 'ebnerd' (stored in metadata)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Article feature store ──────────────────────────────────────────────
    art_df = _build_article_features(articles)
    art_path = out_dir / "articles.parquet"
    # entities must be stored as string to avoid parquet schema issues
    art_df_save = art_df.copy()
    art_df_save["entities"] = art_df_save["entities"].apply(
        lambda e: "|".join(e) if isinstance(e, list) else ""
    )
    art_df_save.to_parquet(art_path, index=False)
    log.info(f"  Saved articles → {art_path}  ({len(art_df):,} rows)")

    # ── 2. Per-split impression + user stores ─────────────────────────────────
    for split_name, imp_df in splits.items():
        # Impressions
        imp_path = out_dir / f"{split_name}_impressions.parquet"
        _save_impressions(imp_df, imp_path)
        log.info(f"  Saved {split_name} impressions → {imp_path}  ({len(imp_df):,} rows)")

        # User features
        user_df = _build_user_features(imp_df)
        user_path = out_dir / f"{split_name}_users.parquet"
        _save_users(user_df, user_path)
        log.info(f"  Saved {split_name} users → {user_path}  ({len(user_df):,} rows)")

    # ── 3. Save metadata ──────────────────────────────────────────────────────
    meta = {
        "dataset": dataset,
        "n_articles": len(articles),
        "splits": {k: len(v) for k, v in splits.items()},
    }
    import json
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Saved metadata → {meta_path}")


def _save_impressions(df: pd.DataFrame, path: Path) -> None:
    """Serialize impressions; list columns stored as pipe-separated strings."""
    save = df.copy()
    save["history"] = save["history"].apply(
        lambda x: "|".join(x) if isinstance(x, list) else ""
    )
    save["candidates"] = save["candidates"].apply(
        lambda x: "|".join(x) if isinstance(x, list) else ""
    )
    save["labels"] = save["labels"].apply(
        lambda x: "|".join(map(str, x)) if isinstance(x, list) else ""
    )
    save.to_parquet(path, index=False)


def _save_users(df: pd.DataFrame, path: Path) -> None:
    """Serialize user features; list columns stored as pipe-separated strings."""
    save = df.copy()
    save["history"] = save["history"].apply(
        lambda x: "|".join(x) if isinstance(x, list) else ""
    )
    save["recency_weights"] = save["recency_weights"].apply(
        lambda x: "|".join(f"{w:.4f}" for w in x) if isinstance(x, list) else ""
    )
    save.to_parquet(path, index=False)


# ── Load helpers (used by retrieval / eval modules) ───────────────────────────

def load_articles(processed_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(Path(processed_dir) / "articles.parquet")
    df["entities"] = df["entities"].apply(
        lambda e: e.split("|") if isinstance(e, str) and e else []
    )
    return df


def load_impressions(processed_dir: Path, split: str) -> pd.DataFrame:
    df = pd.read_parquet(Path(processed_dir) / f"{split}_impressions.parquet")
    df["history"] = df["history"].apply(
        lambda x: x.split("|") if isinstance(x, str) and x else []
    )
    df["candidates"] = df["candidates"].apply(
        lambda x: x.split("|") if isinstance(x, str) and x else []
    )
    df["labels"] = df["labels"].apply(
        lambda x: list(map(int, x.split("|"))) if isinstance(x, str) and x else []
    )
    return df


def load_users(processed_dir: Path, split: str) -> pd.DataFrame:
    df = pd.read_parquet(Path(processed_dir) / f"{split}_users.parquet")
    df["history"] = df["history"].apply(
        lambda x: x.split("|") if isinstance(x, str) and x else []
    )
    df["recency_weights"] = df["recency_weights"].apply(
        lambda x: list(map(float, x.split("|"))) if isinstance(x, str) and x else []
    )
    return df
