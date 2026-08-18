"""
src/pipeline/parse.py — Q1: Parse raw datasets into unified schema
===================================================================

Unified schema
--------------
articles DataFrame columns:
    article_id   str
    title        str
    abstract     str
    body         str  (empty string when not available)
    category     str
    subcategory  str
    entities     list[str]
    dataset      str  ('mind' | 'ebnerd')

impressions DataFrame columns:
    impression_id  str
    user_id        str
    time           datetime64[ns, UTC]
    history        list[str]   article_ids the user clicked before this session
    candidates     list[str]   article_ids shown in this impression
    labels         list[int]   1=clicked, 0=not-clicked (parallel to candidates)
    dataset        str
"""

import ast
import json
import logging
import re
from pathlib import Path
import pathlib
from typing import Tuple

import pandas as pd

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# MIND-small
# ══════════════════════════════════════════════════════════════════════════════

_MIND_NEWS_COLS = [
    "article_id", "category", "subcategory", "title",
    "abstract", "url", "title_entities", "abstract_entities",
]
_MIND_BEH_COLS = [
    "impression_id", "user_id", "time", "history", "impressions",
]


def _parse_mind_news(news_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        news_path,
        sep="\t",
        header=None,
        names=_MIND_NEWS_COLS,
        dtype=str,
    ).fillna("")
    # Extract entity names from JSON strings
    def _extract_entities(s: str):
        try:
            return [e["SurfaceForms"][0] for e in json.loads(s) if e.get("SurfaceForms")]
        except Exception:
            return []

    df["entities"] = df["title_entities"].apply(_extract_entities)
    df["body"] = ""
    df["dataset"] = "mind"
    return df[["article_id", "title", "abstract", "body",
               "category", "subcategory", "entities", "dataset"]]


def _parse_mind_behaviors(beh_path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(
        beh_path,
        sep="\t",
        header=None,
        names=_MIND_BEH_COLS,
        dtype=str,
    ).fillna("")

    # Parse time — MIND format: "11/15/2019 10:12:27 AM"
    df["time"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p", utc=True)

    # History: space-separated article ids
    df["history"] = df["history"].apply(
        lambda x: x.strip().split() if x.strip() else []
    )

    # Impressions: "N1-1 N2-0 N3-1" → candidates + labels
    def _parse_impression(s: str):
        candidates, labels = [], []
        for token in s.strip().split():
            parts = token.rsplit("-", 1)
            if len(parts) == 2:
                candidates.append(parts[0])
                labels.append(int(parts[1]))
            else:
                candidates.append(parts[0])
                labels.append(0)
        return candidates, labels

    parsed = df["impressions"].apply(_parse_impression)
    df["candidates"] = parsed.apply(lambda x: x[0])
    df["labels"] = parsed.apply(lambda x: x[1])
    df["split"] = split_name
    df["dataset"] = "mind"

    return df[["impression_id", "user_id", "time", "history",
               "candidates", "labels", "split", "dataset"]]


def parse_mind(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (articles, impressions) DataFrames for MIND-small."""
    raw_dir = Path(raw_dir)
    articles_list, impressions_list = [], []

    for split in ("train", "dev"):
        split_dir = raw_dir / split
        news_path = split_dir / "news.tsv"
        beh_path = split_dir / "behaviors.tsv"

        if not news_path.exists():
            log.warning(f"  MIND {split} not found at {split_dir}, skipping")
            continue

        articles_list.append(_parse_mind_news(news_path))
        impressions_list.append(_parse_mind_behaviors(beh_path, split))

    # Deduplicate articles across splits
    articles = (
        pd.concat(articles_list, ignore_index=True)
        .drop_duplicates(subset=["article_id"])
        .reset_index(drop=True)
    )
    impressions = pd.concat(impressions_list, ignore_index=True).reset_index(drop=True)
    return articles, impressions


# ══════════════════════════════════════════════════════════════════════════════
# EB-NeRD
# ══════════════════════════════════════════════════════════════════════════════

def _find_ebnerd_dirs(raw_dir: Path):
    """Locate train/validation subdirs in the extracted EB-NeRD archive."""
    demo_dir = raw_dir / "demo"
    # The archive may extract to demo/ or demo/ebnerd_demo/
    candidates = [demo_dir, raw_dir / "ebnerd_demo"]
    for base in candidates:
        if (base / "train").exists():
            return base
    # Fallback: search recursively
    for p in raw_dir.rglob("articles.parquet"):
        return p.parent.parent
    return demo_dir


def _parse_ebnerd_articles(articles_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(articles_path)

    # EB-NeRD actual column names (inspect what's present):
    #   article_id, title, subtitle, abstract, body,
    #   category (int code), category_str (human-readable), subcategory,
    #   entities, published_time, ...
    # We keep only what we need and rename to unified schema.

    col_map = {}
    # Prefer human-readable category string
    if "category_str" in df.columns:
        col_map["category_str"] = "category"
        # Drop the integer category column to avoid duplicates
        if "category" in df.columns:
            df = df.drop(columns=["category"])
    # subtitle → abstract (if no 'abstract' column exists)
    if "subtitle" in df.columns and "abstract" not in df.columns:
        col_map["subtitle"] = "abstract"

    if col_map:
        df = df.rename(columns=col_map)

    # Ensure all required columns exist with sensible defaults
    for col, default in [
        ("abstract",    ""),
        ("body",        ""),
        ("category",    "unknown"),
        ("subcategory", ""),
        ("entities",    None),
    ]:
        if col not in df.columns:
            if default is None:
                df[col] = [[] for _ in range(len(df))]
            else:
                df[col] = default

    df["article_id"] = df["article_id"].astype(str)
    df["dataset"] = "ebnerd"
    # Entities may be stored as lists of dicts
    def _norm_entities(e):
        if isinstance(e, list):
            return [str(x.get("entity_id", x)) if isinstance(x, dict) else str(x) for x in e]
        return []
    df["entities"] = df["entities"].apply(_norm_entities)
    return df[["article_id", "title", "abstract", "body",
               "category", "subcategory", "entities", "dataset"]]


def _parse_ebnerd_behaviors(behaviors_path: pathlib.Path, history_path: pathlib.Path, split_name: str) -> pd.DataFrame:
    beh = pd.read_parquet(behaviors_path)

    # Build user history map from history.parquet (article_id_fixed array per user)
    user_history = {}
    if history_path.exists():
        hist_df = pd.read_parquet(history_path)
        for _, row in hist_df.iterrows():
            uid = str(int(row['user_id']))
            aids = [str(int(a)) for a in row['article_id_fixed'] if not pd.isna(a)]
            user_history[uid] = aids

    rows = []
    for _, r in beh.iterrows():
        uid = str(int(r['user_id']))
        imp_id = str(int(r['impression_id']))
        t = pd.Timestamp(r['impression_time']).tz_localize('UTC') if r['impression_time'].tzinfo is None else pd.Timestamp(r['impression_time'])

        inview = [str(int(a)) for a in r['article_ids_inview']] if hasattr(r['article_ids_inview'], '__iter__') else []
        clicked_set = set(str(int(a)) for a in r['article_ids_clicked']) if hasattr(r['article_ids_clicked'], '__iter__') else set()
        labels = [1 if c in clicked_set else 0 for c in inview]
        history = user_history.get(uid, [])

        rows.append({'impression_id': imp_id, 'user_id': uid, 'time': t,
                     'history': history, 'candidates': inview, 'labels': labels,
                     'split': split_name, 'dataset': 'ebnerd'})

    return pd.DataFrame(rows)


def parse_ebnerd(raw_dir) -> tuple:
    """Return (articles, impressions) DataFrames for EB-NeRD demo."""
    raw_dir = pathlib.Path(raw_dir)
    base = _find_ebnerd_dirs(raw_dir)
    log.info(f"  EB-NeRD base dir: {base}")

    articles_path = base / "articles.parquet"
    if not articles_path.exists():
        articles_path = base / "train" / "articles.parquet"

    articles = _parse_ebnerd_articles(articles_path)

    impressions_list = []
    for split, dirname in [("train", "train"), ("dev", "validation")]:
        beh_path = base / dirname / "behaviors.parquet"
        hist_path = base / dirname / "history.parquet"
        if not beh_path.exists():
            beh_path = base / split / "behaviors.parquet"
            hist_path = base / split / "history.parquet"
        if beh_path.exists():
            impressions_list.append(_parse_ebnerd_behaviors(beh_path, hist_path, split))
        else:
            log.warning(f"  EB-NeRD behaviors not found for split '{split}' at {beh_path}")

    impressions = pd.concat(impressions_list, ignore_index=True).reset_index(drop=True)
    return articles, impressions
