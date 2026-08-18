"""
src/pipeline/split.py — Q1: Temporal train/val/test split
==========================================================

Critical constraint (Q9 anti-gaming):
  - NEVER use random split for interaction data
  - Use time-based splitting ONLY
  - Assert no future-click leakage: train impressions must all be
    strictly before val impressions, which must be before test

Split strategy:
  - Sort all impressions by time ascending
  - test  = last `test_days` days
  - val   = `val_days` days immediately preceding test
  - train = everything before val
"""

import logging
from datetime import timedelta
from typing import Dict

import pandas as pd

log = logging.getLogger(__name__)


def temporal_split(
    impressions: pd.DataFrame,
    test_days: int = 1,
    val_days: int = 1,
) -> Dict[str, pd.DataFrame]:
    """
    Split impressions temporally into train / val / test.

    Parameters
    ----------
    impressions : DataFrame with column 'time' (timezone-aware datetime)
    test_days   : number of trailing days to use as test
    val_days    : number of days preceding test to use as validation

    Returns
    -------
    dict with keys 'train', 'val', 'test'
    """
    if "time" not in impressions.columns:
        raise ValueError("impressions must have a 'time' column")

    t_min = impressions["time"].min()
    t_max = impressions["time"].max()
    log.info(f"  Impression time range: {t_min} → {t_max}")

    # Boundary timestamps
    test_start = t_max - timedelta(days=test_days)
    val_start = test_start - timedelta(days=val_days)

    mask_test = impressions["time"] > test_start
    mask_val = (impressions["time"] > val_start) & (impressions["time"] <= test_start)
    mask_train = impressions["time"] <= val_start

    splits = {
        "train": impressions[mask_train].copy().reset_index(drop=True),
        "val":   impressions[mask_val].copy().reset_index(drop=True),
        "test":  impressions[mask_test].copy().reset_index(drop=True),
    }

    # ── Anti-gaming assertion (Q9) ─────────────────────────────────────────────
    _assert_no_leakage(splits)

    for name, df in splits.items():
        log.info(f"  {name}: {len(df):,} impressions")

    return splits


def _assert_no_leakage(splits: Dict[str, pd.DataFrame]) -> None:
    """
    Assert strict temporal ordering between splits.
    Raises AssertionError if any train impression is after any val/test impression,
    or any val impression is after any test impression.
    This catches future-click leakage (Q9).
    """
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]

    if len(train) > 0 and len(val) > 0:
        train_max = train["time"].max()
        val_min = val["time"].min()
        assert train_max <= val_min, (
            f"LEAKAGE: train has impressions at {train_max} which is AFTER "
            f"val starts at {val_min}"
        )

    if len(val) > 0 and len(test) > 0:
        val_max = val["time"].max()
        test_min = test["time"].min()
        assert val_max <= test_min, (
            f"LEAKAGE: val has impressions at {val_max} which is AFTER "
            f"test starts at {test_min}"
        )

    log.info("  ✓ No future-click leakage detected (anti-gaming assertion passed)")
