"""
src/evaluation/bootstrap.py — Q4: Bootstrap 95% Confidence Intervals
======================================================================

Computes bootstrap CIs for any scalar metric by resampling impressions
with replacement.
"""

import logging
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def bootstrap_ci(
    impressions: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], Dict[str, float]],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap confidence intervals for all metrics returned by `metric_fn`.

    Parameters
    ----------
    impressions : DataFrame of impressions (full split)
    metric_fn   : function(impressions_sample) → dict of metric_name → value
    n_bootstrap : number of bootstrap resamples (default 1000)
    ci          : confidence level (default 0.95 → 95% CI)
    seed        : random seed for reproducibility

    Returns
    -------
    dict of metric_name → (lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(impressions)

    # Accumulate per-bootstrap metric values
    boot_results: Dict[str, List[float]] = {}

    for b in range(n_bootstrap):
        if (b + 1) % 100 == 0:
            log.debug(f"  Bootstrap {b + 1}/{n_bootstrap}")

        # Sample with replacement
        idx = rng.integers(0, n, size=n)
        sample = impressions.iloc[idx].reset_index(drop=True)

        try:
            metrics = metric_fn(sample)
        except Exception as e:
            log.warning(f"  Bootstrap sample {b} failed: {e}")
            continue

        for k, v in metrics.items():
            boot_results.setdefault(k, []).append(v)

    alpha = 1.0 - ci
    lo_pct = 100 * (alpha / 2)
    hi_pct = 100 * (1 - alpha / 2)

    ci_dict = {}
    for metric_name, values in boot_results.items():
        arr = np.array(values)
        ci_dict[metric_name] = (
            float(np.percentile(arr, lo_pct)),
            float(np.percentile(arr, hi_pct)),
        )

    return ci_dict


def format_ci_table(
    metrics: Dict[str, float],
    ci_bounds: Dict[str, Tuple[float, float]],
) -> str:
    """Format metrics + CIs as a readable table string."""
    lines = [
        f"{'Metric':<20} {'Value':>8}  {'95% CI':>20}",
        "-" * 52,
    ]
    for key, val in metrics.items():
        if key in ci_bounds:
            lo, hi = ci_bounds[key]
            ci_str = f"[{lo:.4f}, {hi:.4f}]"
        else:
            ci_str = "N/A"
        lines.append(f"{key:<20} {val:>8.4f}  {ci_str:>20}")
    return "\n".join(lines)
