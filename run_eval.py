"""
run_eval.py — Q4: Offline Evaluation Harness
=============================================
Run the full evaluation over pre-computed ranked results.

Usage:
    python run_eval.py --dataset mind   --retriever bm25
    python run_eval.py --dataset mind   --retriever semantic
    python run_eval.py --dataset ebnerd --retriever bm25
    python run_eval.py --dataset ebnerd --retriever semantic
    python run_eval.py --dataset mind   --retriever both   # compare side-by-side
"""

import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.bootstrap import bootstrap_ci, format_ci_table
from src.evaluation.metrics import (
    compute_beyond_accuracy_metrics,
    compute_ranking_metrics,
)
from src.evaluation.slicing import get_all_slices
from src.pipeline.feature_store import load_articles, load_impressions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_eval")

K_VALUES = (5, 10)
TOP_K_BEYOND = 10
N_BOOTSTRAP = 1000


def _load_ranked(dataset: str, retriever: str, split: str) -> pd.DataFrame:
    """Load pre-computed ranked results from outputs/."""
    ranked_path = Path(f"outputs/predictions/{dataset}/{retriever}/{split}_ranked.parquet")
    if not ranked_path.exists():
        raise FileNotFoundError(
            f"Ranked results not found at {ranked_path}. "
            f"Run: python -m src.retrieval.{retriever}_retriever --dataset {dataset} --split {split}"
        )
    df = pd.read_parquet(ranked_path)

    ranked_col = f"{retriever}_ranked"

    # Deserialise list columns if stored as strings (compatibility)
    for col in ["candidates", "labels", ranked_col]:
        if col in df.columns and df[col].dtype == object:
            try:
                sample = df[col].iloc[0]
                if isinstance(sample, str):
                    if col == "labels":
                        df[col] = df[col].apply(lambda x: list(map(int, x.split("|"))) if x else [])
                    else:
                        df[col] = df[col].apply(lambda x: x.split("|") if x else [])
            except Exception:
                pass

    return df, ranked_col


def evaluate_one(
    dataset: str,
    retriever: str,
    split: str,
    run_bootstrap: bool = True,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    log.info(f"\n{'═' * 60}")
    log.info(f"  dataset={dataset}  retriever={retriever}  split={split}")
    log.info(f"{'═' * 60}")

    processed_dir = Path(f"data/processed/{dataset}")
    articles = load_articles(processed_dir)

    # Load ranked impressions
    impressions, ranked_col = _load_ranked(dataset, retriever, split)

    # ── Slices ─────────────────────────────────────────────────────────────────
    slices = get_all_slices(impressions, articles, ranked_col)

    results = {}
    for slice_name, slice_df in slices.items():
        if len(slice_df) == 0:
            log.warning(f"  Slice '{slice_name}' is empty, skipping.")
            continue

        log.info(f"\n  ── Slice: {slice_name} ({len(slice_df):,} impressions) ──")

        # Accuracy metrics
        acc_metrics = compute_ranking_metrics(slice_df, ranked_col, k_values=K_VALUES)

        # Beyond-accuracy (only for 'all' slice to save time)
        if slice_name == "all":
            bey_metrics = compute_beyond_accuracy_metrics(
                slice_df, ranked_col, articles, top_k=TOP_K_BEYOND
            )
        else:
            bey_metrics = {}

        all_metrics = {**acc_metrics, **bey_metrics}

        # Bootstrap CI (only for 'all' slice)
        ci_bounds = {}
        if run_bootstrap and slice_name == "all":
            log.info(f"  Running {n_bootstrap} bootstrap resamples …")
            metric_fn = partial(compute_ranking_metrics, ranked_col=ranked_col, k_values=K_VALUES)
            ci_bounds = bootstrap_ci(slice_df, metric_fn, n_bootstrap=n_bootstrap)

        # Print results
        table = format_ci_table(all_metrics, ci_bounds)
        print(f"\n[{dataset.upper()}] [{retriever.upper()}] [{split}] — Slice: {slice_name}")
        print(table)

        results[slice_name] = {
            "metrics": all_metrics,
            "ci": {k: list(v) for k, v in ci_bounds.items()},
            "n": len(slice_df),
        }

    # ── Save ────────────────────────────────────────────────────────────────────
    out_dir = Path(f"outputs/results/{dataset}/{retriever}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\n  Results saved → {out_path}")
    return results


def compare_retrievers(dataset: str, split: str):
    """Print a side-by-side comparison table for BM25 vs semantic."""
    results = {}
    for retriever in ("bm25", "semantic"):
        try:
            res = evaluate_one(dataset, retriever, split, run_bootstrap=False)
            results[retriever] = res.get("all", {}).get("metrics", {})
        except FileNotFoundError as e:
            log.warning(str(e))
            results[retriever] = {}

    print(f"\n{'═' * 70}")
    print(f"  COMPARISON: {dataset.upper()} — {split}  |  BM25 vs Semantic")
    print(f"{'═' * 70}")
    all_metrics = set()
    for v in results.values():
        all_metrics.update(v.keys())
    print(f"{'Metric':<20} {'BM25':>12} {'Semantic':>12}  {'Δ (sem-bm25)':>14}")
    print("-" * 62)
    for m in sorted(all_metrics):
        bm25_val = results.get("bm25", {}).get(m, float("nan"))
        sem_val = results.get("semantic", {}).get(m, float("nan"))
        delta = sem_val - bm25_val
        print(f"{m:<20} {bm25_val:>12.4f} {sem_val:>12.4f}  {delta:>+14.4f}")


def main():
    parser = argparse.ArgumentParser(description="Q4: Offline Evaluation Harness")
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--retriever", choices=["bm25", "semantic", "both"], required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--no_bootstrap", action="store_true", help="Skip bootstrap CI (faster)")
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    if args.retriever == "both":
        compare_retrievers(args.dataset, args.split)
    else:
        evaluate_one(
            args.dataset,
            args.retriever,
            args.split,
            run_bootstrap=not args.no_bootstrap,
            n_bootstrap=args.n_bootstrap,
        )


if __name__ == "__main__":
    main()
