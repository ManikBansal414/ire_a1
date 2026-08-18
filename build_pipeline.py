"""
build_pipeline.py — Q1: One-command pipeline rebuild
=====================================================
Run:  python build_pipeline.py [--dataset mind|ebnerd|both] [--split_days_test 1] [--split_days_val 1]

This script orchestrates:
  1. Download raw data (MIND-small, EB-NeRD demo)
  2. Parse into unified schema
  3. Temporal train/val/test split (NO random split)
  4. Build article + user feature store
"""

import argparse
import logging
import sys
from pathlib import Path

# ── ensure src/ is on path ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.download import download_mind, download_ebnerd
from src.pipeline.parse import parse_mind, parse_ebnerd
from src.pipeline.split import temporal_split
from src.pipeline.feature_store import build_feature_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_pipeline")


def run_mind(args):
    log.info("═══════════════ MIND pipeline ═══════════════")
    raw_dir = Path("data/raw/mind")
    processed_dir = Path("data/processed/mind")

    # 1. Download
    download_mind(raw_dir)

    # Check at least one split is present before proceeding
    train_news = raw_dir / "train" / "news.tsv"
    dev_news   = raw_dir / "dev"   / "news.tsv"
    if not train_news.exists() and not dev_news.exists():
        log.error(
            "MIND raw files not found. Please follow the manual download "
            "instructions printed above, then re-run this script."
        )
        return

    # 2. Parse
    articles, impressions = parse_mind(raw_dir)
    log.info(f"  articles={len(articles):,}  impressions={len(impressions):,}")

    # 3. Temporal split
    splits = temporal_split(
        impressions,
        test_days=args.split_days_test,
        val_days=args.split_days_val,
    )
    log.info(
        f"  train={len(splits['train']):,}  val={len(splits['val']):,}  test={len(splits['test']):,}"
    )

    # 4. Feature store
    build_feature_store(articles, splits, processed_dir, dataset="mind")
    log.info(f"  Feature store saved -> {processed_dir}")


def run_ebnerd(args):
    log.info("═══════════════ EB-NeRD pipeline ═══════════════")
    raw_dir = Path("data/raw/ebnerd")
    processed_dir = Path("data/processed/ebnerd")

    # 1. Download
    download_ebnerd(raw_dir)

    # Check raw files exist
    demo_articles = raw_dir / "demo" / "articles.parquet"
    if not demo_articles.exists():
        log.error(
            "EB-NeRD raw files not found. Please follow the manual download "
            "instructions printed above, then re-run this script."
        )
        return

    # 2. Parse
    articles, impressions = parse_ebnerd(raw_dir)
    log.info(f"  articles={len(articles):,}  impressions={len(impressions):,}")

    # 3. Temporal split
    splits = temporal_split(
        impressions,
        test_days=args.split_days_test,
        val_days=args.split_days_val,
    )
    log.info(
        f"  train={len(splits['train']):,}  val={len(splits['val']):,}  test={len(splits['test']):,}"
    )

    # 4. Feature store
    build_feature_store(articles, splits, processed_dir, dataset="ebnerd")
    log.info(f"  Feature store saved -> {processed_dir}")


def main():
    parser = argparse.ArgumentParser(description="One-command pipeline rebuild (Q1)")
    parser.add_argument(
        "--dataset",
        choices=["mind", "ebnerd", "both"],
        default="both",
        help="Which dataset to process (default: both)",
    )
    parser.add_argument(
        "--split_days_test",
        type=int,
        default=1,
        help="How many trailing days to use as test split",
    )
    parser.add_argument(
        "--split_days_val",
        type=int,
        default=1,
        help="How many days preceding test to use as val split",
    )
    args = parser.parse_args()

    if args.dataset in ("mind", "both"):
        run_mind(args)
    if args.dataset in ("ebnerd", "both"):
        run_ebnerd(args)

    log.info("✓ Pipeline complete. Run `python run_eval.py` next.")


if __name__ == "__main__":
    main()
