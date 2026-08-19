"""
src/submission/generate_mind.py — Q5: MIND Codabench submission
================================================================

MIND prediction format (competition 13967):
    One line per impression:
        <impression_id> [<rank1> <rank2> ... <rankN>]
    where rankN is the 1-indexed rank of each candidate article.

Usage:
    python -m src.submission.generate_mind [--retriever bm25|semantic] [--split test]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

log = logging.getLogger(__name__)


def _rank_candidates(candidates, ranked_list):
    """
    Given a list of candidates and a ranked list (best first),
    return 1-based ranks for each candidate.
    """
    rank_map = {aid: i + 1 for i, aid in enumerate(ranked_list)}
    # Candidates not in ranked list get rank = len(ranked_list)+1 (worst)
    default_rank = len(ranked_list) + 1
    return [rank_map.get(c, default_rank) for c in candidates]


def generate_mind_submission(
    dataset: str,
    retriever: str,
    split: str,
    out_dir: Path,
) -> Path:
    import pandas as pd

    # Use per-impression reranked col for submission (proper lexical/semantic order)
    # Fall back to global ranked if reranked not available
    import pandas as _pd2
    _tmp_path = Path(f"outputs/predictions/mind/{retriever}/{split}_ranked.parquet")
    _tmp_cols = _pd2.read_parquet(_tmp_path).columns.tolist() if _tmp_path.exists() else []
    reranked_col = f"{retriever}_reranked"
    ranked_col = reranked_col if reranked_col in _tmp_cols else f"{retriever}_ranked"
    del _tmp_path, _tmp_cols, _pd2
    ranked_path = Path(f"outputs/predictions/{dataset}/{retriever}/{split}_ranked.parquet")

    if not ranked_path.exists():
        raise FileNotFoundError(
            f"Ranked results not found: {ranked_path}\n"
            f"Run: python -m src.retrieval.{retriever}_retriever --dataset {dataset} --split {split}"
        )

    df = pd.read_parquet(ranked_path)

    # Deserialise if stored as pipe-separated strings
    import numpy as np
    for col in ["candidates", ranked_col]:
        if df[col].dtype == object:
            sample = df[col].iloc[0] if len(df) > 0 else ""
            if isinstance(sample, np.ndarray):
                df[col] = df[col].apply(lambda x: list(map(str, x)))
            elif isinstance(sample, str):
                df[col] = df[col].apply(lambda x: x.split("|") if isinstance(x, str) and x else [])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prediction.txt"

    lines = []
    for _, row in df.iterrows():
        imp_id = row["impression_id"]
        candidates = row["candidates"] if isinstance(row["candidates"], list) else []
        ranked = row[ranked_col] if isinstance(row[ranked_col], list) else []

        ranks = _rank_candidates(candidates, ranked)
        rank_str = " ".join(map(str, ranks))
        lines.append(f"{imp_id} [{rank_str}]")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info(f"MIND submission saved → {out_path}  ({len(lines):,} impressions)")
    return out_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Q5: Generate MIND Codabench submission")
    parser.add_argument("--retriever", choices=["bm25", "semantic"], default="bm25")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    out_dir = Path(f"outputs/submissions/mind/{args.retriever}")
    generate_mind_submission("mind", args.retriever, args.split, out_dir)
    print(f"Submission file ready: {out_dir}/prediction.txt")
    print("Upload to: https://www.codabench.org/competitions/13967/")


if __name__ == "__main__":
    main()
