"""
src/retrieval/semantic_retriever.py — Q3: Semantic Candidate Generation
=========================================================================

Pipeline:
  1. Load pre-built embeddings (EB-NeRD provides contrastive-trained ones)
     OR compute embeddings with sentence-transformers (MIND)
  2. Build a FAISS flat (brute-force) ANN index — exact for small scale
  3. User representation = mean-pool of clicked article embeddings
     (recency-weighted: most recent click counts more)
  4. Retrieve top-K candidates by cosine similarity
  5. Report recall@K for K ∈ {50, 100, 200}

Usage (as script):
    python -m src.retrieval.semantic_retriever --dataset mind --split val
    python -m src.retrieval.semantic_retriever --dataset ebnerd --split val
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.pipeline.feature_store import load_articles, load_impressions

log = logging.getLogger(__name__)

# Model used for computing embeddings when not pre-supplied
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── Embedding computation ──────────────────────────────────────────────────────

def compute_embeddings(texts: List[str], model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> np.ndarray:
    """
    Compute sentence embeddings using sentence-transformers.
    Returns float32 array of shape (N, dim).
    """
    from sentence_transformers import SentenceTransformer
    log.info(f"  Loading sentence-transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    log.info(f"  Encoding {len(texts):,} texts (batch_size={batch_size}) …")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-norm → cosine ≡ dot product
    )
    return embeddings.astype(np.float32)


def load_or_compute_embeddings(
    articles: "pd.DataFrame",
    embed_path: Path,
    model_name: str = DEFAULT_MODEL,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load embeddings from disk or compute them.
    Returns (embeddings, article_ids).
    embed_path: .npy file path.
    """
    embed_path = Path(embed_path)
    ids_path = embed_path.with_suffix(".ids.pkl")

    if embed_path.exists() and ids_path.exists() and not force_recompute:
        log.info(f"  Loading embeddings from {embed_path} …")
        embeddings = np.load(embed_path)
        with open(ids_path, "rb") as f:
            article_ids = pickle.load(f)
        log.info(f"  Loaded {embeddings.shape[0]:,} embeddings (dim={embeddings.shape[1]})")
        return embeddings, article_ids

    # Compute
    embed_path.parent.mkdir(parents=True, exist_ok=True)
    texts = (
        articles["title"].fillna("") + " " + articles["abstract"].fillna("")
    ).str.strip().tolist()
    article_ids = articles["article_id"].tolist()

    embeddings = compute_embeddings(texts, model_name)

    # Save
    np.save(embed_path, embeddings)
    with open(ids_path, "wb") as f:
        pickle.dump(article_ids, f)
    log.info(f"  Embeddings saved → {embed_path}")
    return embeddings, article_ids


# ── FAISS index ────────────────────────────────────────────────────────────────

class FAISSIndex:
    """Brute-force FAISS index (IndexFlatIP for L2-normalized vectors = cosine)."""

    def __init__(self):
        self.index = None
        self.article_ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}

    def build(self, embeddings: np.ndarray, article_ids: List[str]) -> None:
        import faiss
        dim = embeddings.shape[1]
        log.info(f"  Building FAISS IndexFlatIP (dim={dim}, n={len(article_ids):,}) …")
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.article_ids = article_ids
        self._id_to_idx = {aid: i for i, aid in enumerate(article_ids)}
        log.info("  FAISS index built.")

    def retrieve(self, query_vec: np.ndarray, top_k: int) -> List[str]:
        """
        Retrieve top-k article_ids for a single query vector.
        query_vec: 1D float32 array (already L2-normalized).
        """
        if self.index is None:
            raise RuntimeError("Index not built — call build() first.")
        q = query_vec.reshape(1, -1).astype(np.float32)
        _, indices = self.index.search(q, top_k)
        return [self.article_ids[i] for i in indices[0] if 0 <= i < len(self.article_ids)]

    def get_embedding(self, article_id: str) -> Optional[np.ndarray]:
        """Return the stored embedding for a given article_id."""
        import faiss
        idx = self._id_to_idx.get(article_id)
        if idx is None:
            return None
        vec = np.zeros((1, self.index.d), dtype=np.float32)
        self.index.reconstruct(idx, vec[0])
        return vec[0]

    def save(self, path: Path) -> None:
        import faiss
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save index + metadata separately
        faiss.write_index(self.index, str(path))
        meta_path = path.with_suffix(".meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({"article_ids": self.article_ids, "_id_to_idx": self._id_to_idx}, f)
        log.info(f"  FAISS index saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "FAISSIndex":
        import faiss
        path = Path(path)
        obj = cls()
        obj.index = faiss.read_index(str(path))
        meta_path = path.with_suffix(".meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        obj.article_ids = meta["article_ids"]
        obj._id_to_idx = meta["_id_to_idx"]
        return obj


# ── User representation ────────────────────────────────────────────────────────

def _user_vector(
    history: List[str],
    faiss_index: FAISSIndex,
    recency_decay: float = 0.9,
) -> Optional[np.ndarray]:
    """
    Mean-pool (recency-weighted) embeddings of clicked articles.
    Returns L2-normalized vector, or None if no history.
    """
    if not history:
        return None

    vecs, weights = [], []
    for i, aid in enumerate(history):
        emb = faiss_index.get_embedding(aid)
        if emb is not None:
            # Most recent = index len-1 → weight = 1.0; older → decay^gap
            age = len(history) - 1 - i
            vecs.append(emb)
            weights.append(recency_decay ** age)

    if not vecs:
        return None

    weights_arr = np.array(weights, dtype=np.float32)
    weights_arr /= weights_arr.sum()
    user_vec = np.average(np.stack(vecs, axis=0), axis=0, weights=weights_arr)

    # L2-normalize (so dot product = cosine similarity)
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec /= norm
    return user_vec.astype(np.float32)


# ── Recall@K evaluation ────────────────────────────────────────────────────────

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def evaluate_recall(
    impressions: "pd.DataFrame",
    faiss_index: FAISSIndex,
    k_values: Tuple[int, ...] = (50, 100, 200),
    recency_decay: float = 0.9,
    max_history: int = 50,
) -> Dict[str, float]:
    recall_sums = {k: 0.0 for k in k_values}
    n = 0

    for _, row in tqdm(impressions.iterrows(), total=len(impressions), desc="  Evaluating semantic"):
        history = (row["history"] if isinstance(row["history"], list) else [])[-max_history:]
        candidates = row["candidates"] if isinstance(row["candidates"], list) else []
        labels = row["labels"] if isinstance(row["labels"], list) else []

        relevant = [c for c, l in zip(candidates, labels) if l == 1]
        if not relevant:
            continue

        user_vec = _user_vector(history, faiss_index, recency_decay)
        if user_vec is None:
            continue

        retrieved = faiss_index.retrieve(user_vec, top_k=max(k_values))

        for k in k_values:
            recall_sums[k] += recall_at_k(retrieved, relevant, k)
        n += 1

    if n == 0:
        return {f"recall@{k}": 0.0 for k in k_values}
    return {f"recall@{k}": recall_sums[k] / n for k in k_values}


# ── Ranked results for evaluation harness ─────────────────────────────────────

def run_semantic_retrieval(
    impressions: "pd.DataFrame",
    faiss_index: FAISSIndex,
    top_k: int = 100,
    recency_decay: float = 0.9,
    max_history: int = 50,
) -> "pd.DataFrame":
    """Add 'semantic_ranked' column to impressions."""
    results = []
    for _, row in tqdm(impressions.iterrows(), total=len(impressions), desc="  Semantic retrieval"):
        history = (row["history"] if isinstance(row["history"], list) else [])[-max_history:]
        user_vec = _user_vector(history, faiss_index, recency_decay)
        retrieved = faiss_index.retrieve(user_vec, top_k=top_k) if user_vec is not None else []
        results.append(retrieved)

    impressions = impressions.copy()
    impressions["semantic_ranked"] = results
    return impressions


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Q3: Semantic Retrieval")
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rebuild_index", action="store_true")
    args = parser.parse_args()

    processed_dir = Path(f"data/processed/{args.dataset}")
    embed_dir = Path(f"outputs/embeddings/{args.dataset}")
    index_path = Path(f"outputs/indexes/{args.dataset}_faiss.index")
    results_dir = Path(f"outputs/predictions/{args.dataset}/semantic")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load articles
    log.info(f"Loading articles …")
    articles = load_articles(processed_dir)

    # Load or compute embeddings
    embed_path = embed_dir / "article_embeddings.npy"
    embeddings, article_ids = load_or_compute_embeddings(
        articles, embed_path, model_name=args.model, force_recompute=args.rebuild_index
    )

    # Build or load FAISS index
    if index_path.exists() and not args.rebuild_index:
        log.info(f"Loading FAISS index from {index_path} …")
        faiss_idx = FAISSIndex.load(index_path)
    else:
        faiss_idx = FAISSIndex()
        faiss_idx.build(embeddings, article_ids)
        faiss_idx.save(index_path)

    # Load impressions
    log.info(f"Loading {args.split} impressions …")
    impressions = load_impressions(processed_dir, args.split)

    # Evaluate recall@K
    metrics = evaluate_recall(impressions, faiss_idx, k_values=(50, 100, 200))

    print(f"\n{'='*50}")
    print(f"Semantic Recall@K  |  dataset={args.dataset}  split={args.split}")
    print(f"{'='*50}")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")
    print()

    # Save ranked results
    result_df = run_semantic_retrieval(impressions, faiss_idx, top_k=args.top_k)
    out_path = results_dir / f"{args.split}_ranked.parquet"
    result_df[["impression_id", "user_id", "candidates", "labels", "semantic_ranked"]].to_parquet(
        out_path, index=False
    )
    log.info(f"Ranked results saved → {out_path}")

    import json
    metrics_path = results_dir / f"{args.split}_recall.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Recall metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
