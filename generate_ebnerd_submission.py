"""
Generate EB-NeRD Codabench submission using the pretrained LightGBM ranker.
Resumes from a checkpoint if a partial predictions.txt exists.

Usage:
    python generate_ebnerd_submission.py
"""
import pandas as pd
import numpy as np
import pickle
import time
import zipfile
from pathlib import Path
from collections import Counter
import lightgbm as lgb

MODEL_PATH = "ebnerd_lgbm_model.txt"
EMB_PATH = "data/processed/ebnerd/ebnerd_embeddings.pkl"
ARTICLES_PATH = "data/raw/ebnerd/ebnerd_testset/articles.parquet"
BEHAVIORS_PATH = "data/raw/ebnerd/ebnerd_testset/test/behaviors.parquet"
HISTORY_PATH = "data/raw/ebnerd/ebnerd_testset/test/history.parquet"
OUT_DIR = Path("outputs/submissions/ebnerd/lgbm")
BATCH_SIZE = 20000


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "predictions.txt"
    zip_path = OUT_DIR / "submission.zip"

    print("1. Loading model...")
    model = lgb.Booster(model_file=MODEL_PATH)

    print("2. Loading articles...")
    articles_df = pd.read_parquet(ARTICLES_PATH)
    categories, subcategories, titles = {}, {}, {}
    for row in articles_df.itertuples():
        nid = int(row.article_id)
        categories[nid] = str(row.category)
        subcategories[nid] = str(row.subcategory)
        titles[nid] = set(str(row.title).lower().split())

    print("3. Loading embeddings...")
    with open(EMB_PATH, "rb") as f:
        emb_dict = pickle.load(f)
    emb_dim = len(next(iter(emb_dict.values())))
    zero_vec = np.zeros(emb_dim, dtype=np.float32)
    for nid in emb_dict:
        v = emb_dict[nid]
        norm = np.linalg.norm(v)
        if norm > 0:
            emb_dict[nid] = v / norm

    print("4. Loading history...")
    hist_df = pd.read_parquet(HISTORY_PATH)
    history_dict = dict(zip(hist_df["user_id"], hist_df["article_id_fixed"]))

    print("5. Loading test behaviors...")
    df_test = pd.read_parquet(BEHAVIORS_PATH)
    total = len(df_test)

    # Resume logic
    resume_after = None
    if out_path.exists():
        with open(out_path, "r") as f:
            lines = f.readlines()
        if lines:
            resume_after = int(lines[-1].split()[0])
            print(f"   Resuming from impression_id {resume_after} ({len(lines)} already written)")

    active_uid = -1
    user_vec = recent_vec = zero_vec
    cat_counts = Counter()
    subcat_counts = Counter()
    hist_words = set()
    hist_len = 0
    hist_vecs = np.empty((0, emb_dim))

    results = []
    skipping = resume_after is not None

    mode = "a" if resume_after else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for idx, row in enumerate(df_test.itertuples()):
            imp_id = row.impression_id

            if skipping:
                if imp_id == resume_after:
                    skipping = False
                continue

            uid = row.user_id
            cands = row.article_ids_inview

            if not len(cands):
                results.append(f"{imp_id} []\n")
                continue

            if uid != active_uid:
                active_uid = uid
                history = list(history_dict.get(uid, []))
                hist_len = len(history)

                if hist_len > 0:
                    hist_vecs = np.array([emb_dict.get(nid, zero_vec) for nid in history])
                    recent_vecs = np.array([emb_dict.get(nid, zero_vec) for nid in history[-5:]])

                    user_vec = np.mean(hist_vecs, axis=0)
                    n = np.linalg.norm(user_vec)
                    if n > 0: user_vec = user_vec / n

                    recent_vec = np.mean(recent_vecs, axis=0)
                    n = np.linalg.norm(recent_vec)
                    if n > 0: recent_vec = recent_vec / n

                    cat_counts = Counter(categories.get(nid, "") for nid in history)
                    subcat_counts = Counter(subcategories.get(nid, "") for nid in history)
                    hist_words = set()
                    for nid in history:
                        hist_words.update(titles.get(nid, set()))
                else:
                    user_vec = recent_vec = zero_vec
                    cat_counts = Counter()
                    subcat_counts = Counter()
                    hist_words = set()
                    hist_vecs = np.empty((0, emb_dim))

            n_cands = len(cands)
            features = np.zeros((n_cands, 8), dtype=np.float32)
            cand_vecs = np.array([emb_dict.get(cid, zero_vec) for cid in cands])

            features[:, 0] = cand_vecs @ user_vec
            if hist_len > 0:
                features[:, 1] = np.max(cand_vecs @ hist_vecs.T, axis=1)
            features[:, 2] = cand_vecs @ recent_vec

            if hist_len > 0:
                for i, cid in enumerate(cands):
                    c_cat = categories.get(cid, "")
                    if c_cat: features[i, 3] = cat_counts.get(c_cat, 0) / hist_len
                    c_sub = subcategories.get(cid, "")
                    if c_sub: features[i, 4] = subcat_counts.get(c_sub, 0) / hist_len
                    c_words = titles.get(cid, set())
                    if hist_words and c_words:
                        inter = len(hist_words & c_words)
                        union = len(hist_words | c_words)
                        features[i, 5] = inter / union if union > 0 else 0

            features[:, 6] = hist_len
            features[:, 7] = np.arange(n_cands)

            scores = model.predict(features)
            order = np.argsort(-scores)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, n_cands + 1)
            results.append(f"{imp_id} [{','.join(map(str, ranks))}]\n")

            if len(results) >= BATCH_SIZE:
                f.write("".join(results))
                results = []
                pct = 100 * (idx + 1) / total
                print(f"   Processed {idx+1}/{total} ({pct:.1f}%) in {(time.time()-t0)/60:.1f}m")

        if results:
            f.write("".join(results))

    print("\n6. Zipping submission...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname="predictions.txt")

    print(f"\nDONE! Total time: {(time.time()-t0)/60:.1f} minutes")
    print(f"Submission: {zip_path}")


if __name__ == "__main__":
    main()
