import pandas as pd
import numpy as np
import pickle
import time
import sys
from collections import Counter
import lightgbm as lgb

MODEL_PATH = "ebnerd_lgbm_model.txt"
EMB_PATH = "data/processed/ebnerd/ebnerd_embeddings.pkl"
ARTICLES_PATH = "data/raw/ebnerd/ebnerd_testset/articles.parquet"
BEHAVIORS_PATH = "data/raw/ebnerd/ebnerd_testset/test/behaviors.parquet"
HISTORY_PATH = "data/raw/ebnerd/ebnerd_testset/test/history.parquet"

def run_chunk(resume_after_id, out_path, end_after_id=None):
    t0 = time.time()
    print(f"[Worker] Loading data...")
    model = lgb.Booster(model_file=MODEL_PATH)
    
    articles_df = pd.read_parquet(ARTICLES_PATH)
    categories, subcategories, titles = {}, {}, {}
    for row in articles_df.itertuples():
        nid = int(row.article_id)
        categories[nid] = str(row.category)
        subcategories[nid] = str(row.subcategory)
        titles[nid] = set(str(row.title).lower().split())

    with open(EMB_PATH, "rb") as f:
        emb_dict = pickle.load(f)
    emb_dim = len(next(iter(emb_dict.values())))
    zero_vec = np.zeros(emb_dim, dtype=np.float32)
    for nid in emb_dict:
        v = emb_dict[nid]; norm = np.linalg.norm(v)
        if norm > 0: emb_dict[nid] = v / norm

    hist_df = pd.read_parquet(HISTORY_PATH)
    history_dict = dict(zip(hist_df["user_id"], hist_df["article_id_fixed"]))

    df_test = pd.read_parquet(BEHAVIORS_PATH)
    total = len(df_test)
    print(f"[Worker] Data loaded in {time.time()-t0:.1f}s")

    active_uid = -1
    user_vec = recent_vec = zero_vec
    cat_counts = Counter()
    subcat_counts = Counter()
    hist_words = set()
    hist_len = 0
    hist_vecs = np.empty((0, emb_dim))
    
    skipping = True
    
    # BATCHING LOGIC for massive speedup
    BATCH_SIZE = 15000 # Number of impressions to process at once
    batch_features = []
    batch_cands_sizes = []
    batch_imp_ids = []

    # Make sure output file is cleared if starting fresh chunk
    open(out_path, 'w', encoding='utf-8').close()

    def process_batch():
        nonlocal batch_features, batch_cands_sizes, batch_imp_ids
        if not batch_features: return
        
        flat_features = np.vstack(batch_features)
        # 10x-50x speedup by calling predict ONCE on a massive matrix!
        flat_scores = model.predict(flat_features)
        
        results = []
        offset = 0
        for imp_id, size in zip(batch_imp_ids, batch_cands_sizes):
            if size == 0:
                results.append(f"{imp_id} []\n")
            else:
                scores = flat_scores[offset:offset+size]
                order = np.argsort(-scores)
                ranks = np.empty_like(order)
                ranks[order] = np.arange(1, size+1)
                results.append(f"{imp_id} [{','.join(map(str,ranks))}]\n")
                offset += size
                
        with open(out_path, "a", encoding="utf-8") as f:
            f.write("".join(results))
            
        batch_features = []
        batch_cands_sizes = []
        batch_imp_ids = []

    for idx, row in enumerate(df_test.itertuples()):
        imp_id = row.impression_id
        
        if skipping:
            if imp_id == resume_after_id:
                skipping = False
            continue
        
        if end_after_id and imp_id > end_after_id:
            break

        uid = row.user_id
        cands = row.article_ids_inview

        if not len(cands):
            batch_imp_ids.append(imp_id)
            batch_cands_sizes.append(0)
            continue

        if uid != active_uid:
            active_uid = uid
            history = list(history_dict.get(uid, []))
            hist_len = len(history)
            if hist_len > 0:
                hist_vecs = np.array([emb_dict.get(nid, zero_vec) for nid in history])
                recent_vecs = np.array([emb_dict.get(nid, zero_vec) for nid in history[-5:]])
                user_vec = np.mean(hist_vecs, axis=0); n = np.linalg.norm(user_vec)
                if n > 0: user_vec /= n
                recent_vec = np.mean(recent_vecs, axis=0); n = np.linalg.norm(recent_vec)
                if n > 0: recent_vec /= n
                cat_counts = Counter(categories.get(nid, "") for nid in history)
                subcat_counts = Counter(subcategories.get(nid, "") for nid in history)
                hist_words = set()
                for nid in history: hist_words.update(titles.get(nid, set()))
            else:
                user_vec = recent_vec = zero_vec
                cat_counts = Counter(); subcat_counts = Counter()
                hist_words = set(); hist_vecs = np.empty((0, emb_dim))

        n_cands = len(cands)
        features = np.zeros((n_cands, 8), dtype=np.float32)
        cand_vecs = np.array([emb_dict.get(cid, zero_vec) for cid in cands])
        features[:, 0] = cand_vecs @ user_vec
        if hist_len > 0: features[:, 1] = np.max(cand_vecs @ hist_vecs.T, axis=1)
        features[:, 2] = cand_vecs @ recent_vec
        if hist_len > 0:
            for i, cid in enumerate(cands):
                c = categories.get(cid, "")
                if c: features[i,3] = cat_counts.get(c,0)/hist_len
                c = subcategories.get(cid, "")
                if c: features[i,4] = subcat_counts.get(c,0)/hist_len
                cw = titles.get(cid, set())
                if hist_words and cw:
                    inter = len(hist_words & cw); union = len(hist_words | cw)
                    features[i,5] = inter/union if union>0 else 0
        features[:, 6] = hist_len; features[:, 7] = np.arange(n_cands)

        batch_features.append(features)
        batch_cands_sizes.append(n_cands)
        batch_imp_ids.append(imp_id)
        
        if len(batch_imp_ids) >= BATCH_SIZE:
            process_batch()
            pct = 100*(idx+1)/total
            print(f"[Worker-{out_path}] {idx+1}/{total} ({pct:.1f}%) {(time.time()-t0)/60:.1f}m")
            
    # Process final batch
    process_batch()
    print(f"[Worker] Done! {(time.time()-t0)/60:.1f}m")

if __name__ == "__main__":
    part = int(sys.argv[1])
    
    # Fast forward over the already done 7.8M rows up to imp_id 339542695
    import pandas as pd
    df = pd.read_parquet(BEHAVIORS_PATH, columns=["impression_id"])
    ids = df["impression_id"].values
    
    base_idx = np.searchsorted(ids, 339542695)
    remaining = len(ids) - base_idx
    split1_idx = base_idx + remaining // 3
    split2_idx = base_idx + 2 * remaining // 3
    
    split1_id = int(ids[split1_idx])
    split2_id = int(ids[split2_idx])
    
    if part == 1:
        run_chunk(339542695, "outputs/submissions/ebnerd/lgbm/part1.txt", split1_id)
    elif part == 2:
        run_chunk(split1_id, "outputs/submissions/ebnerd/lgbm/part2.txt", split2_id)
    elif part == 3:
        run_chunk(split2_id, "outputs/submissions/ebnerd/lgbm/part3.txt", None)
