"""Quick unit tests for core modules."""
import sys
sys.path.insert(0, ".")

import pandas as pd
from datetime import datetime, timezone, timedelta

# ── Test 1: Temporal split + anti-gaming assertion ────────────────────────────
from src.pipeline.split import temporal_split

base = datetime(2024, 1, 1, tzinfo=timezone.utc)
times = [base + timedelta(hours=i) for i in range(100)]
imps = pd.DataFrame({
    'impression_id': range(100),
    'user_id': ['u1'] * 100,
    'time': times,
    'history': [[] for _ in range(100)],
    'candidates': [[] for _ in range(100)],
    'labels': [[] for _ in range(100)],
})
splits = temporal_split(imps, test_days=1, val_days=1)
assert len(splits['train']) > 0
assert len(splits['val']) > 0
assert len(splits['test']) > 0
assert splits['train']['time'].max() <= splits['val']['time'].min(), 'LEAKAGE in train/val!'
assert splits['val']['time'].max() <= splits['test']['time'].min(), 'LEAKAGE in val/test!'
print(f"[PASS] Temporal split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

# ── Test 2: nDCG ──────────────────────────────────────────────────────────────
from src.evaluation.metrics import _ndcg_at_k, _mrr, _auc

assert abs(_ndcg_at_k([1, 0, 0, 0, 0], 5) - 1.0) < 1e-6, "nDCG perfect case failed"
assert _ndcg_at_k([0, 1, 0, 0, 0], 5) < 1.0, "nDCG rank-2 should be < 1"
assert _ndcg_at_k([], 5) == 0.0, "nDCG empty should be 0"
print("[PASS] nDCG@5 unit tests")

# ── Test 3: MRR ───────────────────────────────────────────────────────────────
assert _mrr([1, 0, 0]) == 1.0
assert abs(_mrr([0, 1, 0]) - 0.5) < 1e-6
assert _mrr([0, 0, 0]) == 0.0
print("[PASS] MRR unit tests")

# ── Test 4: AUC ───────────────────────────────────────────────────────────────
auc_perfect = _auc([1.0, 0.5, 0.0], [1, 0, 0])
assert auc_perfect == 1.0, f"Perfect AUC should be 1.0, got {auc_perfect}"
auc_single = _auc([1.0, 0.5], [1, 1])  # single class → 0.5
assert auc_single == 0.5
print("[PASS] AUC unit tests")

# ── Test 5: Slicing ───────────────────────────────────────────────────────────
from src.evaluation.slicing import split_cold_warm

cold_imps = imps.copy()
cold_imps['history'] = [[] for _ in range(100)]
warm_imps = imps.copy()
warm_imps['history'] = [['a', 'b', 'c', 'd', 'e', 'f'] for _ in range(100)]
mixed = pd.concat([cold_imps, warm_imps], ignore_index=True)
cold, warm = split_cold_warm(mixed)
assert len(cold) == 100 and len(warm) == 100
print("[PASS] Cold/warm slicing unit tests")

# ── Test 6: BM25 index (requires rank_bm25) ───────────────────────────────────
from src.retrieval.bm25_retriever import BM25Index, _build_query, recall_at_k

articles_df = pd.DataFrame({
    'article_id': ['A1', 'A2', 'A3'],
    'text': ['football sports match', 'cooking recipe pasta', 'football world cup final'],
})
idx = BM25Index()
idx.build(articles_df)
result = idx.retrieve(['football'], top_k=2)
assert 'A1' in result or 'A3' in result, f"Expected football articles, got {result}"
print(f"[PASS] BM25 index: query='football' -> {result}")

r = recall_at_k(['A1', 'A3', 'A2'], ['A3'], k=2)
assert r == 1.0, f"recall@2 should be 1.0, got {r}"
r2 = recall_at_k(['A2', 'A3', 'A1'], ['A1'], k=2)
assert r2 == 0.0, f"recall@2 should be 0.0, got {r2}"
print("[PASS] recall@K unit tests")

# ── Test 7: Feature store save/load roundtrip ─────────────────────────────────
import tempfile, pathlib
from src.pipeline.feature_store import build_feature_store, load_articles, load_impressions

with tempfile.TemporaryDirectory() as tmpdir:
    art = pd.DataFrame({
        'article_id': ['A1', 'A2'],
        'title': ['Title One', 'Title Two'],
        'abstract': ['Abstract one', 'Abstract two'],
        'body': ['', ''],
        'category': ['sports', 'food'],
        'subcategory': ['', ''],
        'entities': [['ent1'], ['ent2']],
        'dataset': ['mind', 'mind'],
    })
    splits_small = {
        'train': imps[:60],
        'val': imps[60:80],
        'test': imps[80:],
    }
    build_feature_store(art, splits_small, pathlib.Path(tmpdir), 'test')
    loaded_art = load_articles(pathlib.Path(tmpdir))
    assert len(loaded_art) == 2
    loaded_imp = load_impressions(pathlib.Path(tmpdir), 'val')
    assert len(loaded_imp) == 20
    print("[PASS] Feature store save/load roundtrip")

print()
print("=" * 50)
print("ALL UNIT TESTS PASSED")
print("=" * 50)
