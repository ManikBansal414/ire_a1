.PHONY: all data bm25 semantic eval submit clean

PYTHON = python

## ── One-command rebuild ───────────────────────────────────────────────────────
all: data bm25 semantic eval

## Download + parse + split + feature-store (Q1)
data:
	$(PYTHON) build_pipeline.py

## BM25 retrieval + recall@K (Q2)
bm25:
	$(PYTHON) -m src.retrieval.bm25_retriever --dataset mind   --split val
	$(PYTHON) -m src.retrieval.bm25_retriever --dataset ebnerd --split val

## Semantic retrieval + recall@K (Q3)
semantic:
	$(PYTHON) -m src.retrieval.semantic_retriever --dataset mind   --split val
	$(PYTHON) -m src.retrieval.semantic_retriever --dataset ebnerd --split val

## Full evaluation harness (Q4)
eval:
	$(PYTHON) run_eval.py --dataset mind   --retriever bm25
	$(PYTHON) run_eval.py --dataset mind   --retriever semantic
	$(PYTHON) run_eval.py --dataset ebnerd --retriever bm25
	$(PYTHON) run_eval.py --dataset ebnerd --retriever semantic

## Generate Codabench submission files (Q5)
submit:
	$(PYTHON) -m src.submission.generate_mind
	$(PYTHON) -m src.submission.generate_ebnerd

## Clean outputs (keep data/ and model files)
clean:
	rm -rf outputs/predictions/ outputs/results/
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
