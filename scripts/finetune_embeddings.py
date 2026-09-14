"""
finetune_embeddings.py

Fine-tunes the production embedding model (BAAI/bge-base-en-v1.5, see
app/rag/chroma_store.py) on the synthetic (query, chunk) pairs from
scripts/generate_embedding_training_data.py, using
MultipleNegativesRankingLoss -- the standard, simple, effective
contrastive fine-tuning recipe for retrieval embeddings: each
(query, positive_passage) pair in a training batch treats every OTHER
passage in that same batch as an implicit negative.

THIRD ROOT CAUSE, found after the first two (too little data, leaked
answers -- see generate_embedding_training_data.py's docstring) were
fixed and the fine-tune STILL underperformed baseline (EVALUATION.md
section 3): this script trained blindly for a fixed 4 epochs with no
validation signal at all (`save_strategy="no"`, no eval_dataset, no
evaluator). The training loss log from that run collapsed from 0.559
to 0.019 over just 84 steps on 328 pairs -- a classic small-dataset
memorization signature, and there was no mechanism to even notice it,
let alone stop before it happened. Fixed here with a held-out internal
validation split (disjoint from training, NOT the same as the
production hand-labeled eval set below) scored each epoch by
InformationRetrievalEvaluator, `load_best_model_at_end` to keep the
best-generalizing checkpoint instead of whatever epoch 4 collapsed to,
and an EarlyStoppingCallback so a dataset this size doesn't run all 4
epochs past the point it stopped generalizing.

IMPORTANT: the internal val-split IR metrics below are a DIFFERENT,
harder measurement than the real eval (1-of-N-training-pairs retrieval
vs. the production 1-of-20-real-candidates pool) -- they exist only to
pick a checkpoint during training. The only number that decides
whether this model ships is scripts/evaluate_embedding_finetune.py's
result against the held-out hand-labeled eval set (app/evaluation/
retrieval_labels.py) -- this script does not decide that on its own.

Output: scripts/finetuned_bge_embeddings/ (a saved SentenceTransformer
model directory) -- NOT written back into app/rag/chroma_store.py's
production model path.
"""

import json
import random
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from transformers import EarlyStoppingCallback

DATA_PATH = str(Path(__file__).resolve().parent / "embedding_training_pairs.jsonl")
OUTPUT_DIR = str(Path(__file__).resolve().parent / "finetuned_bge_embeddings")
BASE_MODEL = "BAAI/bge-base-en-v1.5"
VAL_FRACTION = 0.1  # held out from training, used only to pick the best epoch
SEED = 42


def main():
    pairs = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"Loaded {len(pairs)} training pairs")
    if len(pairs) < 20:
        raise ValueError(f"Only {len(pairs)} pairs -- too few to fine-tune meaningfully, check generation step")

    rng = random.Random(SEED)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n_val = max(10, int(len(shuffled) * VAL_FRACTION))
    val_pairs, train_pairs = shuffled[:n_val], shuffled[n_val:]
    print(f"Split: {len(train_pairs)} train / {len(val_pairs)} held-out (internal, for checkpoint selection only)")

    # MultipleNegativesRankingLoss expects paired columns (anchor,
    # positive) -- here (query, chunk_text) -- read positionally by
    # the loss from the dataset's column order, not by column name.
    train_dataset = Dataset.from_dict({
        "query": [p["query"] for p in train_pairs],
        "chunk_text": [p["chunk_text"] for p in train_pairs],
    })
    # A real HF eval_dataset is required for eval_strategy="epoch" to
    # actually build an eval dataloader at all -- the IR evaluator
    # below is what we actually care about, but the Trainer's own
    # evaluation_loop needs a non-None eval_dataset to run before it
    # hands off to the evaluator (see base Trainer.evaluation_loop).
    eval_dataset = Dataset.from_dict({
        "query": [p["query"] for p in val_pairs],
        "chunk_text": [p["chunk_text"] for p in val_pairs],
    })

    # Corpus = every pair's chunk (train + val), so the held-out
    # queries have real distractors to be ranked against, not just
    # their own 10%-slice answer chunk.
    corpus = {f"c{i}": p["chunk_text"] for i, p in enumerate(pairs)}
    chunk_to_cid = {p["chunk_text"]: f"c{i}" for i, p in enumerate(pairs)}
    queries = {f"q{i}": p["query"] for i, p in enumerate(val_pairs)}
    relevant_docs = {f"q{i}": {chunk_to_cid[p["chunk_text"]]} for i, p in enumerate(val_pairs)}

    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        ndcg_at_k=[5],
        precision_recall_at_k=[5],
        mrr_at_k=[5],
        accuracy_at_k=[5],
        map_at_k=[5],
        name="val",
        show_progress_bar=False,
    )
    best_model_metric = "eval_val_cosine_ndcg@5"

    model = SentenceTransformer(BASE_MODEL)
    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=16,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=best_model_metric,
        greater_is_better=True,
        logging_steps=10,
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=ir_evaluator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    best_score = trainer.state.best_metric
    print(f"Best internal val NDCG@5 (checkpoint-selection signal, NOT the production metric): {best_score}")

    model.save(OUTPUT_DIR)
    print(f"Saved fine-tuned model (best checkpoint by internal val NDCG@5) -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
