"""
finetune_embeddings.py

Fine-tunes the production embedding model (BAAI/bge-base-en-v1.5, see
app/rag/chroma_store.py) on the synthetic (query, chunk) pairs from
scripts/generate_embedding_training_data.py, using
MultipleNegativesRankingLoss -- the standard, simple, effective
contrastive fine-tuning recipe for retrieval embeddings: each
(query, positive_passage) pair in a training batch treats every OTHER
passage in that same batch as an implicit negative, so no explicit
hard-negative mining is needed for a training set this size.

Output: scripts/finetuned_bge_embeddings/ (a saved SentenceTransformer
model directory) -- NOT written back into app/rag/chroma_store.py's
production model path. Whether to actually swap it into production
depends entirely on scripts/evaluate_embedding_finetune.py's result
against the held-out hand-labeled eval set (app/evaluation/
retrieval_labels.py) -- this script does not decide that on its own.
"""

import json
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss

DATA_PATH = str(Path(__file__).resolve().parent / "embedding_training_pairs.jsonl")
OUTPUT_DIR = str(Path(__file__).resolve().parent / "finetuned_bge_embeddings")
BASE_MODEL = "BAAI/bge-base-en-v1.5"


def main():
    pairs = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"Loaded {len(pairs)} training pairs")
    if len(pairs) < 20:
        raise ValueError(f"Only {len(pairs)} pairs -- too few to fine-tune meaningfully, check generation step")

    # MultipleNegativesRankingLoss expects paired columns (anchor,
    # positive) -- here (query, chunk_text) -- read positionally by
    # the loss from the dataset's column order, not by column name.
    dataset = Dataset.from_dict({
        "query": [p["query"] for p in pairs],
        "chunk_text": [p["chunk_text"] for p in pairs],
    })

    model = SentenceTransformer(BASE_MODEL)
    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=16,
        warmup_ratio=0.1,
        save_strategy="no",
        logging_steps=10,
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
    )
    trainer.train()

    model.save(OUTPUT_DIR)
    print(f"Saved fine-tuned model -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
