"""
diagnose_rlvr_generation.py

One-off diagnostic, not part of the training pipeline: loads the base
model exactly as train_grpo.py does (4-bit, same prompt) and prints a
few RAW completions in full, so we can actually see why every rollout
in the real run hit the token cap without ever producing a parseable
"Verdict: <Buy/Hold/Sell>" line -- guessing at a token-budget fix
without looking at real output risks repeating the same mistake again.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATASET = Path(__file__).resolve().parent / "rlvr_dataset_train.jsonl"
N_EXAMPLES = 3
MAX_NEW_TOKENS = 600  # generous -- this run is diagnostic, not memory-constrained by a training loop


def main():
    rows = [json.loads(line) for line in open(DATASET, encoding="utf-8") if line.strip()][:N_EXAMPLES]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
    )
    model.eval()

    for i, row in enumerate(rows, 1):
        print(f"\n{'=' * 80}\nEXAMPLE {i}: {row['ticker']} ({row['as_of_date']})\n{'=' * 80}")
        print("--- PROMPT ---")
        print(row["prompt"])

        messages = [{"role": "user", "content": row["prompt"]}]
        templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        # This transformers version returns a BatchEncoding (dict-like,
        # needs .input_ids) rather than a raw tensor -- handle both so
        # this isn't version-pinned to one specific behavior.
        input_ids = (templated.input_ids if hasattr(templated, "input_ids") else templated).to(model.device)
        print(f"\n--- PROMPT TOKEN COUNT: {input_ids.shape[1]} ---")

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion_ids = output_ids[0, input_ids.shape[1]:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        hit_eos = completion_ids[-1].item() == tokenizer.eos_token_id

        print(f"\n--- COMPLETION ({len(completion_ids)} tokens, hit_eos={hit_eos}) ---")
        print(completion)


if __name__ == "__main__":
    main()
