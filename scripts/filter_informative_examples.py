"""
filter_informative_examples.py

Pre-training data curation, inspired by DAPO's "dynamic sampling"
(arXiv:2503.14476): GRPO only produces a gradient when a group's
rollouts disagree on reward (see the project's own finding -- only
2/27 steps in a real run had nonzero reward variance, wasting the
other ~93% of GPU time on groups that all landed on the same
correct/incorrect outcome). TRL's GRPOTrainer doesn't implement
DAPO's live resampling (open feature request, huggingface/trl#4764),
so this does the offline equivalent: probe every training example
with the BASE model before training starts, keep only the ones where
the model doesn't already unanimously agree with itself, and train on
that curated set instead of the full one.

This is a real, GPU-bound pass (N_PROBES generations per example), not
a quick script -- run it once, on a schedule that doesn't overlap with
an active training run on the same GPU.

Usage:
    python scripts/filter_informative_examples.py --dataset scripts/rlvr_dataset_train_v2.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.training.rlvr_reward import compute_reward

DEFAULT_DATASET = str(Path(__file__).resolve().parent / "rlvr_dataset_train_v2.jsonl")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 4, not 2 (train_grpo.py's num_generations) -- a 2-sample probe is too
# noisy to trust as "this example is unanimous": it could show zero
# variance by chance even for a genuinely informative example. 4
# independent samples gives a real read on the model's actual
# consistency here, at roughly 2x the probing cost of using 2.
N_PROBES = 4


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n-probes", type=int, default=N_PROBES)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument(
        "--output-filtered", default=None,
        help="Path for the curated (informative-only) JSONL. Defaults to "
             "<dataset>_informative.jsonl next to --dataset.",
    )
    p.add_argument(
        "--output-report", default=None,
        help="Path for the full per-example probe report (JSON, includes examples that got "
             "filtered OUT too, for inspection). Defaults to <dataset>_probe_report.json.",
    )
    return p.parse_args()


def _load_model(model_name, load_in_4bit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            ),
            device_map={"": 0},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        model = model.to("cuda") if torch.cuda.is_available() else model

    model.eval()
    return tokenizer, model


def _generate_one(tokenizer, model, prompt, max_new_tokens):
    import torch

    messages = [{"role": "user", "content": prompt}]
    templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    # Same BatchEncoding-vs-tensor handling as evaluate_rlvr_model.py /
    # diagnose_rlvr_generation.py -- this transformers version returns
    # a BatchEncoding, not a raw tensor.
    input_ids = (templated.input_ids if hasattr(templated, "input_ids") else templated).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,  # matches train_grpo.py's own sampling temperature -- the probe should see what training would see
            pad_token_id=tokenizer.eos_token_id,
        )
    completion_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def probe_example(tokenizer, model, row, n_probes, max_new_tokens):
    """Generates n_probes independent completions for one training
    example and scores each with the real GRPO reward function --
    returns the per-probe rewards and their population stdev (0.0 means
    every probe landed on the same reward, exactly the GRPO-starving
    case this whole script exists to filter out)."""
    rewards = []
    for _ in range(n_probes):
        completion = _generate_one(tokenizer, model, row["prompt"], max_new_tokens)
        rewards.append(compute_reward(completion, row["realized_return_pct"]))
    return {
        "ticker": row["ticker"],
        "as_of_date": row["as_of_date"],
        "rewards": rewards,
        "reward_std": pstdev(rewards),
        "informative": pstdev(rewards) > 0,
    }


def main():
    args = _parse_args()
    dataset_path = Path(args.dataset)
    output_filtered = Path(args.output_filtered) if args.output_filtered else \
        dataset_path.with_name(dataset_path.stem + "_informative.jsonl")
    output_report = Path(args.output_report) if args.output_report else \
        dataset_path.with_name(dataset_path.stem + "_probe_report.json")

    rows = [json.loads(line) for line in open(dataset_path, encoding="utf-8") if line.strip()]
    print(f"Probing {len(rows)} examples x {args.n_probes} generations each "
          f"({len(rows) * args.n_probes} total generations)...", file=sys.stderr)

    tokenizer, model = _load_model(args.model, args.load_in_4bit)

    results = []
    kept_rows = []
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['ticker']} ({row['as_of_date']})...", file=sys.stderr)
        probe = probe_example(tokenizer, model, row, args.n_probes, args.max_new_tokens)
        results.append(probe)
        print(f"  rewards={probe['rewards']}  std={probe['reward_std']:.3f}  "
              f"informative={probe['informative']}", file=sys.stderr)
        if probe["informative"]:
            kept_rows.append(row)

    with open(output_filtered, "w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row) + "\n")

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    n_informative = sum(r["informative"] for r in results)
    print(f"\n{n_informative}/{len(rows)} examples ({100 * n_informative / len(rows):.1f}%) are informative "
          f"(model disagrees with itself across {args.n_probes} probes)", file=sys.stderr)
    print(f"Curated training set -> {output_filtered}", file=sys.stderr)
    print(f"Full probe report -> {output_report}", file=sys.stderr)


if __name__ == "__main__":
    main()
