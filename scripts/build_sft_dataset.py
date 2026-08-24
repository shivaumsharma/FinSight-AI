"""
build_sft_dataset.py

Rejection-sampling SFT dataset builder (STaR-style: Zelikman et al.,
"STaR: Bootstrapping Reasoning With Reasoning") -- the follow-up to
filter_informative_examples.py's finding that only 23/140 (16.4%) of
scripts/rlvr_dataset_train_v2.jsonl produced any GRPO gradient at all,
because GRPO can only learn from a training example when the sampled
group DISAGREES with itself. That's a needlessly high bar: an example
the base model gets right on EVERY sample is useless to GRPO (zero
variance) but is a perfectly good supervised-fine-tuning target -- we
already have a verified-correct (prompt, completion) pair sitting right
there, for free.

This script re-probes the same training set (same N_PROBES=4 at the
same temperature=0.8 as filter_informative_examples.py, for an
apples-to-apples probe), but instead of only recording pass/fail
variance, it SAVES the completion text of every sample that scores
correct via the real app.training.rlvr_reward.compute_reward. Standard
supervised cross-entropy loss then gives a gradient on every single one
of those rows -- no reward-variance requirement, no wasted examples.

From the 2026-08-21 probe report (rlvr_dataset_train_v2_probe_report.json):
  44 examples were always-correct (4/4)   -- free SFT targets, GRPO discarded these
  23 examples were mixed (GRPO-informative) -- usually 1-3/4 correct, also usable here
  73 examples were always-wrong (0/4)     -- no correct sample to imitate, skipped
That's 67 unique examples with >=1 usable target instead of GRPO's 23,
from the exact same 140-example dataset -- no new tickers needed.

Usage:
    python scripts/build_sft_dataset.py --dataset scripts/rlvr_dataset_train_v2.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.training.rlvr_reward import compute_reward

DEFAULT_DATASET = str(Path(__file__).resolve().parent / "rlvr_dataset_train_v2.jsonl")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Same N as filter_informative_examples.py's probe, so this reuses the
# exact same sampling distribution that already-computed report
# describes -- not a fresh, differently-calibrated probe.
N_SAMPLES = 4

# Caps how many correct completions from the same prompt get kept as
# separate SFT rows. Some redundancy across samples for the SAME prompt
# is useful (a little phrasing diversity beats one memorized answer),
# but keeping all 4 for a 4/4-correct example is mostly near-duplicate
# text for one data point -- 2 balances that against not exploding the
# dataset with near-identical rows for the "easy" examples specifically.
MAX_CORRECT_PER_EXAMPLE = 2


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n-samples", type=int, default=N_SAMPLES)
    p.add_argument("--max-correct-per-example", type=int, default=MAX_CORRECT_PER_EXAMPLE)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument(
        "--output-sft", default=None,
        help="Path for the SFT (prompt, completion) JSONL. Defaults to <dataset>_sft.jsonl.",
    )
    p.add_argument(
        "--output-report", default=None,
        help="Path for the full per-example probe report (JSON, includes examples with zero "
             "usable completions too, for inspection). Defaults to <dataset>_sft_report.json.",
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
    # diagnose_rlvr_generation.py / filter_informative_examples.py --
    # this transformers version returns a BatchEncoding, not a raw tensor.
    input_ids = (templated.input_ids if hasattr(templated, "input_ids") else templated).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,  # matches train_grpo.py's own sampling temperature and filter_informative_examples.py's probe
            pad_token_id=tokenizer.eos_token_id,
        )
    completion_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def probe_example(tokenizer, model, row, n_samples, max_new_tokens):
    """Samples n_samples completions and returns every one that scores
    correct against the real GRPO reward function -- these are the raw
    material for SFT rows, before the max-per-example cap is applied."""
    samples = []
    for _ in range(n_samples):
        completion = _generate_one(tokenizer, model, row["prompt"], max_new_tokens)
        reward = compute_reward(completion, row["realized_return_pct"])
        samples.append({"completion": completion, "reward": reward})
    return samples


def main():
    args = _parse_args()
    dataset_path = Path(args.dataset)
    output_sft = Path(args.output_sft) if args.output_sft else \
        dataset_path.with_name(dataset_path.stem + "_sft.jsonl")
    output_report = Path(args.output_report) if args.output_report else \
        dataset_path.with_name(dataset_path.stem + "_sft_report.json")

    rows = [json.loads(line) for line in open(dataset_path, encoding="utf-8") if line.strip()]
    print(f"Probing {len(rows)} examples x {args.n_samples} samples each "
          f"({len(rows) * args.n_samples} total generations)...", file=sys.stderr)

    tokenizer, model = _load_model(args.model, args.load_in_4bit)

    report = []
    sft_rows = []
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['ticker']} ({row['as_of_date']})...", file=sys.stderr)
        samples = probe_example(tokenizer, model, row, args.n_samples, args.max_new_tokens)
        correct = [s["completion"] for s in samples if s["reward"] == 1.0]
        # Dedupe exact-match repeats (the same greedy-ish completion
        # sampled twice at temperature=0.8 does happen) before capping --
        # otherwise the cap could keep two identical rows instead of two
        # actually-different ones.
        unique_correct = list(dict.fromkeys(correct))[: args.max_correct_per_example]

        report.append({
            "ticker": row["ticker"], "as_of_date": row["as_of_date"],
            "n_correct": len(correct), "n_kept": len(unique_correct),
            "samples": samples,
        })
        for completion in unique_correct:
            sft_rows.append({
                "ticker": row["ticker"], "as_of_date": row["as_of_date"],
                "prompt": row["prompt"], "completion": completion,
                "realized_return_pct": row["realized_return_pct"],
            })
        print(f"  {len(correct)}/{args.n_samples} correct, kept {len(unique_correct)}", file=sys.stderr)

    with open(output_sft, "w", encoding="utf-8") as f:
        for row in sft_rows:
            f.write(json.dumps(row) + "\n")

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    n_usable_examples = sum(1 for r in report if r["n_correct"] > 0)
    print(f"\n{n_usable_examples}/{len(rows)} examples had >=1 correct sample "
          f"({100 * n_usable_examples / len(rows):.1f}%)", file=sys.stderr)
    print(f"SFT dataset: {len(sft_rows)} (prompt, completion) rows -> {output_sft}", file=sys.stderr)
    print(f"Full probe report -> {output_report}", file=sys.stderr)


if __name__ == "__main__":
    main()
