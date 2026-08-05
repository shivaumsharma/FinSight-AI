"""
evaluate_rlvr_model.py

The honest final check for the RLVR pipeline -- see the project plan
doc's "New files" section. Runs a (base or RL-trained) model's direction
calls on the HELD-OUT dataset only (scripts/rlvr_dataset_heldout.jsonl,
produced by build_rlvr_dataset.py's two-axis ticker+window split), and
scores them with scripts/phase2_backtest.py's own score_rating() and
_naive_baseline_accuracy() -- reused directly, not reimplemented, so
"the model beat/lost to naive Always-Buy by N points" means exactly what
it already means everywhere else in this project (EVALUATION.md).

Deliberately does NOT trust the training loop's own reward curve as the
result -- a policy can look great on its own training objective while
still being wrong on the metric that matters (realized-return accuracy
on data it never trained on). This script is the independent judge.

Usage
-----
Evaluate a LoRA-adapted checkpoint against the base model, side by side:
    python scripts/evaluate_rlvr_model.py --adapter-dir scripts/grpo_output

Evaluate the base model alone (e.g. to get a pre-RLVR reference number):
    python scripts/evaluate_rlvr_model.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2_backtest import BUY_THRESHOLD, SELL_THRESHOLD, _naive_baseline_accuracy, score_rating

from app.training.rlvr_prompt import parse_verdict

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_HELDOUT_PATH = str(Path(__file__).resolve().parent / "rlvr_dataset_heldout.jsonl")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Base model (default: {DEFAULT_MODEL})")
    p.add_argument("--adapter-dir", default=None, help="Path to a trained LoRA adapter (scripts/train_grpo.py's --output-dir). Omit to evaluate the base model alone.")
    p.add_argument("--dataset", default=DEFAULT_HELDOUT_PATH, help="Held-out JSONL dataset")
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def _load_model(model_name, adapter_dir):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "Missing training/inference dependencies. Run:\n"
            "    pip install -r requirements-train.txt\n"
            f"(original error: {e})"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    if adapter_dir:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise SystemExit(
                "Missing peft (needed to load a LoRA adapter). Run:\n"
                "    pip install -r requirements-train.txt\n"
                f"(original error: {e})"
            )
        model = PeftModel.from_pretrained(model, adapter_dir)

    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return tokenizer, model


def _generate(tokenizer, model, prompt, max_new_tokens):
    import torch

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for evaluation -- GRPO's exploration temperature belongs in training, not the judged eval run
            pad_token_id=tokenizer.eos_token_id,
        )
    completion_ids = output_ids[0, inputs.shape[1]:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def main():
    args = _parse_args()

    rows = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise SystemExit(f"No examples found in {args.dataset} -- run scripts/build_rlvr_dataset.py first.")

    print(f"Held-out examples: {len(rows)}   Model: {args.model}   Adapter: {args.adapter_dir or '(none -- base model)'}",
          file=sys.stderr)
    print(f"Accuracy bands (from phase2_backtest.py): Buy>+{BUY_THRESHOLD}%  Sell<{SELL_THRESHOLD}%  "
          f"Hold in [{SELL_THRESHOLD}%,+{BUY_THRESHOLD}%]", file=sys.stderr)

    tokenizer, model = _load_model(args.model, args.adapter_dir)

    scored = []
    unparseable = 0
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['ticker']} ({row['as_of_date']})...", file=sys.stderr)
        completion = _generate(tokenizer, model, row["prompt"], args.max_new_tokens)
        verdict = parse_verdict(completion)
        if verdict is None:
            unparseable += 1
            print(f"  [unparseable] {completion[:120]!r}", file=sys.stderr)
            continue
        correct = score_rating(verdict, row["realized_return_pct"])
        scored.append({**row, "recommendation": verdict, "correct": correct})
        print(f"  verdict={verdict}  realized={row['realized_return_pct']:+.1f}%  correct={correct}", file=sys.stderr)

    valid = [r for r in scored if r["correct"] is not None]
    overall_acc = 100 * sum(r["correct"] for r in valid) / len(valid) if valid else None

    print(file=sys.stderr)
    print(f"Scored: {len(valid)}/{len(rows)}   Unparseable: {unparseable}", file=sys.stderr)
    print(f"Model accuracy: {overall_acc:.1f}%" if overall_acc is not None else "Model accuracy: N/A (no scoreable rows)",
          file=sys.stderr)

    for fixed_rating in ("Buy", "Hold", "Sell"):
        acc = _naive_baseline_accuracy(scored, fixed_rating)
        if acc is not None:
            print(f"  vs. Always-{fixed_rating}: {acc:.1f}%"
                  + (f"  ({overall_acc - acc:+.1f} points)" if overall_acc is not None else ""),
                  file=sys.stderr)


if __name__ == "__main__":
    main()
