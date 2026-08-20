"""
evaluate_rlvr_model.py

The honest final check for the RLVR pipeline -- see the project plan
doc's "New files" section. Runs a (base or RL-trained) model's direction
calls on the HELD-OUT dataset only (scripts/rlvr_dataset_heldout.jsonl,
produced by build_rlvr_dataset.py's two-axis ticker+window split), and
scores them with scripts/phase2_backtest.py's own score_rating() and
naive_baseline_accuracy() -- reused directly, not reimplemented, so
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

from phase2_backtest import BUY_THRESHOLD, SELL_THRESHOLD, naive_baseline_accuracy, score_rating

from app.training.rlvr_prompt import parse_verdict

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_HELDOUT_PATH = str(Path(__file__).resolve().parent / "rlvr_dataset_heldout.jsonl")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Base model (default: {DEFAULT_MODEL})")
    p.add_argument("--adapter-dir", default=None, help="Path to a trained LoRA adapter (scripts/train_grpo.py's --output-dir). Omit to evaluate the base model alone.")
    p.add_argument("--dataset", default=DEFAULT_HELDOUT_PATH, help="Held-out JSONL dataset")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--load-in-4bit", action="store_true",
        help=(
            "Same QLoRA-style 4-bit quantized load as scripts/train_grpo.py's own flag -- "
            "needed to fit the default 7B model on a consumer/laptop GPU (e.g. 6-8GB). "
            "The adapter (if any) is applied on top of the quantized base, same as training."
        ),
    )
    return p.parse_args()


def _load_model(model_name, adapter_dir, load_in_4bit):
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

    if load_in_4bit:
        if not torch.cuda.is_available():
            raise SystemExit("--load-in-4bit needs a CUDA GPU (bitsandbytes 4-bit has no CPU path).")
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

    # A 4-bit-quantized model is already placed via device_map at load
    # time -- calling .to("cuda") on it again is unnecessary and, for a
    # bitsandbytes-quantized model, unsafe (it can silently break the
    # quantized layers' device bookkeeping).
    if torch.cuda.is_available() and not load_in_4bit:
        model = model.to("cuda")
    model.eval()
    return tokenizer, model


def _generate(tokenizer, model, prompt, max_new_tokens):
    import torch

    messages = [{"role": "user", "content": prompt}]
    templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    # This transformers version returns a BatchEncoding (dict-like, needs
    # .input_ids) rather than a raw tensor -- handle both rather than
    # assume one, found live via scripts/diagnose_rlvr_generation.py.
    inputs = (templated.input_ids if hasattr(templated, "input_ids") else templated).to(model.device)
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

    tokenizer, model = _load_model(args.model, args.adapter_dir, args.load_in_4bit)

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
        acc = naive_baseline_accuracy(scored, fixed_rating)
        if acc is not None:
            print(f"  vs. Always-{fixed_rating}: {acc:.1f}%"
                  + (f"  ({overall_acc - acc:+.1f} points)" if overall_acc is not None else ""),
                  file=sys.stderr)

    _print_confusion_matrix(valid)


def _true_bucket(realized_return_pct):
    """Same Buy/Hold/Sell bucketing score_rating() checks a rating
    against, applied directly to a realized return so a genuine "what
    actually happened" label exists to build a confusion matrix from --
    score_rating() itself only ever returns a bool for one rating at a
    time, not the bucket."""
    if realized_return_pct > BUY_THRESHOLD:
        return "Buy"
    if realized_return_pct < SELL_THRESHOLD:
        return "Sell"
    return "Hold"


def _print_confusion_matrix(valid_rows):
    """Per-class precision/recall/F1 and a full confusion matrix, not
    just the blended accuracy number above -- same rigor already
    applied to the ML valuation classifier (app/valuation/
    ml_valuation_classifier.py's train_and_evaluate), and for the same
    reason: one accuracy figure hides a model that's simply never
    calling one class at all (see that module's own docstring)."""
    if not valid_rows:
        return

    from sklearn.metrics import classification_report, confusion_matrix

    labels = ["Buy", "Hold", "Sell"]
    y_true = [_true_bucket(r["realized_return_pct"]) for r in valid_rows]
    y_pred = [r["recommendation"] for r in valid_rows]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion matrix (rows=actual, cols=predicted):", file=sys.stderr)
    print("            " + "".join(f"{l:>8}" for l in labels), file=sys.stderr)
    for label, row in zip(labels, cm):
        print(f"  actual {label:<5}" + "".join(f"{v:>8}" for v in row), file=sys.stderr)

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    print("\nPer-class precision/recall/F1:", file=sys.stderr)
    for label in labels:
        c = report.get(label, {})
        print(f"  {label:<6} precision={c.get('precision', 0):.2f}  recall={c.get('recall', 0):.2f}  "
              f"f1={c.get('f1-score', 0):.2f}  support={c.get('support', 0):.0f}", file=sys.stderr)


if __name__ == "__main__":
    main()
