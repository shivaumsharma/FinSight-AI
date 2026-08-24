"""
train_sft.py

Rejection-sampling SFT (STaR-style) training entrypoint -- the sibling
to train_grpo.py, using scripts/build_sft_dataset.py's curated
(prompt, completion) pairs instead of GRPO's (prompt, realized_return_pct)
rows. See build_sft_dataset.py's own docstring for why: GRPO's 2026-08-21
runs showed the mechanism working (50% of steps had real gradient by
run 2, up from ~7%) but accuracy still didn't move (40.2% base -> 41.0%
-> 40.2%), because 15-30 sparse binary-reward steps over a 23-example
set is too little total learning signal even with a working gradient.
Standard supervised cross-entropy gives a gradient on every token of
every training row, not just the ones lucky enough to show reward
variance within a sampled group -- this is the "SFT warm-start" half of
the standard SFT-then-RL pipeline, which the project's first RLVR pass
skipped by going straight to GRPO.

Needs the same `pip install -r requirements-train.txt` as train_grpo.py
(trl, peft, bitsandbytes) -- no new dependency, trl.SFTTrainer/SFTConfig
are already in the same package.

Usage
-----
Real run:
    python scripts/train_sft.py --dataset scripts/rlvr_dataset_train_v2_sft.jsonl \\
        --load-in-4bit --gradient-checkpointing

CPU wiring smoke test:
    python scripts/train_sft.py --model Qwen/Qwen2.5-0.5B-Instruct \\
        --dataset scripts/rlvr_dataset_train_v2_sft.jsonl --max-steps 2
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATASET = str(Path(__file__).resolve().parent / "rlvr_dataset_train_v2_sft.jsonl")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "sft_output")

# Same local MLflow store train_grpo.py / train_ml_classifier.py already
# write to -- a separate experiment name ("finsight-sft-rejection-sampling")
# within it so these runs don't mix with the GRPO ones.
_MLRUNS_DIR = Path(__file__).resolve().parent.parent / "mlruns"
_MLRUNS_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{_MLRUNS_DIR / 'mlflow.db'}")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "finsight-sft-rejection-sampling")

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Base model (default: {DEFAULT_MODEL})")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to a JSONL dataset with 'prompt'/'completion' columns")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-train-epochs", type=float, default=3.0, help="Small dataset (~60-130 rows) -- a few epochs, not 1, matching typical LoRA SFT practice")
    p.add_argument("--max-steps", type=int, default=-1, help="Cap total training steps (-1 = no cap); set low for a smoke test")
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument(
        "--learning-rate", type=float, default=2e-4,
        help=(
            "Deliberately higher than train_grpo.py's 1e-5 -- that value is an RL-appropriate LR "
            "(small, stable policy updates); LoRA supervised fine-tuning on a small, clean dataset "
            "is a different regime and standard practice uses ~1e-4 to 2e-4 here."
        ),
    )
    p.add_argument("--max-length", type=int, default=1024, help="Prompt+completion token cap (financial-summary prompts + <=256-token completions comfortably fit)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--no-lora", action="store_true", help="Full fine-tune instead of LoRA (needs far more VRAM -- not recommended)")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    return p.parse_args()


def _model_init_kwargs(args, torch):
    """Identical to train_grpo.py's helper of the same name -- same
    4-bit/device-map constraints apply to any trainer loading this model
    on this GPU, RL or supervised."""
    if not torch.cuda.is_available():
        return {"device_map": None, "low_cpu_mem_usage": False}

    if not args.load_in_4bit:
        return {}

    from transformers import BitsAndBytesConfig

    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        "device_map": {"": 0},
    }


def main():
    args = _parse_args()

    try:
        import torch
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
        from peft import LoraConfig
    except ImportError as e:
        raise SystemExit(
            "Missing training dependencies. Run:\n"
            "    pip install -r requirements-train.txt\n"
            f"(original error: {e})"
        )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    print(f"Loaded {len(dataset)} SFT examples from {args.dataset}", file=sys.stderr)

    if args.load_in_4bit and args.no_lora:
        raise SystemExit("--load-in-4bit requires LoRA (can't backprop into 4-bit-quantized base weights) -- drop --no-lora.")
    if args.load_in_4bit and not torch.cuda.is_available():
        raise SystemExit("--load-in-4bit needs a CUDA GPU (bitsandbytes 4-bit has no CPU path).")

    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=1,
        # completion_only_loss left at its default (None): SFTTrainer
        # auto-detects it from the dataset having "prompt"+"completion"
        # columns (see trl/trainer/sft_trainer.py's __init__), which
        # build_sft_dataset.py's output always has -- masks the prompt
        # tokens out of the loss automatically, no formatting_func needed.
        report_to=[],  # same reason as train_grpo.py: post-hoc MLflow logging below, not trl's live callback
        use_cpu=not torch.cuda.is_available(),
        model_init_kwargs=_model_init_kwargs(args, torch),
    )

    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved trained adapter -> {args.output_dir}", file=sys.stderr)

    _log_run_to_mlflow(trainer, args)


def _log_run_to_mlflow(trainer, args) -> None:
    """Same post-hoc-logging pattern as train_grpo.py's own helper, for
    the same reason (trl's internal metric names can contain characters
    MLflow's strict validator rejects live) -- reads
    trainer.state.log_history after the fact instead."""
    try:
        import re

        import mlflow

        mlflow.set_experiment("finsight-sft-rejection-sampling")
        with mlflow.start_run(run_name=f"{Path(args.model).name}_sft_{'4bit' if args.load_in_4bit else 'full'}"):
            mlflow.log_params({
                "model": args.model,
                "dataset": args.dataset,
                "num_train_epochs": args.num_train_epochs,
                "max_steps": args.max_steps,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
                "load_in_4bit": args.load_in_4bit,
                "gradient_checkpointing": args.gradient_checkpointing,
            })
            for entry in trainer.state.log_history:
                step = entry.get("step")
                for key, value in entry.items():
                    if key == "step" or not isinstance(value, (int, float)):
                        continue
                    safe_key = re.sub(r"[^A-Za-z0-9_\-. /]", "_", key)
                    mlflow.log_metric(safe_key, value, step=step)
            mlflow.log_artifacts(args.output_dir, artifact_path="adapter")
        print("Logged MLflow run -> mlruns/ (run `mlflow ui` to browse)", file=sys.stderr)
    except Exception as e:
        print(f"[warning] MLflow logging failed (adapter was still saved successfully): {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
