"""
train_grpo.py

GRPO (Group Relative Policy Optimization) training entrypoint for the
RLVR direction-prediction task -- see app/training/rlvr_prompt.py and
rlvr_reward.py for the prompt/reward design, and
scripts/build_rlvr_dataset.py for how the training data is produced.

Targets a separate, local open model via LoRA (peft) -- NOT the hosted
Groq API this project's live report generation uses today (a hosted
chat-completions API can't be weight-updated by GRPO; see the project
plan doc's "Key finding" section).

Needs `pip install -r requirements-train.txt` (trl, peft, bitsandbytes)
-- deliberately NOT part of requirements.txt or requirements-dev.txt,
since these are heavy, GPU-oriented, training-only dependencies that
would bloat the deploy/CI path for no benefit (same reasoning as this
project's existing three-tier requirements split).

Usage
-----
Real run (needs a GPU for a 7B model in practice):
    python scripts/train_grpo.py --dataset scripts/rlvr_dataset_train.jsonl

CPU wiring smoke test (see project plan doc's verification step 3 --
confirms dataset -> generation -> parse -> reward -> GRPO loss runs
end to end, before spending any GPU-rental money):
    python scripts/train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct \\
        --dataset scripts/rlvr_dataset_train.jsonl \\
        --max-steps 2 --num-generations 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.training.rlvr_reward import grpo_reward_fn

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # see project plan doc's "Model choice" section for why 7B, not 1.5B
DEFAULT_DATASET = str(Path(__file__).resolve().parent / "rlvr_dataset_train.jsonl")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "grpo_output")

# Standard attention-projection LoRA targets for Qwen2.5-family models
# (q_proj/k_proj/v_proj/o_proj) -- matches the architecture's actual
# module names, not a generic guess.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Base model (default: {DEFAULT_MODEL})")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to a JSONL dataset with a 'prompt' column")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-generations", type=int, default=8, help="Rollouts sampled per prompt (GRPO group size)")
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help="Cap total training steps (-1 = no cap); set low for a smoke test")
    p.add_argument(
        "--per-device-train-batch-size", type=int, default=None,
        help=(
            "Defaults to --num-generations if not set -- trl's GRPOConfig requires the effective "
            "generation batch (per_device_train_batch_size * steps_per_generation) to be evenly "
            "divisible by num_generations; matching the two by default sidesteps that constraint "
            "for the common case. Override only if you also know what steps_per_generation you need."
        ),
    )
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--no-lora", action="store_true", help="Full fine-tune instead of LoRA (needs far more VRAM -- not recommended)")
    return p.parse_args()


def main():
    args = _parse_args()

    try:
        import torch
        from datasets import load_dataset
        from trl import GRPOConfig, GRPOTrainer
        from peft import LoraConfig
    except ImportError as e:
        raise SystemExit(
            "Missing training dependencies. Run:\n"
            "    pip install -r requirements-train.txt\n"
            f"(original error: {e})"
        )

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    print(f"Loaded {len(dataset)} training examples from {args.dataset}", file=sys.stderr)

    per_device_train_batch_size = args.per_device_train_batch_size or args.num_generations

    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        # >0 so rollouts within a group actually differ -- GRPO's
        # advantage estimate needs variance across the num_generations
        # samples for the SAME prompt; temperature=0 would make every
        # rollout identical and every advantage zero.
        temperature=0.8,
        logging_steps=1,
        report_to=[],
        # Auto-detected, not hardcoded either way: a real GPU training
        # run must NOT force CPU, but transformers' TrainingArguments
        # otherwise assumes bf16/GPU by default and refuses to run on a
        # CPU-only smoke-test machine without this explicitly set.
        use_cpu=not torch.cuda.is_available(),
        # Without this, from_pretrained's automatic device_map/disk-
        # offload heuristics can split a model across the real "cpu"
        # device and a placeholder "meta" device (seen during the CPU
        # smoke test: backward() then fails with "expected device meta
        # but got cpu", since gradients can't flow between the two).
        # Forcing everything onto one real device sidesteps that.
        model_init_kwargs=(
            {} if torch.cuda.is_available()
            else {"device_map": None, "low_cpu_mem_usage": False}
        ),
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=grpo_reward_fn,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved trained adapter -> {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
