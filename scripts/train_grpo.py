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
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.training.rlvr_reward import grpo_reward_fn

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # see project plan doc's "Model choice" section for why 7B, not 1.5B
DEFAULT_DATASET = str(Path(__file__).resolve().parent / "rlvr_dataset_train.jsonl")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "grpo_output")

# Same local MLflow store scripts/train_ml_classifier.py already writes
# to -- one shared experiment-tracking store for the whole project,
# "finsight-rlvr-grpo" as its own experiment within it so it doesn't mix
# with the valuation classifier's runs. Set via env vars (not
# mlflow.set_tracking_uri() in-process) because report_to=["mlflow"]
# below hands logging to transformers' own MLflowCallback, which reads
# its target from these same env vars rather than any config passed
# into GRPOConfig directly.
_MLRUNS_DIR = Path(__file__).resolve().parent.parent / "mlruns"
_MLRUNS_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{_MLRUNS_DIR / 'mlflow.db'}")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "finsight-rlvr-grpo")

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
    p.add_argument(
        "--load-in-4bit", action="store_true",
        help=(
            "QLoRA-style 4-bit quantized base model (bitsandbytes NF4) -- what actually makes the "
            "default 7B model fit on a consumer/laptop GPU (e.g. a 6-8GB card) instead of needing a "
            "rented A100. Only meaningful with a CUDA device and --lora (the default); has no effect "
            "on CPU (bitsandbytes 4-bit needs a GPU) and is incompatible with --no-lora (you can't "
            "backprop into 4-bit-quantized base weights)."
        ),
    )
    p.add_argument(
        "--gradient-checkpointing", action="store_true",
        help=(
            "Recompute activations during the backward pass instead of storing them -- trades some "
            "training speed for meaningfully lower activation memory, real headroom on a tight VRAM "
            "budget (e.g. lets --max-completion-length go higher on the same 6-8GB card that only "
            "just fit the smoke test without this)."
        ),
    )
    return p.parse_args()


def _model_init_kwargs(args, torch):
    if not torch.cuda.is_available():
        # Same "meta" vs. real-device split from_pretrained's auto
        # device_map heuristics can produce -- see the caller's own
        # comment for the CPU smoke-test failure this avoids.
        return {"device_map": None, "low_cpu_mem_usage": False}

    if not args.load_in_4bit:
        return {}

    from transformers import BitsAndBytesConfig

    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            # Nested (double) quantization -- an extra ~0.4 bits/param
            # saved by quantizing the quantization constants themselves,
            # real headroom on a 6-8GB card where every few hundred MB
            # of base-model footprint is the difference between fitting
            # and OOMing on the LoRA/optimizer/activation memory left over.
            bnb_4bit_use_double_quant=True,
        ),
        # bitsandbytes' 4-bit layers require every param to already be
        # materialized on ONE real CUDA device before quantization --
        # letting from_pretrained's default heuristic auto-shard across
        # devices (or a meta device) breaks that assumption.
        "device_map": {"": 0},
    }


def main():
    args = _parse_args()

    try:
        import torch
        from datasets import load_dataset
        from transformers import GenerationConfig
        from trl import GRPOConfig, GRPOTrainer
        from peft import LoraConfig
    except ImportError as e:
        raise SystemExit(
            "Missing training dependencies. Run:\n"
            "    pip install -r requirements-train.txt\n"
            f"(original error: {e})"
        )

    if torch.cuda.is_available():
        # Free speedup on any Ampere-or-newer GPU (e.g. RTX 30-series):
        # matmuls/convolutions run in TF32 instead of full FP32 with a
        # precision loss (~10 bits of mantissa vs. 23) that's a non-issue
        # here -- LoRA training is already bf16 compute, and generation/
        # reward scoring don't need FP32 precision either. Off by default
        # in torch even when the hardware supports it.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    print(f"Loaded {len(dataset)} training examples from {args.dataset}", file=sys.stderr)

    per_device_train_batch_size = args.per_device_train_batch_size or args.num_generations

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

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        gradient_checkpointing=args.gradient_checkpointing,
        # >0 so rollouts within a group actually differ -- GRPO's
        # advantage estimate needs variance across the num_generations
        # samples for the SAME prompt; temperature=0 would make every
        # rollout identical and every advantage zero.
        temperature=0.8,
        logging_steps=1,
        # NOT "mlflow" -- found live: trl's own internal profiling
        # metric names (e.g. "profiling/Time taken:
        # GRPOTrainer._prepare_inputs") contain a colon, which MLflow's
        # strict metric-name validator rejects outright, crashing the
        # entire run the first time that metric gets logged. A real
        # trl/MLflow incompatibility, not something retrying fixes --
        # MLflow logging for this script now happens after the fact
        # (see _log_run_to_mlflow below), reading the finished run's
        # own log file instead of relying on this live callback path.
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
        model_init_kwargs=_model_init_kwargs(args, torch),
        # GRPOTrainer builds its own internal generation_config from a
        # SINGLE `tokenizer.eos_token_id` (see trl's grpo_trainer.py) --
        # for Qwen2.5-Instruct that's only <|im_end|> (151645), not the
        # model's own real stopping set, `[151645, 151643]`
        # (<|im_end|> + <|endoftext|>), from generation_config.json.
        # Found live: every rollout in a real run hit the token cap and
        # NEVER stopped naturally (completions/mean_terminated_length:
        # 0 on every single step) despite the identical prompt closing
        # cleanly in 90-120 tokens under plain model.generate() with no
        # generation_config override at all -- this override, not more
        # max_completion_length, was the actual fix.
        generation_kwargs={"eos_token_id": GenerationConfig.from_pretrained(args.model).eos_token_id},
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

    _log_run_to_mlflow(trainer, args)


def _log_run_to_mlflow(trainer, args) -> None:
    """
    Logs this finished run to the same local MLflow store
    scripts/train_ml_classifier.py uses, under its own
    "finsight-rlvr-grpo" experiment -- deliberately AFTER training
    completes, not via trl's live report_to=["mlflow"] callback (see
    that flag's own comment above for why: trl logs a profiling metric
    name containing a colon, which crashed the run outright the one
    time this was tried live). Reads trainer.state.log_history, which
    the Trainer always populates regardless of report_to, so no
    tracking data is actually lost by moving this after the fact.
    Wrapped in try/except -- a logging failure here must never cost a
    successful training run its saved adapter.
    """
    try:
        import re

        import mlflow

        mlflow.set_experiment("finsight-rlvr-grpo")
        with mlflow.start_run(run_name=f"{Path(args.model).name}_{'4bit' if args.load_in_4bit else 'full'}_{args.max_steps}steps"):
            mlflow.log_params({
                "model": args.model,
                "max_steps": args.max_steps,
                "num_generations": args.num_generations,
                "max_completion_length": args.max_completion_length,
                "load_in_4bit": args.load_in_4bit,
                "gradient_checkpointing": args.gradient_checkpointing,
                "learning_rate": args.learning_rate,
            })
            for entry in trainer.state.log_history:
                step = entry.get("step")
                for key, value in entry.items():
                    if key == "step" or not isinstance(value, (int, float)):
                        continue
                    # MLflow metric names allow only alnum/_/-/./space//
                    # -- trl's own metric names (e.g. "profiling/Time
                    # taken: X") use a colon this rejects outright.
                    safe_key = re.sub(r"[^A-Za-z0-9_\-. /]", "_", key)
                    mlflow.log_metric(safe_key, value, step=step)
            mlflow.log_artifacts(args.output_dir, artifact_path="adapter")
        print("Logged MLflow run -> mlruns/ (run `mlflow ui` to browse)", file=sys.stderr)
    except Exception as e:
        print(f"[warning] MLflow logging failed (adapter was still saved successfully): {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
