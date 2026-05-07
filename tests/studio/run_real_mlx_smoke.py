# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
End-to-end MLX smoke test on real Apple Silicon.

Trains `unsloth/gemma-3-270m-it` for 7 deterministic LoRA steps on an
in-memory dataset of the SAME row repeated:

    "<<HELLO!!>> My name is Unsloth!"

then asks the trained model to complete the prompt

    "<<HELLO!!>> My name is "

and asserts the completion contains "Unsloth".

Captures and asserts:

  - Per-step training loss (from MLXTrainer's add_step_callback).
  - Loss is finite and does not diverge across the 7 steps.
  - Pre- and post-training gradient norms (computed manually via
    mx.nn.value_and_grad over a single batch of the training text;
    the trainer does not currently expose per-step grad norms).
  - Inference output contains "Unsloth".

This script is only runnable on a real Apple Silicon host (the import
chain pulls real `mlx`, `mlx-lm`, and `unsloth_zoo.mlx_*`). It is
invoked from .github/workflows/mlx-ci.yml on the macos-14 runner.

Determinism: seeds Python `random`, `numpy`, and `mlx.core.random`
before any MLX import, and forwards `random_state=SEED` to both
`FastMLXModel.from_pretrained` and `FastMLXModel.get_peft_model`
(both call `_seed_mlx_random_state` internally), and `seed=SEED` to
`MLXTrainingConfig` (drives batch shuffling). Metal still has minor
nondeterminism from reduction-order in atomics, so loss assertions
are bounds rather than exact-match.
"""

from __future__ import annotations

import math
import os
import random as _random
import sys

import numpy as np


SEED = 3407


def _seed_everything() -> None:
    _random.seed(SEED)
    np.random.seed(SEED)
    # mlx.core.random must be seeded after the import; we can't avoid
    # the import here. This must run BEFORE FastMLXModel.from_pretrained.
    import mlx.core as mx

    mx.random.seed(SEED)


def _compute_loss_and_grad_norm(model, tokenizer, text: str) -> tuple[float, float]:
    """Run one forward+backward over a single training example and
    return (loss, ||grad||_2) so we can compare pre- vs post-training.

    Uses the same next-token cross-entropy loss the trainer uses (no
    masking — the tiny synthetic dataset has no instruction/response
    split).
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    ids = list(tokenizer.encode(text))
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        ids.append(int(eos_id))
    if len(ids) < 2:
        raise RuntimeError(
            f"tokenized text too short to compute loss: {len(ids)} tokens"
        )

    inputs = mx.array([ids[:-1]], dtype = mx.int32)
    targets = mx.array([ids[1:]], dtype = mx.int32)

    def loss_fn(m):
        logits = m(inputs)
        return nn.losses.cross_entropy(logits, targets, reduction = "mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss_val, grad = loss_and_grad(model)

    norm_sq = mx.array(0.0, dtype = mx.float32)
    for _name, value in tree_flatten(grad):
        norm_sq = norm_sq + mx.sum(value.astype(mx.float32) * value.astype(mx.float32))
    grad_norm = mx.sqrt(norm_sq)

    return float(loss_val.item()), float(grad_norm.item())


def main() -> int:
    _seed_everything()

    import mlx.core as mx
    from unsloth_zoo.mlx_loader import FastMLXModel
    from unsloth_zoo.mlx_trainer import MLXTrainer, MLXTrainingConfig

    text_row = "<<HELLO!!>> My name is Unsloth!"
    model_name = "unsloth/gemma-3-270m-it"
    hf_token = os.environ.get("HF_TOKEN") or None

    print(f"Loading {model_name} (fp16, no quant)...", flush = True)
    model, tokenizer = FastMLXModel.from_pretrained(
        model_name,
        load_in_4bit = False,
        dtype = "float16",
        text_only = True,
        max_seq_length = 128,
        random_state = SEED,
        token = hf_token,
        trust_remote_code = False,
    )

    # Re-seed RNG between load and LoRA injection so the LoRA init is
    # reproducible regardless of how many random draws the loader did.
    mx.random.seed(SEED)

    print("Applying LoRA r=8 on attention modules...", flush = True)
    model = FastMLXModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0.0,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing = False,
        random_state = SEED,
        finetune_language_layers = True,
        finetune_attention_modules = True,
        finetune_mlp_modules = False,
    )

    # Tiny synthetic in-memory dataset: same row repeated. The trainer
    # consumes any iterable of dicts with the dataset_text_field key.
    dataset = [{"text": text_row}] * 32

    print("Pre-training loss + grad norm (single-batch probe)...", flush = True)
    pre_loss, pre_grad_norm = _compute_loss_and_grad_norm(model, tokenizer, text_row)
    print(f"  pre  loss={pre_loss:.4f}  grad_norm={pre_grad_norm:.4f}", flush = True)
    assert math.isfinite(pre_loss), f"pre-train loss is non-finite: {pre_loss}"
    assert math.isfinite(
        pre_grad_norm
    ), f"pre-train grad_norm is non-finite: {pre_grad_norm}"
    assert pre_grad_norm > 0, f"pre-train grad_norm is zero: {pre_grad_norm}"

    print("Constructing MLXTrainer (max_steps=7, lr=1e-3, bs=2)...", flush = True)
    config = MLXTrainingConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 1,
        max_steps = 7,
        learning_rate = 1e-3,
        warmup_steps = 0,
        lr_scheduler_type = "constant",
        optim = "adamw",
        weight_decay = 0.0,
        max_grad_norm = 1.0,
        logging_steps = 1,
        max_seq_length = 64,
        seed = SEED,
        use_cce = False,
        compile = False,
        gradient_checkpointing = False,
        output_dir = "/tmp/unsloth_mlx_smoke",
        save_steps = 0,
        eval_steps = 0,
        dataset_text_field = "text",
    )

    trainer = MLXTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        args = config,
    )

    losses: list[tuple[int, float]] = []
    lrs: list[tuple[int, float]] = []

    def _on_step(step, total, loss, lr, tok_s, peak_gb, elapsed, num_tokens):
        losses.append((int(step), float(loss)))
        lrs.append((int(step), float(lr)))
        print(
            f"  step {step}/{total}  loss={loss:.4f}  lr={lr:.2e}  "
            f"tok/s={tok_s:.0f}  peak={peak_gb:.2f}GB",
            flush = True,
        )

    trainer.add_step_callback(_on_step)

    print("Running 7 training steps...", flush = True)
    train_result = trainer.train()
    print(f"Trainer summary: {train_result}", flush = True)

    print("Post-training loss + grad norm (single-batch probe)...", flush = True)
    post_loss, post_grad_norm = _compute_loss_and_grad_norm(model, tokenizer, text_row)
    print(f"  post loss={post_loss:.4f}  grad_norm={post_grad_norm:.4f}", flush = True)

    # Loss + grad norm assertions
    assert len(losses) == 7, f"expected 7 step callbacks, got {len(losses)}: {losses}"
    for step, loss in losses:
        assert math.isfinite(loss), f"step {step} loss not finite: {loss}"
        assert 0 < loss < 50, f"step {step} loss out of bounds: {loss}"

    first_loss = losses[0][1]
    last_loss = losses[-1][1]
    print(f"loss[0]={first_loss:.4f} loss[6]={last_loss:.4f}", flush = True)
    # On a single repeated row the model should bend towards the data.
    # Allow some headroom for Metal nondeterminism but require we are
    # not wildly diverging.
    assert last_loss < first_loss * 1.1, (
        f"loss diverged across 7 steps: first={first_loss:.4f} " f"last={last_loss:.4f}"
    )
    assert math.isfinite(post_loss), f"post-train loss not finite: {post_loss}"
    assert math.isfinite(
        post_grad_norm
    ), f"post-train grad_norm not finite: {post_grad_norm}"
    assert post_loss < pre_loss, (
        f"post-train loss {post_loss:.4f} >= pre-train loss {pre_loss:.4f} — "
        f"7 steps of LoRA on a single repeated row should reduce loss"
    )

    # Inference: prompt -> "Unsloth" continuation
    print("Inference: completing '<<HELLO!!>> My name is '...", flush = True)
    from mlx_lm import generate

    model.eval()
    prompt = "<<HELLO!!>> My name is "
    output = generate(
        model,
        tokenizer,
        prompt = prompt,
        max_tokens = 24,
        verbose = False,
    )
    print(f"  prompt: {prompt!r}", flush = True)
    print(f"  output: {output!r}", flush = True)
    assert "Unsloth" in output, (
        f"expected 'Unsloth' in completion of {prompt!r}; got {output!r}. "
        f"Loss went {first_loss:.4f}->{last_loss:.4f}, post={post_loss:.4f}, "
        f"pre_grad_norm={pre_grad_norm:.4f} post_grad_norm={post_grad_norm:.4f}."
    )

    print(
        f"\nOK: real-MLX training+inference smoke passed.\n"
        f"  losses: {[round(l, 4) for _, l in losses]}\n"
        f"  pre  loss={pre_loss:.4f} grad_norm={pre_grad_norm:.4f}\n"
        f"  post loss={post_loss:.4f} grad_norm={post_grad_norm:.4f}\n"
        f"  generation: {output!r}",
        flush = True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
