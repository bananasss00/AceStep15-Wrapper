"""
Optimizer & Scheduler Factory for Full Fine-Tuning

Provides ``build_optimizer()`` and ``build_scheduler()`` so that
``full_finetune.py`` doesn't need to hard-code AdamW / CosineAnnealing.

Supported optimizers:
    adamw       -- torch.optim.AdamW (default, fused on CUDA)
    adamw8bit   -- bitsandbytes.optim.AdamW8bit (optional dep)
    adafactor   -- transformers.optimization.Adafactor
    prodigy     -- prodigyopt.Prodigy (optional dep, auto-tunes LR)
    prodigy_plus -- prodigyplus.ProdigyPlusScheduleFree (optional dep)
    ademamix    -- bundled acestep AdEMAMix

Supported schedulers:
    cosine              -- warmup + CosineAnnealingLR (single smooth decay)
    cosine_restarts     -- warmup + CosineAnnealingWarmRestarts (cyclical)
    linear              -- warmup + LinearLR decay to near-zero
    constant            -- warmup then flat LR
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ConstantLR,
    LinearLR,
    SequentialLR,
)

from loguru import logger


def build_optimizer(
    params: Iterable,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    device_type: str = "cuda",
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.optim.Optimizer:
    """Create an optimizer from a string key.

    Falls back to AdamW when an optional dependency is missing.
    """
    optimizer_type = optimizer_type.lower().strip()

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    # --- Prodigy Plus Schedule Free ---
    if optimizer_type == "prodigy_plus":
        try:
            from prodigyplus import ProdigyPlusScheduleFree

            logger.info("[Optim] Using ProdigyPlusScheduleFree optimizer (full config)")

            actual_lr = lr if lr != 1e-4 else 1.0

            return ProdigyPlusScheduleFree(
                params,
                lr=actual_lr,
                betas=optimizer_kwargs.get("betas", (0.9, 0.99)),
                beta3=optimizer_kwargs.get("beta3", None),
                weight_decay=weight_decay,
                weight_decay_by_lr=optimizer_kwargs.get("weight_decay_by_lr", True),
                d0=optimizer_kwargs.get("d0", 1e-6),
                d_coef=optimizer_kwargs.get("d_coef", 1.0),
                d_limiter=optimizer_kwargs.get("d_limiter", True),
                prodigy_steps=optimizer_kwargs.get("prodigy_steps", 0),
                schedulefree_c=optimizer_kwargs.get("schedulefree_c", 0),
                eps=optimizer_kwargs.get("eps", 1e-8),
                split_groups=optimizer_kwargs.get("split_groups", True),
                split_groups_mean=optimizer_kwargs.get("split_groups_mean", False),
                factored=optimizer_kwargs.get("factored", True),
                factored_fp32=optimizer_kwargs.get("factored_fp32", True),
                use_bias_correction=optimizer_kwargs.get("use_bias_correction", False),
                use_stableadamw=optimizer_kwargs.get("use_stableadamw", True),
                use_schedulefree=optimizer_kwargs.get("use_schedulefree", True),
                use_speed=optimizer_kwargs.get("use_speed", False),
                stochastic_rounding=optimizer_kwargs.get("stochastic_rounding", True),
                fused_back_pass=optimizer_kwargs.get("fused_back_pass", False),
                use_cautious=optimizer_kwargs.get("use_cautious", False),
                use_grams=optimizer_kwargs.get("use_grams", False),
                use_adopt=optimizer_kwargs.get("use_adopt", False),
                use_orthograd=optimizer_kwargs.get("use_orthograd", False),
                use_focus=optimizer_kwargs.get("use_focus", False),
            )
        except ImportError:
            logger.warning(
                "[Optim] prodigy-plus-schedule-free not installed. "
                "Falling back to AdamW."
            )
            optimizer_type = "adamw"

    # --- Prodigy ---
    if optimizer_type == "prodigy":
        try:
            from prodigyopt import Prodigy

            logger.info(
                "[Optim] Using Prodigy optimizer (adaptive LR -- set LR=1.0 for best results)"
            )

            d_coef = optimizer_kwargs.get("d_coef", 1.0)
            d0 = optimizer_kwargs.get("d0", 1e-6)
            use_bias_correction = optimizer_kwargs.get("use_bias_correction", False)
            safeguard_warmup = optimizer_kwargs.get("safeguard_warmup", False)

            actual_lr = lr if lr != 1e-4 else 1.0

            return Prodigy(
                params,
                lr=actual_lr,
                weight_decay=weight_decay,
                d_coef=d_coef,
                d0=d0,
                use_bias_correction=use_bias_correction,
                safeguard_warmup=safeguard_warmup,
            )
        except ImportError:
            logger.warning(
                "[Optim] prodigyopt not installed. Falling back to AdamW. "
                "Install with: pip install prodigyopt>=1.1.2"
            )
            optimizer_type = "adamw"

    # --- AdEMAMix ---
    if optimizer_type == "ademamix":
        try:
            from acestep.training_v2.ademamix import AdEMAMix

            logger.info("[Optim] Using AdEMAMix optimizer")

            alpha = optimizer_kwargs.get("ademamix_alpha", 5.0)
            beta3 = optimizer_kwargs.get("ademamix_beta3", 0.9)
            t_alpha_beta3 = optimizer_kwargs.get("ademamix_t_alpha_beta3", None)

            return AdEMAMix(
                params,
                lr=lr,
                weight_decay=weight_decay,
                alpha=alpha,
                betas=(0.9, 0.999, beta3),
                t_alpha_beta3=t_alpha_beta3,
                eps=1e-8,
            )
        except ImportError:
            logger.warning(
                "[Optim] acestep.training_v2.ademamix not found. Falling back to AdamW."
            )
            optimizer_type = "adamw"

    # --- AdamW 8-bit ---
    if optimizer_type == "adamw8bit":
        try:
            from bitsandbytes.optim import AdamW8bit

            logger.info("[Optim] Using AdamW8bit optimizer (lower VRAM)")
            return AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            logger.warning(
                "[Optim] bitsandbytes not installed. Falling back to AdamW. "
                "Install with: pip install bitsandbytes>=0.45.0"
            )
            optimizer_type = "adamw"

    # --- Adafactor ---
    if optimizer_type == "adafactor":
        try:
            from transformers.optimization import Adafactor

            logger.info("[Optim] Using Adafactor optimizer (minimal state memory)")

            scale_parameter = optimizer_kwargs.get("scale_parameter", False)
            relative_step = optimizer_kwargs.get("relative_step", False)
            warmup_init = optimizer_kwargs.get("warmup_init", False)

            return Adafactor(
                params,
                lr=lr,
                weight_decay=weight_decay,
                scale_parameter=scale_parameter,
                relative_step=relative_step,
                warmup_init=warmup_init,
            )
        except ImportError:
            logger.warning("[Optim] transformers not installed. Falling back to AdamW.")
            optimizer_type = "adamw"

    # Default: AdamW
    kwargs: Dict[str, Any] = {"lr": lr, "weight_decay": weight_decay}
    if device_type == "cuda":
        kwargs["fused"] = True
    logger.info("[Optim] Using AdamW optimizer")
    return AdamW(params, **kwargs)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    total_steps: int = 1000,
    warmup_steps: int = 500,
    lr: float = 1e-4,
    optimizer_type: str = "adamw",
    n_restarts: int = 4,
):
    """Create a learning rate scheduler from a string key.

    When the optimizer is Prodigy family, defaults to constant schedule
    (Prodigy manages LR internally).
    """
    scheduler_type = scheduler_type.lower().strip()

    # Prodigy family (Plus & Standard) manages LR internally
    if optimizer_type in ("prodigy", "prodigy_plus") and scheduler_type not in (
        "constant",
        "constant_with_warmup",
    ):
        logger.info(
            "[Optim] %s detected -- overriding scheduler to 'constant'",
            optimizer_type,
        )
        scheduler_type = "constant"

    # Clamp warmup to avoid exceeding total
    warmup_steps = min(warmup_steps, max(1, total_steps // 10))

    warmup_sched = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    remaining = max(1, total_steps - warmup_steps)

    if scheduler_type in ("constant", "constant_with_warmup"):
        main_sched = ConstantLR(optimizer, factor=1.0, total_iters=total_steps)
    elif scheduler_type == "linear":
        main_sched = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=remaining,
        )
    elif scheduler_type == "cosine_restarts":
        main_sched = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, remaining // max(1, n_restarts)),
            T_mult=1,
            eta_min=lr * 0.01,
        )
    else:
        # cosine (default) -- single smooth decay to eta_min, no restarts
        main_sched = CosineAnnealingLR(
            optimizer,
            T_max=remaining,
            eta_min=lr * 0.01,
        )

    return SequentialLR(
        optimizer, [warmup_sched, main_sched], milestones=[warmup_steps]
    )
