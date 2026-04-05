"""
Full Fine-tuning Trainer for ACE-Step

Trainer for full fine-tuning of ACE-Step DiT decoder without LoRA adapters.
Supports training from preprocessed tensor files for optimal performance.
"""

import os
import shutil
import time
import random
import math
from typing import Optional, List, Dict, Any, Tuple, Generator
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

try:
    from lightning.fabric import Fabric
    from lightning.fabric.loggers import TensorBoardLogger

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    logger.warning(
        "Lightning Fabric not installed. Training will use basic training loop."
    )

from acestep.training.optim_factory import build_optimizer, build_scheduler
from acestep.training.configs import TrainingConfig
from acestep.training.data_module import PreprocessedDataModule
from acestep.training.path_safety import safe_path

# --- ПРАВИЛЬНЫЙ ИМПОРТ ИЗ V2 (Continuous Timesteps + CFG Dropout) ---
from acestep.training_v2.timestep_sampling import apply_cfg_dropout, sample_timesteps


def _normalize_device_type(device: Any) -> str:
    """Normalize torch device or string to canonical device type."""
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        return device.split(":", 1)[0]
    return str(device)


def _select_compute_dtype(device_type: str) -> torch.dtype:
    """Pick the compute dtype for each accelerator."""
    if device_type in ("cuda", "xpu"):
        return torch.bfloat16
    if device_type == "mps":
        return torch.float16
    return torch.float32


def _select_fabric_precision(device_type: str) -> str:
    """Pick Fabric precision plugin setting for each accelerator."""
    if device_type in ("cuda", "xpu"):
        return "bf16-mixed"
    if device_type == "mps":
        return "16-mixed"
    return "32-true"


def _ensure_trainable_params_fp32(module: nn.Module) -> Tuple[int, int]:
    """Force trainable floating-point parameters to fp32."""
    casted = 0
    total = 0
    for p in module.parameters():
        if not p.requires_grad:
            continue
        total += 1
        if p.is_floating_point() and p.dtype != torch.float32:
            with torch.no_grad():
                p.data = p.data.float()
            casted += 1
    return casted, total


def _count_nonfinite_grads(params: List[torch.nn.Parameter]) -> Tuple[int, int]:
    """Count non-finite gradient tensors among params with gradients."""
    nonfinite = 0
    total_with_grad = 0
    for p in params:
        g = p.grad
        if g is None:
            continue
        total_with_grad += 1
        if not torch.isfinite(g).all():
            nonfinite += 1
    return nonfinite, total_with_grad


def _ensure_optimizer_params_fp32(optimizer: torch.optim.Optimizer) -> Tuple[int, int]:
    """Force optimizer parameter tensors to fp32 when trainable."""
    casted = 0
    total = 0
    for group in optimizer.param_groups:
        for p in group.get("params",[]):
            if p is None:
                continue
            total += 1
            if p.is_floating_point() and p.dtype != torch.float32:
                with torch.no_grad():
                    p.data = p.data.float()
                casted += 1
    return casted, total


class PreprocessedFullFinetuneModule(nn.Module):
    """Full Fine-tuning Module using preprocessed tensors.

    This module trains ONLY the DiT decoder - encoder is not needed
    because all inputs are pre-computed tensors.
    """

    def __init__(
        self,
        model: nn.Module,
        training_config: TrainingConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()

        self.training_config = training_config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.device_type = _normalize_device_type(self.device)
        self.dtype = _select_compute_dtype(self.device_type)
        self.transfer_non_blocking = self.device_type in ("cuda", "xpu")
        
        # Динамически вытягиваем параметры таймстепов из конфига текущей модели (turbo/base/sft)
        self.timestep_mu = getattr(model.config, 'timestep_mu', -0.4)
        self.timestep_sigma = getattr(model.config, 'timestep_sigma', 1.0)
        self.data_proportion = getattr(model.config, 'data_proportion', 0.5)
        
        # Получаем уровень CFG dropout из training_config (по умолчанию 0.15)
        self.cfg_ratio = getattr(training_config, "cfg_ratio", 0.15)
        
        # Получаем эмбеддинг для безусловной генерации
        if hasattr(model, "null_condition_emb"):
            self._null_cond_emb = model.null_condition_emb
        else:
            self._null_cond_emb = None

        self.training_losses =[]

        # OPTIMIZATION: Only register decoder as a submodule to save VRAM.
        self.add_module("decoder", model.decoder)

        if getattr(training_config, "gradient_checkpointing", False):
            try:
                if hasattr(self.decoder, "gradient_checkpointing_enable"):
                    self.decoder.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled for decoder")
                elif hasattr(self.decoder, "gradient_checkpointing"):
                    self.decoder.gradient_checkpointing = True
                    logger.info("Gradient checkpointing enabled for decoder")
            except Exception as e:
                logger.warning(f"Failed to enable gradient checkpointing: {e}")

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        record_loss: bool = True,
    ) -> torch.Tensor:
        if self.device_type in ("cuda", "xpu", "mps"):
            autocast_ctx = torch.autocast(
                device_type=self.device_type, dtype=self.dtype
            )
        else:
            autocast_ctx = nullcontext()
            
        with autocast_ctx:
            target_latents = batch["target_latents"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )  # x0
            attention_mask = batch["attention_mask"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )
            encoder_hidden_states = batch["encoder_hidden_states"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )
            encoder_attention_mask = batch["encoder_attention_mask"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )
            context_latents = batch["context_latents"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )

            bsz = target_latents.shape[0]

            # ---- CFG Dropout (Правильный подход из V2) ----
            if self._null_cond_emb is not None and self.cfg_ratio > 0.0:
                encoder_hidden_states = apply_cfg_dropout(
                    encoder_hidden_states, self._null_cond_emb, cfg_ratio=self.cfg_ratio
                )

            # Flow matching: sample noise x1 and interpolate with data x0
            x1 = torch.randn_like(target_latents)  # Noise
            x0 = target_latents  # Data

            # ---- Continuous timestep sampling ----
            t, r = sample_timesteps(
                batch_size=bsz,
                device=self.device,
                dtype=self.dtype,
                data_proportion=self.data_proportion,
                timestep_mu=self.timestep_mu,
                timestep_sigma=self.timestep_sigma,
                use_meanflow=False,
            )
            t_ = t.unsqueeze(-1).unsqueeze(-1)

            # Interpolate: x_t = t * x1 + (1 - t) * x0
            xt = t_ * x1 + (1.0 - t_) * x0

            # Forward through decoder
            decoder_outputs = self.decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=t,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            )

            # Flow matching loss
            flow = x1 - x0
            diffusion_loss = F.mse_loss(decoder_outputs[0], flow)

        diffusion_loss = diffusion_loss.float()

        if record_loss:
            self.training_losses.append(diffusion_loss.item())

        return diffusion_loss


def _save_complete_model(
    trained_decoder_state: Dict[str, torch.Tensor],
    output_dir: str,
    source_model_dir: str,
    variant: str = "turbo",
) -> str:
    """Save a complete HuggingFace-compatible model checkpoint in BF16."""
    os.makedirs(output_dir, exist_ok=True)

    safetensors_available = False
    try:
        from safetensors.torch import load_file as st_load
        from safetensors.torch import save_file as st_save
        safetensors_available = True
    except ImportError:
        pass

    if source_model_dir and os.path.isdir(source_model_dir) and safetensors_available:
        try:
            logger.info(f"[Save] Merging trained decoder into original model from {source_model_dir}")

            original_path = os.path.join(source_model_dir, "model.safetensors")
            if not os.path.exists(original_path):
                raise FileNotFoundError(f"No model.safetensors found in {source_model_dir}")

            logger.info("[Save] Loading original model safetensors...")
            original_state = st_load(original_path)

            original_tensors: Dict[str, torch.Tensor] = {}
            for k, v in original_state.items():
                if isinstance(v, torch.Tensor):
                    original_tensors[k] = v
            logger.info(f"[Save] Original model has {len(original_state)} keys, {len(original_tensors)} tensors")

            cleaned_state: Dict[str, torch.Tensor] = {}
            for key, value in trained_decoder_state.items():
                clean_key = key
                for prefix in ("decoder.", "module.decoder.", "_forward_module."):
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        break
                cleaned_state[clean_key] = value

            merged_state: Dict[str, torch.Tensor] = {}
            merged_count = 0
            missing_count = 0

            for orig_key in original_tensors.keys():
                orig_val = original_tensors[orig_key]

                match_key = orig_key
                if match_key.startswith("decoder."):
                    match_key = match_key[len("decoder."):]

                if match_key in cleaned_state:
                    trained_tensor = cleaned_state[match_key]
                    if trained_tensor.shape == orig_val.shape:
                        bf16_tensor = trained_tensor.clone().to(dtype=torch.bfloat16) if trained_tensor.is_floating_point() else trained_tensor.clone()
                        merged_state[orig_key] = bf16_tensor
                        merged_count += 1
                    else:
                        logger.warning(
                            f"[Save] Shape mismatch for '{orig_key}': "
                            f"trained={trained_tensor.shape}"
                        )
                        merged_state[orig_key] = orig_val.clone()
                else:
                    merged_state[orig_key] = orig_val.clone()

            missing_count = len(cleaned_state) - merged_count

            logger.info(f"[Save] Merged {merged_count} decoder keys, {missing_count} trained keys not found in original")

            logger.info(f"[Save] Saving merged model to {output_dir}")
            st_save(merged_state, os.path.join(output_dir, "model.safetensors"))
            logger.info(f"[Save] Full model saved to {output_dir} (BF16, single safetensors file)")

        except Exception as exc:
            import traceback
            logger.error(
                f"[Save] Failed to merge model: {exc}\n{traceback.format_exc()}"
            )
            torch.save(
                trained_decoder_state,
                os.path.join(output_dir, "model_state_dict.pt"),
            )
    else:
        logger.warning(
            "[Save] No source model directory or safetensors not available. Saving raw state_dict only."
        )
        torch.save(
            trained_decoder_state,
            os.path.join(output_dir, "model_state_dict.pt"),
        )

    # Copy supporting files
    if source_model_dir and os.path.isdir(source_model_dir):
        _copy_model_support_files(source_model_dir, output_dir, variant)

    return output_dir


def _copy_model_support_files(
    source_dir: str,
    target_dir: str,
    variant: str = "turbo",
) -> None:
    src_silence = os.path.join(source_dir, "silence_latent.pt")
    if os.path.exists(src_silence):
        dst_silence = os.path.join(target_dir, "silence_latent.pt")
        shutil.copy2(src_silence, dst_silence)
        logger.info("[Save] Copied silence_latent.pt")

    code_files =["configuration_acestep_v15.py"]
    if variant == "turbo":
        code_files.append("modeling_acestep_v15_turbo.py")
    elif variant == "base":
        code_files.append("modeling_acestep_v15_base.py")
    elif variant == "sft":
        code_files.append("modeling_acestep_v15_sft.py")

    for filename in code_files:
        src_file = os.path.join(source_dir, filename)
        if not os.path.exists(src_file):
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            src_file = os.path.join(
                project_root, "acestep", "models", variant, filename
            )
        if os.path.exists(src_file):
            dst_file = os.path.join(target_dir, filename)
            shutil.copy2(src_file, dst_file)
            logger.info(f"[Save] Copied {filename}")


class FullFinetuneTrainer:
    """High-level trainer for ACE-Step full fine-tuning.

    Uses Lightning Fabric for distributed training and mixed precision.
    Supports training from preprocessed tensor directories.
    """

    def __init__(
        self,
        dit_handler,
        training_config: TrainingConfig,
        source_model_dir: Optional[str] = None,
    ):
        self.dit_handler = dit_handler
        param = next(self.dit_handler.parameters())
        self.device = param.device
        self.dtype = param.dtype
        training_config.output_dir = safe_path(training_config.output_dir)
        self.training_config = training_config

        self.module = None
        self.fabric = None
        self.is_training = False

        self._source_model_dir: Optional[str] = None
        if source_model_dir and os.path.isdir(source_model_dir):
            self._source_model_dir = os.path.abspath(source_model_dir)
        else:
            try:
                full_model = getattr(dit_handler, "model", None)
                if full_model is not None and hasattr(full_model, "config"):
                    name_or_path = getattr(full_model.config, "_name_or_path", None)
                    if name_or_path and os.path.isdir(name_or_path):
                        self._source_model_dir = os.path.abspath(name_or_path)
            except Exception:
                pass

    def train_from_preprocessed(
        self,
        tensor_dir: str,
        training_state: Optional[Dict] = None,
        resume_from: Optional[str] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        self.is_training = True

        try:
            try:
                tensor_dir = safe_path(tensor_dir)
            except ValueError:
                yield 0, 0.0, f"❌ Rejected unsafe tensor directory: {tensor_dir}"
                return
            if not os.path.isdir(tensor_dir):
                yield 0, 0.0, f"❌ Tensor directory not found: {tensor_dir}"
                return

            torch.manual_seed(self.training_config.seed)
            random.seed(self.training_config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.training_config.seed)
            try:
                import numpy as np

                np.random.seed(self.training_config.seed)
            except Exception:
                pass

            self.module = PreprocessedFullFinetuneModule(
                model=self.dit_handler,
                training_config=self.training_config,
                device=self.device,
                dtype=self.dtype,
            )

            # Unfreeze decoder parameters for full fine-tuning
            for param in self.module.decoder.parameters():
                param.requires_grad = True

            trainable_count = sum(
                1 for p in self.module.parameters() if p.requires_grad
            )
            logger.info(
                f"Unfroze {trainable_count} trainable parameters for full fine-tuning"
            )

            del self.dit_handler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            data_module = PreprocessedDataModule(
                tensor_dir=tensor_dir,
                batch_size=self.training_config.batch_size,
                num_workers=self.training_config.num_workers,
                pin_memory=self.training_config.pin_memory,
                prefetch_factor=self.training_config.prefetch_factor,
                persistent_workers=self.training_config.persistent_workers,
                pin_memory_device=self.training_config.pin_memory_device,
                val_split=getattr(self.training_config, "val_split", 0.0),
            )

            data_module.setup("fit")

            if len(data_module.train_dataset) == 0:
                yield 0, 0.0, "❌ No valid samples found in tensor directory"
                return

            yield (
                0,
                0.0,
                f"📂 Loaded {len(data_module.train_dataset)} preprocessed samples",
            )

            if LIGHTNING_AVAILABLE:
                yield from self._train_with_fabric(
                    data_module, training_state, resume_from
                )
            else:
                yield from self._train_basic(data_module, training_state)

        except Exception as e:
            logger.exception("Training failed")
            yield 0, 0.0, f"❌ Training failed: {str(e)}"
        finally:
            self.is_training = False

    def _train_with_fabric(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict],
        resume_from: Optional[str] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        os.makedirs(self.training_config.output_dir, exist_ok=True)

        device_type = self.module.device_type
        precision = _select_fabric_precision(device_type)
        accelerator = (
            device_type if device_type in ("cuda", "xpu", "mps", "cpu") else "auto"
        )

        tb_logger = None
        try:
            tb_logger = TensorBoardLogger(
                root_dir=self.training_config.output_dir, name="logs"
            )
        except ModuleNotFoundError as e:
            logger.warning(
                f"TensorBoard logger unavailable, continuing without logger: {e}"
            )

        fabric_kwargs = {
            "accelerator": accelerator,
            "devices": 1,
            "precision": precision,
        }
        if tb_logger is not None:
            fabric_kwargs["loggers"] = [tb_logger]
        self.fabric = Fabric(**fabric_kwargs)
        self.fabric.launch()

        yield (
            0,
            0.0,
            f"🚀 Starting full fine-tuning (device: {device_type}, precision: {precision})...",
        )

        if device_type == "mps":
            self.module.decoder = self.module.decoder.to(dtype=torch.float32)
            casted_trainable, total_trainable_tensors = _ensure_trainable_params_fp32(
                self.module.decoder
            )
            logger.info(
                f"Trainable tensor dtype fixup: casted {casted_trainable}/{total_trainable_tensors} to fp32"
            )
        else:
            logger.info("CUDA/XPU: keeping decoder weights in bf16 for VRAM efficiency")

        train_loader = data_module.train_dataloader()
        val_loader = (
            data_module.val_dataloader()
            if hasattr(data_module, "val_dataloader")
            else None
        )

        if training_state is not None:
            training_state["plot_steps"] = []
            training_state["plot_loss"] = []
            training_state["plot_ema"] = []
            training_state["plot_val_steps"] = []
            training_state["plot_val_loss"] =[]
            training_state["plot_best_step"] = None
        ema_loss = None
        ema_alpha = 0.1
        best_val_loss = float("inf")
        best_val_step = None

        trainable_params =[p for p in self.module.parameters() if p.requires_grad]

        if not trainable_params:
            yield 0, 0.0, "❌ No trainable parameters found!"
            return

        yield (
            0,
            0.0,
            f"🎯 Training {sum(p.numel() for p in trainable_params):,} parameters",
        )

        steps_per_epoch = max(
            1,
            math.ceil(
                len(train_loader) / self.training_config.gradient_accumulation_steps
            ),
        )
        total_steps = steps_per_epoch * self.training_config.max_epochs
        warmup_steps = min(self.training_config.warmup_steps, max(1, total_steps // 10))

        optimizer = build_optimizer(
            params=trainable_params,
            optimizer_type=self.training_config.optimizer_type,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            device_type=device_type,
            optimizer_kwargs=getattr(self.training_config, "optimizer_kwargs", {}),
        )

        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_type=getattr(self.training_config, "scheduler_type", "cosine"),
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            lr=self.training_config.learning_rate,
            optimizer_type=self.training_config.optimizer_type,
        )

        self.module, optimizer = self.fabric.setup(self.module, optimizer)
        if device_type == "mps":
            casted_opt_params, total_opt_params = _ensure_optimizer_params_fp32(
                optimizer
            )
            logger.info(
                f"Optimizer param dtype fixup: casted {casted_opt_params}/{total_opt_params} to fp32"
            )
        train_loader = self.fabric.setup_dataloaders(train_loader)

        start_epoch = 0
        global_step = 0
        checkpoint_info = None

        if resume_from:
            try:
                resume_from = safe_path(resume_from)
            except ValueError:
                yield (
                    0,
                    0.0,
                    f"⚠️ Rejected unsafe checkpoint path: {resume_from}, starting fresh",
                )
                resume_from = None
        if resume_from and os.path.exists(resume_from):
            try:
                yield 0, 0.0, f"🔄 Loading checkpoint from {resume_from}..."

                checkpoint_path = os.path.join(resume_from, "model_state_dict.pt")
                if os.path.exists(checkpoint_path):
                    model_state_dict = torch.load(
                        checkpoint_path,
                        map_location=self.module.device,
                        weights_only=True,
                    )
                    self.module.load_state_dict(model_state_dict)
                    start_epoch = 0
                    global_step = 0
                    yield 0, 0.0, "✅ Resumed from checkpoint"
                else:
                    yield (
                        0,
                        0.0,
                        f"⚠️ Checkpoint not found in {resume_from}, starting fresh",
                    )

            except Exception as e:
                logger.exception("Failed to load checkpoint")
                yield 0, 0.0, f"⚠️ Failed to load checkpoint: {e}, starting fresh"
                start_epoch = 0
                global_step = 0
        elif resume_from:
            yield 0, 0.0, f"⚠️ Checkpoint path not found: {resume_from}, starting fresh"

        accumulation_step = 0
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        self.module.train()

        for epoch in range(self.training_config.max_epochs):
            epoch_loss = 0.0
            num_updates = 0
            epoch_start_time = time.time()

            for _batch_idx, batch in enumerate(train_loader):
                if training_state and training_state.get("should_stop", False):
                    yield (
                        global_step,
                        accumulated_loss / max(accumulation_step, 1),
                        "⏹️ Training stopped by user",
                    )
                    return

                loss = self.module.training_step(batch)
                loss = loss / self.training_config.gradient_accumulation_steps

                self.fabric.backward(loss)
                accumulated_loss += loss.item()
                accumulation_step += 1

                if (
                    accumulation_step
                    >= self.training_config.gradient_accumulation_steps
                ):
                    nonfinite_grads, grad_tensors = _count_nonfinite_grads(
                        trainable_params
                    )
                    if nonfinite_grads > 0:
                        optimizer.zero_grad(set_to_none=True)
                        yield (
                            global_step,
                            float("nan"),
                            (
                                f"⚠️ Non-finite gradients ({nonfinite_grads}/{grad_tensors}); "
                                "skipping optimizer step"
                            ),
                        )
                        accumulated_loss = 0.0
                        accumulation_step = 0
                        continue

                    self.fabric.clip_gradients(
                        self.module,
                        optimizer,
                        max_norm=self.training_config.max_grad_norm,
                        error_if_nonfinite=False,
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1

                    avg_loss = accumulated_loss / accumulation_step
                    if global_step % self.training_config.log_every_n_steps == 0:
                        if training_state is not None:
                            if ema_loss is None:
                                ema_loss = avg_loss
                            else:
                                ema_loss = (
                                    ema_alpha * avg_loss + (1 - ema_alpha) * ema_loss
                                )
                            training_state["plot_steps"].append(global_step)
                            training_state["plot_loss"].append(avg_loss)
                            training_state["plot_ema"].append(ema_loss)
                        self.fabric.log("train/loss", avg_loss, step=global_step)
                        self.fabric.log(
                            "train/lr", scheduler.get_last_lr()[0], step=global_step
                        )
                    yield (
                        global_step,
                        avg_loss,
                        f"Epoch {epoch + 1}/{self.training_config.max_epochs}, Step {global_step}, Loss: {avg_loss:.4f}",
                    )

                    epoch_loss += avg_loss
                    num_updates += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

            if accumulation_step > 0:
                nonfinite_grads, grad_tensors = _count_nonfinite_grads(trainable_params)
                if nonfinite_grads > 0:
                    optimizer.zero_grad(set_to_none=True)
                    yield (
                        global_step,
                        float("nan"),
                        (
                            f"⚠️ Non-finite gradients ({nonfinite_grads}/{grad_tensors}); "
                            "skipping optimizer remainder step"
                        ),
                    )
                    accumulated_loss = 0.0
                    accumulation_step = 0
                else:
                    self.fabric.clip_gradients(
                        self.module,
                        optimizer,
                        max_norm=self.training_config.max_grad_norm,
                        error_if_nonfinite=False,
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1
                avg_loss = accumulated_loss / accumulation_step
                if global_step % self.training_config.log_every_n_steps == 0:
                    if training_state is not None:
                        if ema_loss is None:
                            ema_loss = avg_loss
                        else:
                            ema_loss = ema_alpha * avg_loss + (1 - ema_alpha) * ema_loss
                        training_state["plot_steps"].append(global_step)
                        training_state["plot_loss"].append(avg_loss)
                        training_state["plot_ema"].append(ema_loss)
                    self.fabric.log("train/loss", avg_loss, step=global_step)
                    self.fabric.log(
                        "train/lr", scheduler.get_last_lr()[0], step=global_step
                    )
                    yield (
                        global_step,
                        avg_loss,
                        f"Epoch {epoch + 1}/{self.training_config.max_epochs}, Step {global_step}, Loss: {avg_loss:.4f}",
                    )

                    epoch_loss += avg_loss
                    num_updates += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(num_updates, 1)
            if training_state is not None:
                if ema_loss is None:
                    ema_loss = avg_epoch_loss
                else:
                    ema_loss = ema_alpha * avg_epoch_loss + (1 - ema_alpha) * ema_loss
                plot_steps = training_state["plot_steps"]
                if not plot_steps or plot_steps[-1] != global_step:
                    training_state["plot_steps"].append(global_step)
                    training_state["plot_loss"].append(avg_epoch_loss)
                    training_state["plot_ema"].append(ema_loss)
            self.fabric.log("train/epoch_loss", avg_epoch_loss, step=epoch + 1)

            if val_loader is not None:
                self.module.eval()
                total_val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        v_loss = self.module.training_step(val_batch, record_loss=False)
                        total_val_loss += v_loss.item()
                        n_val += 1
                self.module.train()
                val_loss = total_val_loss / max(n_val, 1)
                if training_state is not None:
                    training_state["plot_val_steps"].append(global_step)
                    training_state["plot_val_loss"].append(val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_step = global_step
                    if training_state is not None:
                        training_state["plot_best_step"] = best_val_step
                best_dir = os.path.join(
                    self.training_config.output_dir, "checkpoints", "best"
                )
                os.makedirs(best_dir, exist_ok=True)
                _save_complete_model(
                    self.module.state_dict(),
                    best_dir,
                    self._source_model_dir,
                )

            if (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                checkpoint_dir = os.path.join(
                    self.training_config.output_dir,
                    "checkpoints",
                    f"epoch_{epoch + 1}_loss_{avg_epoch_loss:.4f}",
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                _save_complete_model(
                    self.module.state_dict(),
                    checkpoint_dir,
                    self._source_model_dir,
                )
                yield (
                    global_step,
                    avg_epoch_loss,
                    f"💾 Checkpoint saved at epoch {epoch + 1}",
                )

        final_path = os.path.join(self.training_config.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        _save_complete_model(
            self.module.state_dict(),
            final_path,
            self._source_model_dir,
        )

        final_loss = (
            self.module.training_losses[-1] if self.module.training_losses else 0.0
        )
        yield (
            global_step,
            final_loss,
            f"✅ Full fine-tuning complete! Model saved to {final_path}",
        )

    def _train_basic(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict],
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Basic training loop without Fabric."""
        yield 0, 0.0, "🚀 Starting basic full fine-tuning loop..."

        os.makedirs(self.training_config.output_dir, exist_ok=True)

        train_loader = data_module.train_dataloader()

        trainable_params =[p for p in self.module.parameters() if p.requires_grad]

        if not trainable_params:
            yield 0, 0.0, "❌ No trainable parameters found!"
            return

        steps_per_epoch = max(
            1,
            math.ceil(
                len(train_loader) / self.training_config.gradient_accumulation_steps
            ),
        )
        total_steps = steps_per_epoch * self.training_config.max_epochs
        warmup_steps = min(self.training_config.warmup_steps, max(1, total_steps // 10))

        optimizer = build_optimizer(
            params=trainable_params,
            optimizer_type=self.training_config.optimizer_type,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            device_type=self.module.device_type,
            optimizer_kwargs=getattr(self.training_config, "optimizer_kwargs", {}),
        )

        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_type=getattr(self.training_config, "scheduler_type", "cosine"),
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            lr=self.training_config.learning_rate,
            optimizer_type=self.training_config.optimizer_type,
        )

        global_step = 0
        accumulation_step = 0
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        self.module.train()

        for epoch in range(self.training_config.max_epochs):
            epoch_loss = 0.0
            num_updates = 0
            epoch_start_time = time.time()

            for batch in train_loader:
                if training_state and training_state.get("should_stop", False):
                    yield (
                        global_step,
                        accumulated_loss / max(accumulation_step, 1),
                        "⏹️ Training stopped",
                    )
                    return

                loss = self.module.training_step(batch)
                loss = loss / self.training_config.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                accumulation_step += 1

                if (
                    accumulation_step
                    >= self.training_config.gradient_accumulation_steps
                ):
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, self.training_config.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    avg_loss = accumulated_loss / accumulation_step
                    if global_step % self.training_config.log_every_n_steps == 0:
                        yield (
                            global_step,
                            avg_loss,
                            f"Epoch {epoch + 1}, Step {global_step}, Loss: {avg_loss:.4f}",
                        )

                    epoch_loss += avg_loss
                    num_updates += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

            if accumulation_step > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, self.training_config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = accumulated_loss / accumulation_step
                if global_step % self.training_config.log_every_n_steps == 0:
                    yield (
                        global_step,
                        avg_loss,
                        f"Epoch {epoch + 1}, Step {global_step}, Loss: {avg_loss:.4f}",
                    )

                epoch_loss += avg_loss
                num_updates += 1
                accumulated_loss = 0.0
                accumulation_step = 0

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(num_updates, 1)
            yield (
                global_step,
                avg_epoch_loss,
                f"✅ Epoch {epoch + 1}/{self.training_config.max_epochs} in {epoch_time:.1f}s",
            )

            if (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                checkpoint_dir = os.path.join(
                    self.training_config.output_dir,
                    "checkpoints",
                    f"epoch_{epoch + 1}_loss_{avg_epoch_loss:.4f}",
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                _save_complete_model(
                    self.module.state_dict(),
                    checkpoint_dir,
                    self._source_model_dir,
                )
                yield global_step, avg_epoch_loss, "💾 Checkpoint saved"

        final_path = os.path.join(self.training_config.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        _save_complete_model(
            self.module.state_dict(),
            final_path,
            self._source_model_dir,
        )
        final_loss = (
            self.module.training_losses[-1] if self.module.training_losses else 0.0
        )
        yield (
            global_step,
            final_loss,
            f"✅ Full fine-tuning complete! Model saved to {final_path}",
        )

    def stop(self):
        """Stop training."""
        self.is_training = False