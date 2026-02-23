"""
FixedLoRATrainer -- Orchestration for ACE-Step V2 adapter fine-tuning.

The actual per-step training logic lives in ``fixed_lora_module.py``
(``FixedLoRAModule``).  The non-Fabric fallback loop lives in
``trainer_basic_loop.py``.  Checkpoint, memory, and verification helpers
live in ``trainer_helpers.py``.

Supports both adapter types:
    - **LoRA** via PEFT (``inject_lora_into_dit``)
    - **LoKR** via LyCORIS (``inject_lokr_into_dit``)

Uses shared utilities from ``acestep.training``.
"""

from __future__ import annotations

import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import torch
import torch.nn as nn
from acestep.training_v2.optim import build_optimizer, build_scheduler
from acestep.training.data_module import PreprocessedDataModule

# V2 modules
from acestep.training_v2.configs import TrainingConfigV2
from acestep.training_v2.tensorboard_utils import TrainingLogger
from acestep.training_v2.ui import TrainingUpdate

import gc
from acestep.training_v2.model_loader import load_vae, load_text_encoder, unload_models
import torchaudio

# Split-out modules
from acestep.training_v2.fixed_lora_module import (
    AdapterConfig,
    FixedLoRAModule,
    _normalize_device_type,
    _select_compute_dtype,
    _select_fabric_precision,
)
from acestep.training_v2.trainer_helpers import (
    configure_memory_features,
    offload_non_decoder,
    resume_checkpoint,
    save_adapter_flat,
    save_checkpoint,
    save_final,
    verify_saved_adapter,
)
from acestep.training_v2.trainer_basic_loop import run_basic_training_loop

logger = logging.getLogger(__name__)

# Try to import Lightning Fabric
try:
    from lightning.fabric import Fabric

    _FABRIC_AVAILABLE = True
except ImportError:
    _FABRIC_AVAILABLE = False
    logger.warning("[WARN] Lightning Fabric not installed. Training will use basic loop.")


# ===========================================================================
# FixedLoRATrainer -- orchestration
# ===========================================================================

class FixedLoRATrainer:
    """High-level trainer for corrected ACE-Step adapter fine-tuning.

    Supports both LoRA (PEFT) and LoKR (LyCORIS) adapters.
    Uses Lightning Fabric for mixed precision and gradient scaling.
    Falls back to a basic PyTorch loop when Fabric is not installed.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter_config: AdapterConfig,
        training_config: TrainingConfigV2,
    ) -> None:
        self.model = model
        self.adapter_config = adapter_config
        self.training_config = training_config
        self.adapter_type = training_config.adapter_type

        # Backward-compat alias
        self.lora_config = adapter_config

        self.module: Optional[FixedLoRAModule] = None
        self.fabric: Optional[Any] = None
        self.is_training = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        training_state: Optional[Dict[str, Any]] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Run the full training loop.

        Yields ``(global_step, loss, status_message)`` tuples.
        """
        self.is_training = True
        cfg = self.training_config

        try:
            # -- Validate ---------------------------------------------------
            ds_dir = Path(cfg.dataset_dir)
            if not ds_dir.is_dir():
                yield TrainingUpdate(0, 0.0, f"[FAIL] Dataset directory not found: {ds_dir}", kind="fail")
                return

            # -- Seed -------------------------------------------------------
            torch.manual_seed(cfg.seed)
            random.seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)

            # -- Build module -----------------------------------------------
            device = torch.device(cfg.device)
            dtype = _select_compute_dtype(_normalize_device_type(device))

            self.module = FixedLoRAModule(
                model=self.model,
                adapter_config=self.adapter_config,
                training_config=cfg,
                device=device,
                dtype=dtype,
            )

            # -- Data -------------------------------------------------------
            # Windows uses spawn for multiprocessing; default to 0 workers there
            num_workers = cfg.num_workers
            if sys.platform == "win32" and num_workers > 0:
                logger.info("[Side-Step] Windows detected -- setting num_workers=0 (spawn incompatible)")
                num_workers = 0

            data_module = PreprocessedDataModule(
                tensor_dir=cfg.dataset_dir,
                batch_size=cfg.batch_size,
                num_workers=num_workers,
                pin_memory=cfg.pin_memory,
                prefetch_factor=cfg.prefetch_factor if num_workers > 0 else None,
                persistent_workers=cfg.persistent_workers if num_workers > 0 else False,
                pin_memory_device=cfg.pin_memory_device,
            )
            data_module.setup("fit")

            if len(data_module.train_dataset) == 0:
                yield TrainingUpdate(0, 0.0, "[FAIL] No valid samples found in dataset directory", kind="fail")
                return

            yield TrainingUpdate(0, 0.0, f"[OK] Loaded {len(data_module.train_dataset)} preprocessed samples", kind="info")

            # -- Dispatch to Fabric or basic loop ---------------------------
            if _FABRIC_AVAILABLE:
                yield from self._train_fabric(data_module, training_state)
            else:
                yield from run_basic_training_loop(self, data_module, training_state)

        except Exception as exc:
            logger.exception("Training failed")
            yield TrainingUpdate(0, 0.0, f"[FAIL] Training failed: {exc}", kind="fail")
        finally:
            self.is_training = False

    def stop(self) -> None:
        self.is_training = False

    # ------------------------------------------------------------------
    # Delegate helpers (thin wrappers around trainer_helpers functions)
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_module_wrappers(module: nn.Module) -> list:
        from acestep.training_v2.trainer_helpers import iter_module_wrappers
        return iter_module_wrappers(module)

    @classmethod
    def _configure_memory_features(cls, decoder: nn.Module) -> tuple:
        return configure_memory_features(decoder)

    @staticmethod
    def _offload_non_decoder(model: nn.Module) -> int:
        return offload_non_decoder(model)

    def _save_adapter_flat(self, output_dir: str) -> None:
        save_adapter_flat(self, output_dir)

    def _save_checkpoint(
        self, optimizer: Any, scheduler: Any, epoch: int, global_step: int, ckpt_dir: str,
    ) -> None:
        save_checkpoint(self, optimizer, scheduler, epoch, global_step, ckpt_dir)

    def _save_final(self, output_dir: str) -> None:
        save_final(self, output_dir)

    @staticmethod
    def _verify_saved_adapter(output_dir: str) -> None:
        verify_saved_adapter(output_dir)

    def _resume_checkpoint(
        self, resume_path: str, optimizer: Any, scheduler: Any,
    ) -> Generator[TrainingUpdate, None, Optional[Tuple[int, int]]]:
        return (yield from resume_checkpoint(self, resume_path, optimizer, scheduler))

    # ------------------------------------------------------------------
    # Fabric training loop
    # ------------------------------------------------------------------

    def _generate_preview(self, output_dir: Path, step: int, device: torch.device):
        """
        Generates a preview using deterministic sample selection and optional VRAM offloading.
        Saves as MP3.
        """
        if not self.module: return
        cfg = self.training_config
        
        # Determine strict offloading
        do_offload_dit = cfg.offload_dit_for_preview
        print(f"[LOG] 🎵 Generating preview (Index: {cfg.preview_sample_index}, Offload DiT: {do_offload_dit})...")
        
        try:
            orig_decoder_dev = next(self.module.model.decoder.parameters()).device
        except:
            orig_decoder_dev = device

        try:
            import math
            import torchaudio
            from acestep.training_v2.model_loader import (
                load_vae, 
                load_text_encoder, 
                load_silence_latent, 
                unload_models
            )
            # Mixins for tiled decoding
            from acestep.core.generation.handler.vae_decode import VaeDecodeMixin
            from acestep.core.generation.handler.vae_decode_chunks import VaeDecodeChunksMixin
            from acestep.core.generation.handler.memory_utils import MemoryUtilsMixin
            
            model_variant = cfg.model_variant.lower()
            is_turbo = "turbo" in model_variant
            
            shift = cfg.shift if (cfg.shift is not None and cfg.shift > 0) else (3.0 if is_turbo else 1.0)
            steps = cfg.num_inference_steps if (cfg.num_inference_steps and cfg.num_inference_steps > 0) else (8 if is_turbo else 50)
            guidance_scale = 7.5

            # --- DETERMINISTIC SAMPLE SELECTION ---
            ds_dir = Path(cfg.dataset_dir)
            pt_files = sorted(list(ds_dir.glob("*.pt"))) # Sort for deterministic order
            pt_files = [f for f in pt_files if f.name != "manifest.json"]
            
            # Default values (if file not found or empty)
            caption, lyrics, tag, pos = "Music", "[Instrumental]", "", "prepend"
            ds_bpm, ds_key, ds_ts = "N/A", "N/A", "N/A" # Значения из датасета по умолчанию

            if pt_files:
                # Use modulus to ensure index is always valid
                target_idx = cfg.preview_sample_index % len(pt_files)
                chosen_file = pt_files[target_idx]
                
                try:
                    sample_data = torch.load(chosen_file, map_location="cpu", weights_only=False)
                    meta = sample_data.get("metadata", {})
                    
                    # Extract base fields
                    caption = meta.get("caption", "Music")
                    lyrics = meta.get("lyrics", "[Instrumental]")
                    tag = meta.get("custom_tag", "")
                    pos = meta.get("tag_position", "prepend")
                    
                    # Extract technical fields (Goal 1)
                    ds_bpm = meta.get("bpm", "N/A")
                    ds_key = meta.get("keyscale", "N/A")
                    ds_ts = meta.get("timesignature", "N/A")
                    
                    print(f"[LOG] 📄 Using sample [{target_idx}]: {chosen_file.name}")
                except Exception as e:
                    print(f"[LOG] ⚠️ Failed to load sample {chosen_file.name}: {e}")

            # --- APPLY OVERRIDES (Goal 2) ---
            # Logic: If config value exists (UI filled it), use it. Else use dataset value.
            
            final_caption = cfg.preview_caption if (cfg.preview_caption and cfg.preview_caption.strip()) else caption
            final_lyrics = cfg.preview_lyrics if (cfg.preview_lyrics and cfg.preview_lyrics.strip()) else lyrics
            
            # Handle tag insertion only if we are using the dataset caption (not overriding)
            if (not cfg.preview_caption or not cfg.preview_caption.strip()) and tag:
                if pos == "prepend": test_prompt_raw = f"{tag}, {final_caption}" if final_caption else tag
                elif pos == "append": test_prompt_raw = f"{final_caption}, {tag}" if final_caption else tag
                elif pos == "replace": test_prompt_raw = tag
                else: test_prompt_raw = final_caption
            else:
                test_prompt_raw = final_caption

            final_bpm = cfg.preview_bpm if (cfg.preview_bpm and cfg.preview_bpm.strip()) else str(ds_bpm)
            final_key = cfg.preview_keyscale if (cfg.preview_keyscale and cfg.preview_keyscale.strip()) else str(ds_key)
            final_ts = cfg.preview_timesig if (cfg.preview_timesig and cfg.preview_timesig.strip()) else str(ds_ts)

            # --- PROMPT FORMATTING ---
            default_instruction = "Fill the audio semantic mask based on the given conditions:"
            
            # Dynamic meta string using extracted/overridden values
            meta_str = (
                f"- bpm: {final_bpm}\n"
                f"- timesignature: {final_ts}\n"
                f"- keyscale: {final_key}\n"
                f"- duration: 30 seconds\n"
            )
            
            full_text_prompt = (
                f"# Instruction\n{default_instruction}\n\n"
                f"# Caption\n{test_prompt_raw}\n\n"
                f"# Metas\n{meta_str}<|endoftext|>"
            )
            
            lang = "unknown" # Можно тоже вынести в конфиг если нужно
            full_lyrics_prompt = f"# Languages\n{lang}\n\n# Lyric\n{final_lyrics}<|endoftext|>"

            # print(f"[LOG] full_text_prompt: {full_text_prompt}")
            # print(f"[LOG] full_lyrics_prompt: {full_lyrics_prompt}")

            # --- ENCODING ---
            # If offload_dit is ON, ensure DiT is OFF GPU before loading Text Encoder
            if do_offload_dit:
                self.module.model.decoder.to('cpu')
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            tokenizer, text_enc = load_text_encoder(cfg.checkpoint_dir, device=device, precision=cfg.precision)
            silence_lat = load_silence_latent(
                cfg.checkpoint_dir, device=device, precision=cfg.precision, variant=cfg.model_variant
            )
            dtype = self.module.dtype 

            text_inputs = tokenizer(full_text_prompt, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
            text_hs = text_enc(text_inputs.input_ids.to(device)).last_hidden_state.to(dtype)
            text_mask = text_inputs.attention_mask.to(device).to(dtype)
            
            lyric_inputs = tokenizer(full_lyrics_prompt, padding="max_length", max_length=2048, truncation=True, return_tensors="pt")
            lyric_hs = text_enc.embed_tokens(lyric_inputs.input_ids.to(device)).to(dtype)
            lyric_mask = lyric_inputs.attention_mask.to(device).to(dtype)

            unload_models(text_enc) 

            # --- DiT GENERATION ---
            # Move DiT to GPU
            self.module.model.to(device)
            self.module.model.eval()

            batch_size, seq_len = 1, 512
            
            refer_audio = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
            refer_mask = torch.zeros(batch_size, device=device, dtype=torch.long)
            
            if silence_lat.dim() == 2: silence_lat = silence_lat.unsqueeze(0)
            if silence_lat.shape[1] < seq_len:
                padding = torch.zeros(batch_size, seq_len - silence_lat.shape[1], silence_lat.shape[2], device=device, dtype=dtype)
                current_silence = torch.cat([silence_lat, padding], dim=1)
            else:
                current_silence = silence_lat[:, :seq_len, :]
            
            current_silence = current_silence.expand(batch_size, -1, -1).to(device, dtype)
            src_latents = current_silence.clone()
            chunk_masks = torch.ones(batch_size, seq_len, 64, device=device, dtype=dtype)
            is_covers = torch.zeros(batch_size, device=device, dtype=torch.bool)
            attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=dtype)

            try:
                outputs = self.module.model.generate_audio(
                    text_hidden_states=text_hs,
                    text_attention_mask=text_mask,
                    lyric_hidden_states=lyric_hs,
                    lyric_attention_mask=lyric_mask,
                    refer_audio_acoustic_hidden_states_packed=refer_audio,
                    refer_audio_order_mask=refer_mask,
                    src_latents=src_latents,
                    chunk_masks=chunk_masks,
                    silence_latent=current_silence,
                    attention_mask=attention_mask,
                    is_covers=is_covers,
                    infer_steps=steps,
                    diffusion_guidance_sale=guidance_scale,
                    shift=shift,
                    use_cache=True,
                    infer_method="ode",
                    use_progress_bar=False
                )
                generated_latents = outputs["target_latents"]
                
            except Exception as gen_err:
                print(f"[LOG] ❌ Model.generate_audio failed: {gen_err}")
                raise gen_err

            # --- VAE DECODING ---
            # If offload_dit is ON, move DiT to CPU now to make room for VAE
            if do_offload_dit:
                print("[LOG] 🧹 Offloading DiT to CPU for VAE Decode...")
                # FIXED: This moves the WHOLE model (including null_condition_emb) to CPU
                self.module.model.to('cpu')
                if torch.cuda.is_available(): 
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            # Load VAE to GPU
            vae = load_vae(cfg.checkpoint_dir, device=device, precision=cfg.precision)

            # Wrapper for tiled decoding
            class VaeInferenceWrapper(VaeDecodeMixin, VaeDecodeChunksMixin, MemoryUtilsMixin):
                def __init__(self, vae_model, device_name):
                    self.vae = vae_model
                    self.device = device_name
                    self.use_mlx_vae = False
                    self.disable_tqdm = True 
                    self.offload_to_cpu = False 
                
                def _recursive_to_device(self, module, device, dtype=None):
                    module.to(device)
                    if dtype: module.to(dtype)
                
                def _get_auto_decode_chunk_size(self):
                    return 256 
                
                def _should_offload_wav_to_cpu(self):
                    return False

            vae_wrapper = VaeInferenceWrapper(vae, str(device))
            
            latents_to_decode = generated_latents.to(device).to(vae.dtype).transpose(1, 2)
            
            with torch.no_grad():
                audio = vae_wrapper.tiled_decode(
                    latents_to_decode, 
                    chunk_size=256, 
                    overlap=32      
                )
            
            # Save as MP3
            preview_dir = output_dir / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            # CHANGE: Extension to mp3
            save_path = preview_dir / f"sample_epoch_{step}_idx{target_idx}.mp3"
            
            audio_data = audio[0].detach().cpu().float()
            
            # Normalize
            max_val = audio_data.abs().max()
            if max_val > 0:
                audio_data = audio_data / max_val * 0.95
            
            # CHANGE: Use format="mp3". Requires ffmpeg backend usually.
            try:
                torchaudio.save(str(save_path), audio_data, 48000, format="mp3")
                print(f"[LOG] 💾 Saved preview to: {save_path}")
            except Exception as e:
                 print(f"[LOG] ⚠️ Failed to save MP3 (ffmpeg missing?), fallback to WAV. Error: {e}")
                 save_path_wav = save_path.with_suffix(".wav")
                 torchaudio.save(str(save_path_wav), audio_data, 48000)
                 print(f"[LOG] 💾 Saved preview to: {save_path_wav}")

        except Exception as e:
            print(f"[LOG] ❌ Preview generation failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try: unload_models(vae)
            except: pass
            
            # --- RESTORE ---
            # FIXED: Move the ENTIRE model back to the original device.
            # This ensures 'null_condition_emb' (which lives on self.module.model, not decoder)
            # is returned to GPU after offload_dit moved it to CPU.
            # Previously: self.module.model.decoder.to(orig_decoder_dev) -> Caused null_emb to stay on CPU
            self.module.model.to(orig_decoder_dev)
            
            # Handle Encoder restoration
            if cfg.offload_encoder:
                if hasattr(self.module.model, "encoder"):
                    self.module.model.encoder.to("cpu")
            else:
                if hasattr(self.module.model, "encoder"):
                    self.module.model.encoder.to(orig_decoder_dev)

            self.module.model.decoder.train()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _train_fabric(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict[str, Any]],
    ) -> Generator[TrainingUpdate, None, None]:
        cfg = self.training_config
        assert self.module is not None

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        device_type = self.module.device_type
        precision = _select_fabric_precision(device_type)
        accelerator = device_type if device_type in ("cuda", "xpu", "mps", "cpu") else "auto"

        # -- Fabric init ----------------------------------------------------
        # Always use devices=1 (integer).  Passing devices=[index] (a list)
        # causes Fabric on Windows to create a DistributedSampler wrapper
        # that yields 0 batches, silently breaking the training loop.
        # Instead, we set the default CUDA device so Fabric's single-device
        # mode picks up the correct GPU.
        if device_type == "cuda":
            device_idx = self.module.device.index or 0
            torch.cuda.set_device(device_idx)

        self.fabric = Fabric(
            accelerator=accelerator,
            devices=1,
            precision=precision,
        )
        self.fabric.launch()

        yield TrainingUpdate(0, 0.0, f"[INFO] Starting training (device: {device_type}, precision: {precision})", kind="info")

        # -- TensorBoard logger ---------------------------------------------
        tb = TrainingLogger(cfg.effective_log_dir)

        # -- Dataloader -----------------------------------------------------
        train_loader = data_module.train_dataloader()

        # -- Trainable params / optimizer -----------------------------------
        trainable_params = [p for p in self.module.model.parameters() if p.requires_grad]
        if not trainable_params:
            yield TrainingUpdate(0, 0.0, "[FAIL] No trainable parameters found", kind="fail")
            tb.close()
            return

        yield TrainingUpdate(0, 0.0, f"[INFO] Training {sum(p.numel() for p in trainable_params):,} parameters", kind="info")

        optimizer_type = getattr(cfg, "optimizer_type", "adamw")

        opt_kwargs = getattr(cfg, "optimizer_kwargs", {})

        optimizer = build_optimizer(
            trainable_params,
            optimizer_type=optimizer_type,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            device_type=self.module.device.type,
            optimizer_kwargs=opt_kwargs,
        )
        yield TrainingUpdate(0, 0.0, f"[INFO] Optimizer: {optimizer_type} {opt_kwargs}", kind="info")

        # -- Scheduler -------------------------------------------------------
        steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.gradient_accumulation_steps))
        total_steps = steps_per_epoch * cfg.max_epochs

        scheduler_type = getattr(cfg, "scheduler_type", "cosine")
        scheduler = build_scheduler(
            optimizer,
            scheduler_type=scheduler_type,
            total_steps=total_steps,
            warmup_steps=cfg.warmup_steps,
            lr=cfg.learning_rate,
            optimizer_type=optimizer_type,
        )
        yield TrainingUpdate(0, 0.0, f"[INFO] Scheduler: {scheduler_type}", kind="info")

        # -- Training memory features ----------------------------------------
        if getattr(cfg, "gradient_checkpointing", True):
            ckpt_ok, cache_off, grads_ok = configure_memory_features(
                self.module.model.decoder
            )
            self.module.force_input_grads_for_checkpointing = ckpt_ok
            if ckpt_ok:
                yield TrainingUpdate(
                    0, 0.0,
                    f"[INFO] Gradient checkpointing enabled "
                    f"(use_cache={not cache_off}, input_grads={grads_ok})",
                    kind="info",
                )
            else:
                yield TrainingUpdate(
                    0, 0.0, "[WARN] Gradient checkpointing not supported by this model",
                    kind="warn",
                )
        else:
            yield TrainingUpdate(
                0, 0.0,
                "[INFO] Gradient checkpointing OFF (faster but uses more VRAM)",
                kind="info",
            )

        # -- Encoder/VAE offloading ------------------------------------------
        if getattr(cfg, "offload_encoder", False):
            offloaded = offload_non_decoder(self.module.model)
            if offloaded:
                yield TrainingUpdate(0, 0.0, f"[INFO] Offloaded {offloaded} model components to CPU (saves VRAM)", kind="info")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # -- dtype / Fabric setup -------------------------------------------
        self.module.model = self.module.model.to(self.module.dtype)
        self.module.model.decoder, optimizer = self.fabric.setup(self.module.model.decoder, optimizer)

        # -- Resume ---------------------------------------------------------
        start_epoch = 0
        global_step = 0

        if cfg.resume_from and Path(cfg.resume_from).exists():
            try:
                yield TrainingUpdate(0, 0.0, f"[INFO] Loading checkpoint from {cfg.resume_from}", kind="info")
                resumed = yield from self._resume_checkpoint(
                    cfg.resume_from, optimizer, scheduler,
                )
                if resumed is not None:
                    start_epoch, global_step = resumed
            except Exception as exc:
                logger.exception("Failed to load checkpoint")
                yield TrainingUpdate(0, 0.0, f"[WARN] Checkpoint load failed: {exc} -- starting fresh", kind="warn")
                start_epoch = 0
                global_step = 0

        # -- Training loop --------------------------------------------------
        accumulation_step = 0
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        self.module.model.decoder.train()

        ema_loss = None
        ema_alpha = 0.1

        for epoch in range(start_epoch, cfg.max_epochs):
            epoch_loss = 0.0
            num_updates = 0
            epoch_start = time.time()

            for _batch_idx, batch in enumerate(train_loader):
                # Stop signal
                if training_state and training_state.get("should_stop", False):
                    _stop_loss = (
                        accumulated_loss * cfg.gradient_accumulation_steps
                        / max(accumulation_step, 1)
                    )
                    yield TrainingUpdate(global_step, _stop_loss, "[INFO] Training stopped by user", kind="complete")
                    tb.close()
                    return

                loss = self.module.training_step(batch)
                loss = loss / cfg.gradient_accumulation_steps
                self.fabric.backward(loss)
                accumulated_loss += loss.item()
                del loss  # free scalar tensor immediately
                accumulation_step += 1

                if accumulation_step >= cfg.gradient_accumulation_steps:
                    self.fabric.clip_gradients(
                        self.module.model.decoder, optimizer, max_norm=cfg.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    global_step += 1

                    avg_loss = accumulated_loss * cfg.gradient_accumulation_steps / accumulation_step

                    if ema_loss is None:
                        ema_loss = avg_loss
                    else:
                        ema_loss = ema_alpha * avg_loss + (1 - ema_alpha) * ema_loss

                    _lr = scheduler.get_last_lr()[0]
                    if global_step % cfg.log_every == 0:
                        tb.log_loss(avg_loss, global_step)
                        tb.log_lr(_lr, global_step)
                        yield TrainingUpdate(
                            step=global_step, loss=avg_loss,
                            msg=f"Epoch {epoch + 1}/{cfg.max_epochs}, Step {global_step}, Loss: {avg_loss:.4f}",
                            kind="step", epoch=epoch + 1, max_epochs=cfg.max_epochs, lr=_lr,
                            steps_per_epoch=steps_per_epoch,
                        )

                    if global_step % cfg.log_heavy_every == 0:
                        tb.log_per_layer_grad_norms(self.module.model, global_step)

                    optimizer.zero_grad(set_to_none=True)
                    epoch_loss += avg_loss
                    num_updates += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

                    # Periodic CUDA cache cleanup to prevent intra-epoch
                    # memory fragmentation on consumer GPUs.
                    if torch.cuda.is_available() and global_step % cfg.log_every == 0:
                        torch.cuda.empty_cache()

            # Flush remainder
            if accumulation_step > 0:
                self.fabric.clip_gradients(
                    self.module.model.decoder, optimizer, max_norm=cfg.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                global_step += 1

                avg_loss = accumulated_loss * cfg.gradient_accumulation_steps / accumulation_step
                _lr = scheduler.get_last_lr()[0]
                if global_step % cfg.log_every == 0:
                    tb.log_loss(avg_loss, global_step)
                    tb.log_lr(_lr, global_step)
                    yield TrainingUpdate(
                        step=global_step, loss=avg_loss,
                        msg=f"Epoch {epoch + 1}/{cfg.max_epochs}, Step {global_step}, Loss: {avg_loss:.4f}",
                        kind="step", epoch=epoch + 1, max_epochs=cfg.max_epochs, lr=_lr,
                        steps_per_epoch=steps_per_epoch,
                    )

                optimizer.zero_grad(set_to_none=True)
                epoch_loss += avg_loss
                num_updates += 1
                accumulated_loss = 0.0
                accumulation_step = 0

            # End of epoch
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / max(num_updates, 1)
            tb.log_epoch_loss(avg_epoch_loss, epoch + 1)
            yield TrainingUpdate(
                step=global_step, loss=avg_epoch_loss,
                msg=f"[OK] Epoch {epoch + 1}/{cfg.max_epochs} in {epoch_time:.1f}s, Loss: {avg_epoch_loss:.4f}",
                kind="epoch", epoch=epoch + 1, max_epochs=cfg.max_epochs, epoch_time=epoch_time,
            )

            # Checkpoint
            should_save = (epoch + 1) >= cfg.save_start_epoch and \
                          (epoch + 1) % cfg.save_every_n_epochs == 0

            # Проверка порога Loss (если он задан и больше 0)
            if should_save and cfg.save_loss_threshold > 0:
                # Используем EMA Loss вместо усредненного за эпоху (или fallback)
                current_loss_to_check = ema_loss if ema_loss is not None else avg_epoch_loss
                if current_loss_to_check >= cfg.save_loss_threshold:
                    should_save = False
                    logger.info(f"Skipping checkpoint: EMA Loss {current_loss_to_check:.4f} >= Threshold {cfg.save_loss_threshold}")

            if should_save:
                ckpt_dir = str(output_dir / "checkpoints" / f"epoch_{epoch + 1}")
                self._save_checkpoint(optimizer, scheduler, epoch + 1, global_step, ckpt_dir)

                if cfg.generate_preview:
                    self._generate_preview(output_dir, epoch + 1, self.module.device)

                yield TrainingUpdate(
                    step=global_step, loss=avg_epoch_loss,
                    msg=f"[OK] Checkpoint saved at epoch {epoch + 1}",
                    kind="checkpoint", epoch=epoch + 1, max_epochs=cfg.max_epochs,
                    checkpoint_path=ckpt_dir,
                )

            # Clear CUDA cache AFTER checkpoint save so serialization
            # temporaries are also freed.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # -- Sanity check: did we actually train? ----------------------------
        if global_step == 0:
            tb.close()
            yield TrainingUpdate(
                step=0, loss=0.0,
                msg=(
                    "[FAIL] Training completed 0 steps -- no batches were processed.\n"
                    "       Possible causes:\n"
                    "         - Dataset directory is empty or contains no valid .pt files\n"
                    "         - DataLoader failed to yield batches (device/platform issue)\n"
                    "       Check the dataset path and try again."
                ),
                kind="fail",
            )
            return

        # -- Final save -----------------------------------------------------
        final_path = str(output_dir / "final")
        self._save_final(final_path)
        final_loss = self.module.training_losses[-1] if self.module.training_losses else 0.0

        adapter_label = "LoKR" if self.adapter_type == "lokr" else "LoRA"
        tb.flush()
        tb.close()
        yield TrainingUpdate(
            step=global_step, loss=final_loss,
            msg=(
                f"[OK] Training complete! {adapter_label} saved to {final_path}\n"
                f"     For inference, set your LoRA path to: {final_path}"
            ),
            kind="complete",
        )
