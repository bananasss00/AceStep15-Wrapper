"""
Two-Pass CLI Preprocessing for ACE-Step Training V2.

Converts raw audio files into ``.pt`` tensor files compatible with
``PreprocessedDataModule``.  Uses upstream sub-functions directly and
loads models **sequentially** to minimise peak VRAM:

    Pass 1 (Light ~3 GB):  VAE + Text Encoder  -> intermediate ``.tmp.pt``
    Pass 2 (Heavy ~6 GB):  DIT encoder          -> final ``.pt``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from acestep.training_v2.preprocess_discovery import (
    discover_audio_files as _discover_audio_files,
    load_dataset_metadata as _load_dataset_metadata,
    load_sample_metadata as _load_sample_metadata,
    select_genre_indices as _select_genre_indices,
)
from acestep.training_v2.preprocess_prompt import (
    build_simple_prompt as _build_simple_prompt,
)
from acestep.training_v2.preprocess_vae import (
    TARGET_SR as _TARGET_SR,
    tiled_vae_encode as _tiled_vae_encode,
)

logger = logging.getLogger(__name__)

def preprocess_audio_files(
    audio_dir: Optional[str],
    output_dir: str,
    checkpoint_dir: str,
    variant: str = "turbo",
    max_duration: float = 240.0,
    dataset_json: Optional[str] = None,
    device: str = "auto",
    precision: str = "auto",
    progress_callback: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
) -> Dict[str, Any]:
    from acestep.training_v2.gpu_utils import detect_gpu

    gpu = detect_gpu(device, precision)
    dev = gpu.device
    prec = gpu.precision

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    audio_files = _discover_audio_files(audio_dir, dataset_json)
    if not audio_files:
        logger.warning("[Side-Step] No audio files found")
        return {"processed": 0, "failed": 0, "total": 0, "output_dir": str(out_path)}

    total = len(audio_files)
    logger.info("[Side-Step] Found %d audio files to preprocess", total)

    sample_meta = _load_sample_metadata(dataset_json, audio_files)
    ds_meta = _load_dataset_metadata(dataset_json)

    ds_tag = ds_meta.get("custom_tag", "")
    if ds_tag:
        for sm in sample_meta.values():
            if not sm.get("custom_tag"):
                sm["custom_tag"] = ds_tag

    intermediates, pass1_failed = _pass1_light(
        audio_files=audio_files,
        sample_meta=sample_meta,
        ds_meta=ds_meta,
        out_path=out_path,
        checkpoint_dir=checkpoint_dir,
        variant=variant,
        device=dev,
        precision=prec,
        max_duration=max_duration,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

    processed, pass2_failed = _pass2_heavy(
        intermediates=intermediates,
        out_path=out_path,
        checkpoint_dir=checkpoint_dir,
        variant=variant,
        device=dev,
        precision=prec,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

    failed = pass1_failed + pass2_failed
    result = {
        "processed": processed,
        "failed": failed,
        "total": total,
        "output_dir": str(out_path),
    }
    logger.info("[Side-Step] Preprocessing complete: %d/%d processed, %d failed", processed, total, failed)
    return result


def _pass1_light(
    audio_files: List[Path],
    sample_meta: Dict[str, Dict[str, Any]],
    ds_meta: Dict[str, Any],
    out_path: Path,
    checkpoint_dir: str,
    variant: str,
    device: str,
    precision: str,
    max_duration: float,
    progress_callback: Optional[Callable],
    cancel_check: Optional[Callable],
) -> tuple[List[Path], int]:
    from acestep.training_v2.model_loader import load_vae, load_text_encoder, load_silence_latent, unload_models, _resolve_dtype
    from acestep.training.dataset_builder_modules.preprocess_audio import load_audio_stereo
    from acestep.training.dataset_builder_modules.preprocess_text import encode_text
    from acestep.training.dataset_builder_modules.preprocess_lyrics import encode_lyrics

    dtype = _resolve_dtype(precision)

    logger.info("[Side-Step] Pass 1/2: Loading VAE + Text Encoder ...")
    vae = load_vae(checkpoint_dir, device, precision)
    tokenizer, text_enc = load_text_encoder(checkpoint_dir, device, precision)
    silence_latent = load_silence_latent(checkpoint_dir, device, precision, variant=variant)

    intermediates: List[Path] = []
    failed = 0
    total = len(audio_files)

    tag_position = ds_meta.get("tag_position", "prepend")
    genre_ratio = ds_meta.get("genre_ratio", 0)
    genre_indices = _select_genre_indices(total, genre_ratio)

    try:
        for i, af in enumerate(audio_files):
            if cancel_check and cancel_check(): break
            if progress_callback: progress_callback(i, total, f"[Pass 1] {af.name}")

            final_pt = out_path / f"{af.stem}.pt"
            if final_pt.exists():
                logger.info("[Side-Step] Skipping (final exists): %s", af.name)
                continue

            try:
                audio, _sr = load_audio_stereo(str(af), _TARGET_SR, max_duration)
                audio = audio.unsqueeze(0).to(device=device, dtype=vae.dtype)

                with torch.no_grad():
                    target_latents = _tiled_vae_encode(vae, audio, dtype)

                del audio
                latent_length = target_latents.shape[1]
                attention_mask = torch.ones(1, latent_length, device=device, dtype=dtype)

                sm = sample_meta.get(af.name, {})
                caption = sm.get("caption", af.stem)
                lyrics = sm.get("lyrics", "[Instrumental]")

                use_genre = i in genre_indices
                text_prompt = _build_simple_prompt(sm, tag_position=tag_position, use_genre=use_genre)

                with torch.no_grad():
                    text_hs, text_mask = encode_text(text_enc, tokenizer, text_prompt, device, dtype)
                    lyric_hs, lyric_mask = encode_lyrics(text_enc, tokenizer, lyrics, device, dtype)

                tmp_path = out_path / f"{af.stem}.tmp.pt"
                torch.save({
                    "target_latents": target_latents.squeeze(0).cpu(),
                    "attention_mask": attention_mask.squeeze(0).cpu(),
                    "text_hidden_states": text_hs.cpu(),
                    "text_attention_mask": text_mask.cpu(),
                    "lyric_hidden_states": lyric_hs.cpu(),
                    "lyric_attention_mask": lyric_mask.cpu(),
                    "silence_latent": silence_latent.cpu(),
                    "latent_length": latent_length,
                    "metadata": {
                        "audio_path": str(af),
                        "filename": af.name,
                        "caption": caption,
                        "lyrics": lyrics,
                        "duration": sm.get("duration", 0),
                        "bpm": sm.get("bpm"),
                        "keyscale": sm.get("keyscale", ""),
                        "timesignature": sm.get("timesignature", ""),
                        "genre": sm.get("genre", ""),
                        "is_instrumental": sm.get("is_instrumental", True),
                        "custom_tag": sm.get("custom_tag", ""),
                        "prompt_override": sm.get("prompt_override"),
                    },
                }, tmp_path)

                del target_latents, attention_mask, text_hs, text_mask, lyric_hs, lyric_mask
                if torch.cuda.is_available(): torch.cuda.empty_cache()

                intermediates.append(tmp_path)
            except Exception as exc:
                failed += 1
                logger.error("[Side-Step] Pass 1 FAIL %s: %s", af.name, exc)

    finally:
        unload_models(vae, text_enc, tokenizer, silence_latent)

    if progress_callback: progress_callback(total, total, "[Pass 1] Done")
    return intermediates, failed


def _pass2_heavy(
    intermediates: List[Path],
    out_path: Path,
    checkpoint_dir: str,
    variant: str,
    device: str,
    precision: str,
    progress_callback: Optional[Callable],
    cancel_check: Optional[Callable],
) -> tuple[int, int]:
    if not intermediates: return 0, 0

    from acestep.training_v2.model_loader import load_decoder_for_training, unload_models, _resolve_dtype
    from acestep.training.dataset_builder_modules.preprocess_encoder import run_encoder
    from acestep.training.dataset_builder_modules.preprocess_context import build_context_latents

    dtype = _resolve_dtype(precision)
    model = load_decoder_for_training(checkpoint_dir, variant, device, precision)

    processed = 0
    failed = 0
    total = len(intermediates)

    try:
        for i, tmp_path in enumerate(intermediates):
            if cancel_check and cancel_check(): break
            if progress_callback: progress_callback(i, total, f"[Pass 2] {tmp_path.stem}")

            try:
                data = torch.load(str(tmp_path), weights_only=False)

                model_device = next(model.parameters()).device
                model_dtype = next(model.parameters()).dtype

                text_hs = data["text_hidden_states"].to(model_device, dtype=model_dtype)
                text_mask = data["text_attention_mask"].to(model_device, dtype=model_dtype)
                lyric_hs = data["lyric_hidden_states"].to(model_device, dtype=model_dtype)
                lyric_mask = data["lyric_attention_mask"].to(model_device, dtype=model_dtype)
                silence_latent = data["silence_latent"].to(model_device, dtype=model_dtype)
                latent_length = data["latent_length"]

                encoder_hs, encoder_mask = run_encoder(
                    model,
                    text_hidden_states=text_hs,
                    text_attention_mask=text_mask,
                    lyric_hidden_states=lyric_hs,
                    lyric_attention_mask=lyric_mask,
                    device=str(model_device),
                    dtype=model_dtype,
                )

                if silence_latent.dim() == 2: silence_latent = silence_latent.unsqueeze(0)
                context_latents = build_context_latents(silence_latent, latent_length, str(model_device), model_dtype)
                del silence_latent

                base_name = tmp_path.name.replace(".tmp.pt", ".pt")
                final_path = out_path / base_name
                meta = data["metadata"]
                
                t_hs = data.get("text_hidden_states", torch.zeros(1))
                t_mask = data.get("text_attention_mask", torch.zeros(1))
                l_hs = data.get("lyric_hidden_states", torch.zeros(1))
                l_mask = data.get("lyric_attention_mask", torch.zeros(1))

                torch.save({
                    "target_latents": data["target_latents"],
                    "attention_mask": data["attention_mask"],
                    "encoder_hidden_states": encoder_hs.squeeze(0).cpu(),
                    "encoder_attention_mask": encoder_mask.squeeze(0).cpu(),
                    "context_latents": context_latents.squeeze(0).cpu(),
                    # СОХРАНЯЕМ В ФИНАЛЬНЫЙ PT ФАЙЛ СЫРЫЕ ТЕКСТОВЫЕ ЭМБЕДДИНГИ ДЛЯ FULL E2E FINETUNING (УДАЛЯЯ БАТЧ)
                    "text_hidden_states": t_hs.squeeze(0).cpu() if t_hs.dim() == 3 else t_hs.cpu(),
                    "text_attention_mask": t_mask.squeeze(0).cpu() if t_mask.dim() == 2 else t_mask.cpu(),
                    "lyric_hidden_states": l_hs.squeeze(0).cpu() if l_hs.dim() == 3 else l_hs.cpu(),
                    "lyric_attention_mask": l_mask.squeeze(0).cpu() if l_mask.dim() == 2 else l_mask.cpu(),
                    "metadata": meta,
                }, final_path)

                del encoder_hs, encoder_mask, context_latents, text_hs, text_mask, lyric_hs, lyric_mask, data
                if torch.cuda.is_available(): torch.cuda.empty_cache()

                tmp_path.unlink(missing_ok=True)
                processed += 1

            except Exception as exc:
                failed += 1
                logger.error("[Side-Step] Pass 2 FAIL %s: %s", tmp_path.stem, exc)

    finally:
        unload_models(model)

    if progress_callback: progress_callback(total, total, "[Pass 2] Done")
    return processed, failed