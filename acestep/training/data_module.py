"""
PyTorch Lightning DataModule for LoRA Training

Handles data loading and preprocessing for training ACE-Step LoRA adapters.
Supports both raw audio loading and preprocessed tensor loading.
"""

import os
import json
import random
from typing import Optional, List, Dict, Any, Tuple
from loguru import logger

from acestep.training.path_safety import safe_path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

try:
    from lightning.pytorch import LightningDataModule
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    logger.warning("Lightning not installed. Training module will not be available.")
    class LightningDataModule:
        pass


class PreprocessedTensorDataset(Dataset):
    def __init__(self, tensor_dir: str):
        validated_dir = safe_path(tensor_dir)
        if not os.path.isdir(validated_dir):
            raise ValueError(f"Not an existing directory: {tensor_dir}")
        self.tensor_dir = validated_dir
        self.sample_paths: List[str] = []
        
        manifest_path = safe_path("manifest.json", base=self.tensor_dir)
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            raw_paths = manifest.get("samples", [])
            for raw in raw_paths:
                resolved = self._resolve_manifest_path(raw)
                if resolved is not None:
                    self.sample_paths.append(resolved)
        else:
            for f in os.listdir(self.tensor_dir):
                if f.endswith('.pt') and f != "manifest.json":
                    self.sample_paths.append(safe_path(f, base=self.tensor_dir))
        
        self.valid_paths = [p for p in self.sample_paths if os.path.exists(p)]
        
        if len(self.valid_paths) != len(self.sample_paths):
            logger.warning(f"Some tensor files not found: {len(self.sample_paths) - len(self.valid_paths)} missing")
        
        logger.info(f"PreprocessedTensorDataset: {len(self.valid_paths)} samples from {self.tensor_dir}")
    
    def _resolve_manifest_path(self, raw: str) -> Optional[str]:
        try:
            child = safe_path(raw, base=self.tensor_dir)
            if os.path.exists(child): return child
        except ValueError: pass
        try:
            child = safe_path(raw)
            if os.path.exists(child): return child
        except ValueError: pass
        return None

    def __len__(self) -> int:
        return len(self.valid_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tensor_path = self.valid_paths[idx]
        data = torch.load(tensor_path, map_location='cpu', weights_only=True)
        
        return {
            "target_latents": data["target_latents"],
            "attention_mask": data["attention_mask"],
            "encoder_hidden_states": data["encoder_hidden_states"],
            "encoder_attention_mask": data["encoder_attention_mask"],
            "context_latents": data["context_latents"],
            # Добавлено для полного обучения энкодеров:
            "text_hidden_states": data.get("text_hidden_states", torch.zeros(1)),
            "text_attention_mask": data.get("text_attention_mask", torch.zeros(1)),
            "lyric_hidden_states": data.get("lyric_hidden_states", torch.zeros(1)),
            "lyric_attention_mask": data.get("lyric_attention_mask", torch.zeros(1)),
            "metadata": data.get("metadata", {}),
        }


def collate_preprocessed_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_latent_len = max(s["target_latents"].shape[0] for s in batch)
    max_encoder_len = max(s["encoder_hidden_states"].shape[0] for s in batch)
    
    target_latents, attention_masks = [], []
    encoder_hidden_states, encoder_attention_masks = [], []
    context_latents = []
    
    for sample in batch:
        tl = sample["target_latents"]
        if tl.shape[0] < max_latent_len:
            pad = tl.new_zeros(max_latent_len - tl.shape[0], tl.shape[1])
            tl = torch.cat([tl, pad], dim=0)
        target_latents.append(tl)
        
        am = sample["attention_mask"]
        if am.shape[0] < max_latent_len:
            pad = am.new_zeros(max_latent_len - am.shape[0])
            am = torch.cat([am, pad], dim=0)
        attention_masks.append(am)
        
        cl = sample["context_latents"]
        if cl.shape[0] < max_latent_len:
            pad = cl.new_zeros(max_latent_len - cl.shape[0], cl.shape[1])
            cl = torch.cat([cl, pad], dim=0)
        context_latents.append(cl)
        
        ehs = sample["encoder_hidden_states"]
        if ehs.shape[0] < max_encoder_len:
            pad = ehs.new_zeros(max_encoder_len - ehs.shape[0], ehs.shape[1])
            ehs = torch.cat([ehs, pad], dim=0)
        encoder_hidden_states.append(ehs)
        
        eam = sample["encoder_attention_mask"]
        if eam.shape[0] < max_encoder_len:
            pad = eam.new_zeros(max_encoder_len - eam.shape[0])
            eam = torch.cat([eam, pad], dim=0)
        encoder_attention_masks.append(eam)
    
    result = {
        "target_latents": torch.stack(target_latents),
        "attention_mask": torch.stack(attention_masks),
        "encoder_hidden_states": torch.stack(encoder_hidden_states),
        "encoder_attention_mask": torch.stack(encoder_attention_masks),
        "context_latents": torch.stack(context_latents),
        "metadata": [s["metadata"] for s in batch],
    }

    # Логика паддинга для E2E обучения энкодеров
    if "text_hidden_states" in batch[0] and batch[0]["text_hidden_states"].dim() > 1:
        max_text_len = max(s["text_hidden_states"].shape[0] for s in batch)
        max_lyric_len = max(s["lyric_hidden_states"].shape[0] for s in batch)
        
        t_hs, t_mask, l_hs, l_mask = [], [], [], []
        for s in batch:
            ths = s["text_hidden_states"]
            if ths.shape[0] < max_text_len:
                pad = ths.new_zeros(max_text_len - ths.shape[0], ths.shape[1])
                ths = torch.cat([ths, pad], dim=0)
            t_hs.append(ths)
            
            tm = s["text_attention_mask"]
            if tm.shape[0] < max_text_len:
                pad = tm.new_zeros(max_text_len - tm.shape[0])
                tm = torch.cat([tm, pad], dim=0)
            t_mask.append(tm)
            
            lhs = s["lyric_hidden_states"]
            if lhs.shape[0] < max_lyric_len:
                pad = lhs.new_zeros(max_lyric_len - lhs.shape[0], lhs.shape[1])
                lhs = torch.cat([lhs, pad], dim=0)
            l_hs.append(lhs)
            
            lm = s["lyric_attention_mask"]
            if lm.shape[0] < max_lyric_len:
                pad = lm.new_zeros(max_lyric_len - lm.shape[0])
                lm = torch.cat([lm, pad], dim=0)
            l_mask.append(lm)
            
        result["text_hidden_states"] = torch.stack(t_hs)
        result["text_attention_mask"] = torch.stack(t_mask)
        result["lyric_hidden_states"] = torch.stack(l_hs)
        result["lyric_attention_mask"] = torch.stack(l_mask)

    return result


class PreprocessedDataModule(LightningDataModule if LIGHTNING_AVAILABLE else object):
    def __init__(
        self,
        tensor_dir: str,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory_device: str = "",
        val_split: float = 0.0,
    ):
        if LIGHTNING_AVAILABLE:
            super().__init__()
        
        self.tensor_dir = tensor_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory_device = pin_memory_device
        self.val_split = val_split
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            full_dataset = PreprocessedTensorDataset(self.tensor_dir)
            if self.val_split > 0 and len(full_dataset) > 1:
                n_val = max(1, int(len(full_dataset) * self.val_split))
                n_train = len(full_dataset) - n_val
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    full_dataset, [n_train, n_val]
                )
            else:
                self.train_dataset = full_dataset
                self.val_dataset = None
    
    def train_dataloader(self) -> DataLoader:
        prefetch_factor = None if self.num_workers == 0 else self.prefetch_factor
        persistent_workers = False if self.num_workers == 0 else self.persistent_workers
        kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_preprocessed_batch,
            drop_last=False,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        if self.pin_memory_device:
            kwargs["pin_memory_device"] = self.pin_memory_device
        return DataLoader(**kwargs)
    
    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None: return None
        prefetch_factor = None if self.num_workers == 0 else self.prefetch_factor
        persistent_workers = False if self.num_workers == 0 else self.persistent_workers
        kwargs = dict(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_preprocessed_batch,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        if self.pin_memory_device:
            kwargs["pin_memory_device"] = self.pin_memory_device
        return DataLoader(**kwargs)

class AceStepTrainingDataset(Dataset):
    pass
def collate_training_batch(batch):
    pass
class AceStepDataModule(object):
    pass
def load_dataset_from_json(json_path):
    pass