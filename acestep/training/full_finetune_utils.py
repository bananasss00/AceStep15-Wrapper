"""
Utility functions for full fine-tuning of ACE-Step models.

Contains helper functions for preparing full fine-tuning training,
merging weights, and managing full model checkpoints.
"""

import os
import torch
from typing import Dict, Any, Optional
from loguru import logger

from acestep.training.configs import TrainingConfig
from acestep.training.full_finetune import FullFinetuneTrainer


def create_full_finetune_config(
    output_dir: str = "./full_finetune_output",
    learning_rate: float = 1e-5,
    batch_size: int = 1,
    max_epochs: int = 100,
    gradient_accumulation_steps: int = 4,
    save_every_n_epochs: int = 10,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    gradient_checkpointing: bool = True,
    seed: int = 42,
    **kwargs,
) -> TrainingConfig:
    """
    Create a training configuration optimized for full fine-tuning.

    Args:
        output_dir: Directory to save checkpoints and logs
        learning_rate: Learning rate for training
        batch_size: Training batch size
        max_epochs: Maximum number of training epochs
        gradient_accumulation_steps: Number of gradient accumulation steps
        save_every_n_epochs: Save checkpoint every N epochs
        weight_decay: Weight decay for optimizer
        max_grad_norm: Maximum gradient norm for clipping
        gradient_checkpointing: Whether to enable gradient checkpointing
        seed: Random seed for reproducibility
        **kwargs: Additional configuration parameters

    Returns:
        TrainingConfig object for full fine-tuning
    """
    config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_every_n_epochs=save_every_n_epochs,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        gradient_checkpointing=gradient_checkpointing,
        seed=seed,
        output_dir=output_dir,
        **kwargs,
    )
    return config


def merge_full_finetune_weights(
    base_model_path: str,
    fine_tuned_model_path: str,
    output_path: str,
    merge_ratio: float = 1.0,
) -> None:
    """
    Merge base model weights with fine-tuned weights.

    This function can be used to create a hybrid model that combines
    the base model with fine-tuned weights.

    Args:
        base_model_path: Path to base model checkpoint
        fine_tuned_model_path: Path to fine-tuned model checkpoint
        output_path: Path to save merged model
        merge_ratio: Ratio of fine-tuned weights to use (0.0 = base only, 1.0 = fine-tuned only)
    """
    logger.info(
        f"Merging models: base={base_model_path}, fine_tuned={fine_tuned_model_path}"
    )

    # Load models
    base_state_dict = torch.load(base_model_path, map_location="cpu", weights_only=True)
    fine_tuned_state_dict = torch.load(
        fine_tuned_model_path, map_location="cpu", weights_only=True
    )

    # Merge weights
    merged_state_dict = {}

    # For each parameter, blend base and fine-tuned weights
    for key in base_state_dict.keys():
        if key in fine_tuned_state_dict:
            base_weight = base_state_dict[key]
            fine_tuned_weight = fine_tuned_state_dict[key]

            # Blend weights based on merge_ratio
            merged_weight = (
                1.0 - merge_ratio
            ) * base_weight + merge_ratio * fine_tuned_weight
            merged_state_dict[key] = merged_weight
        else:
            # Keep base weight if not in fine-tuned model
            merged_state_dict[key] = base_state_dict[key]

    # Save merged model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(merged_state_dict, output_path)
    logger.info(f"Merged model saved to {output_path}")


def validate_full_finetune_checkpoint(checkpoint_path: str) -> bool:
    """
    Validate that a checkpoint contains a valid full fine-tuned model.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        True if checkpoint is valid, False otherwise
    """
    try:
        model_path = os.path.join(checkpoint_path, "model_state_dict.pt")
        if not os.path.exists(model_path):
            return False

        # Try to load the checkpoint to validate it
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            return False

        return True
    except Exception as e:
        logger.error(f"Checkpoint validation failed: {e}")
        return False


def get_full_finetune_stats(model_path: str) -> Dict[str, Any]:
    """
    Get statistics about a full fine-tuned model.

    Args:
        model_path: Path to model checkpoint

    Returns:
        Dictionary with model statistics
    """
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        total_params = sum(p.numel() for p in state_dict.values())
        param_shapes = [p.shape for p in state_dict.values()]

        return {
            "total_parameters": total_params,
            "num_tensors": len(state_dict),
            "tensor_shapes": param_shapes,
        }
    except Exception as e:
        logger.error(f"Failed to get model stats: {e}")
        return {}
