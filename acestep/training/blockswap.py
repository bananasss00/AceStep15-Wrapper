"""
BlockSwap Implementation for Low-VRAM Training (Optimized).
Uses non_blocking transfers to reduce overhead.
"""
import torch
import torch.nn as nn
from loguru import logger

class BlockSwapManager:
    def __init__(self, model: nn.Module, device: torch.device, offload_device: str = "cpu"):
        self.model = model
        self.device = device
        # Force CPU device for offload
        self.offload_device = torch.device(offload_device)
        self.hook_handles = []
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def _find_transformer_layers(self) -> nn.ModuleList:
        search_targets = ["layers", "h", "blocks", "transformer_blocks"]
        
        root = self.model
        if hasattr(root, "base_model"):
            root = root.base_model
            if hasattr(root, "model"):
                root = root.model
        
        if hasattr(root, "decoder"):
            root = root.decoder

        queue = [root]
        visited = set()
        
        while queue:
            current = queue.pop(0)
            if id(current) in visited:
                continue
            visited.add(id(current))

            for target in search_targets:
                if hasattr(current, target):
                    candidate = getattr(current, target)
                    if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
                        return candidate

            for name, child in current.named_children():
                queue.append(child)
        
        return None

    def apply(self, offload_ratio: float = 1.0):
        """
        Apply BlockSwap hooks based on ratio.
        
        Args:
            offload_ratio (float): 0.0 to 1.0. 
                                   1.0 = offload all layers (slowest, max memory saving).
                                   0.5 = offload 50% layers (faster).
        """
        layers = self._find_transformer_layers()
        if not layers:
            logger.warning("BlockSwap: Could not find transformer layers. Skipping.")
            return

        total_layers = len(layers)
        # Calculate how many layers to swap
        num_to_swap = int(total_layers * offload_ratio)
        
        logger.info(f"BlockSwap: Optimization enabled. Offloading {num_to_swap}/{total_layers} layers.")

        # Move offloaded layers to CPU immediately
        for i, layer in enumerate(layers):
            if i < num_to_swap:
                layer.to(self.offload_device)
            else:
                layer.to(self.device)

        # --- HOOKS WITH NON_BLOCKING ---
        
        def pre_forward_hook(module, args):
            # Move to GPU asynchronously
            # This overlaps data transfer with previous computation if possible
            module.to(self.device, non_blocking=True)
            return args

        def post_forward_hook(module, args, output):
            # Move back to CPU asynchronously
            module.to(self.offload_device, non_blocking=True)
            return output

        def pre_backward_hook(module, grad_output):
            # Ensure weights are on GPU for gradient computation
            module.to(self.device, non_blocking=True)
            return grad_output

        def post_backward_hook(module, grad_input, grad_output):
            # Cleanup after backward
            module.to(self.offload_device, non_blocking=True)

        for i, layer in enumerate(layers):
            if i < num_to_swap:
                # Apply hooks only to swapped layers
                self.hook_handles.append(layer.register_forward_pre_hook(pre_forward_hook))
                self.hook_handles.append(layer.register_forward_hook(post_forward_hook))
                self.hook_handles.append(layer.register_full_backward_pre_hook(pre_backward_hook))
                self.hook_handles.append(layer.register_full_backward_hook(post_backward_hook))

    def remove(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        
        # Restore all to GPU
        layers = self._find_transformer_layers()
        if layers:
            for layer in layers:
                layer.to(self.device)
        torch.cuda.empty_cache()