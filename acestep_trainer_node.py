import os
import sys
import gc
import glob
import time
import random
import re
import json
import torch
import base64
import copy
import shutil
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
from contextlib import nullcontext

# Для графиков
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from safetensors.torch import load_file, save_file

import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
from server import PromptServer

try:
    from acestep.training_v2.configs import LoRAConfigV2, LoKRConfigV2, TrainingConfigV2
    # ВАЖНО: Не импортировать trainer_fixed здесь! Сначала применяем патч, потом импортируем
    from acestep.training_v2.model_loader import load_decoder_for_training
    from acestep.training_v2.preprocess import preprocess_audio_files
    from acestep.training_v2.gpu_utils import detect_gpu
    from acestep.training_v2.cli.validation import resolve_target_modules
    from acestep.training_v2.estimate import run_estimation
    from acestep.training_v2.fixed_lora_module import FixedLoRAModule
    from acestep.training_v2.timestep_sampling import sample_timesteps
    import acestep.training_v2.timestep_sampling as ts
    from acestep.training_v2.optim import build_optimizer, build_scheduler
except ImportError as e:
    print(f"⚠️ [ACE-Step] Failed to import acestep v2 modules: {e}")

try:
    torch.set_float32_matmul_precision('medium')
except:
    pass

# ======================================================================
# ПАТЧ: БЛОКИРОВКА СОХРАНЕНИЯ СОСТОЯНИЙ ОПТИМИЗАТОРА (РЕЗЕРВНЫЙ)
# Примечание: Основная поддержка добавлена в бекенд trainer_helpers.py
# Этот патч — резервная копия на случай обновлений бекенда
# ======================================================================
try:
    import acestep.training_v2.trainer_helpers as _th_pre_import

    if not hasattr(_th_pre_import, "_original_save_checkpoint"):
        _th_pre_import._original_save_checkpoint = _th_pre_import.save_checkpoint

        def _custom_save_checkpoint(trainer, optimizer, scheduler, epoch, global_step, ckpt_dir):
            import os
            import torch

            _th_pre_import.save_adapter_flat(trainer, ckpt_dir)
            save_state = getattr(trainer.training_config, "save_state", True)
            if not save_state:
                print(f"💾 [Checkpoint] Сохранение адаптеров на epoch {epoch} (training_state пропущен для экономии места)")
                return

            training_state = {
                "epoch": epoch,
                "global_step": global_step,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            state_path = os.path.join(ckpt_dir, "training_state.pt")
            torch.save(training_state, state_path)

            try:
                from safetensors.torch import save_file as _save_safetensors
                meta_tensors = {
                    "epoch": torch.tensor([epoch], dtype=torch.int64),
                    "global_step": torch.tensor([global_step], dtype=torch.int64),
                }
                sf_path = os.path.join(ckpt_dir, "training_state.safetensors")
                _save_safetensors(meta_tensors, sf_path)
            except Exception:
                pass
            print(f"💾 [Checkpoint] Адаптеры + training_state сохранены на epoch {epoch}")

        _th_pre_import.save_checkpoint = _custom_save_checkpoint
        print("✅ [Patch] save_checkpoint резервный патч применён")
except Exception as e:
    print(f"⚠️ [Patch] Ошибка резервного патча save_checkpoint: {e}")

# Теперь импортируем trainer_fixed после применения патча
try:
    from acestep.training_v2.trainer_fixed import FixedLoRATrainer
except ImportError as e:
    print(f"⚠️ [ACE-Step] Failed to import FixedLoRATrainer: {e}")

# Оптимизированный CFG Dropout (Без создания лишних нулей/единиц)
def fast_cfg_dropout(encoder_hidden_states, null_condition_emb, cfg_ratio=0.15):
    bsz = encoder_hidden_states.shape[0]
    device = encoder_hidden_states.device
    mask = (torch.rand(bsz, 1, 1, device=device) >= cfg_ratio)
    return torch.where(mask, encoder_hidden_states, null_condition_emb.to(encoder_hidden_states.dtype))

ts.apply_cfg_dropout = fast_cfg_dropout

try:
    if not hasattr(FixedLoRATrainer, "_original_generate_preview"):
        FixedLoRATrainer._original_generate_preview = FixedLoRATrainer._generate_preview

        def custom_generate_preview(self, output_dir, step, device):
            orig_dir = self.training_config.dataset_dir
            if hasattr(self, "main_tensor_dir"):
                self.training_config.dataset_dir = self.main_tensor_dir
            try:
                self._original_generate_preview(output_dir, step, device)
            finally:
                self.training_config.dataset_dir = orig_dir

        FixedLoRATrainer._generate_preview = custom_generate_preview
except NameError:
    pass

def bypass_safe_path(user_path, base=None):
    if base is not None:
        root = os.path.normpath(os.path.abspath(base))
    else:
        root = os.path.normpath(os.path.abspath(os.getcwd()))
    if os.path.isabs(user_path):
        return os.path.normpath(user_path)
    else:
        return os.path.normpath(os.path.join(root, user_path))

import acestep.training.path_safety
acestep.training.path_safety.safe_path = bypass_safe_path
import acestep.training.data_module
acestep.training.data_module.safe_path = bypass_safe_path
import acestep.training.lora_utils
acestep.training.lora_utils.safe_path = bypass_safe_path
import acestep.training.lokr_utils
acestep.training.lokr_utils.safe_path = bypass_safe_path

# ======================================================================
# FP8 / FP4 STOCHASTIC ROUNDING & QUANTIZATION UTILS
# ======================================================================
def calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=None):
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)))
    )
    mantissa_scaled += torch.rand(mantissa_scaled.size(), dtype=mantissa_scaled.dtype, layout=mantissa_scaled.layout, device=mantissa_scaled.device, generator=generator)
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)

def manual_stochastic_round_to_float8(x, dtype, generator=None):
    if dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif dtype == getattr(torch, "float8_e5m2", None):
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    else:
        raise ValueError("Unsupported dtype")

    x = x.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)
    exponent = torch.clamp(torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS, 0, 2**EXPONENT_BITS - 1)

    normal_mask = ~(exponent == 0)
    abs_x[:] = calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=generator)

    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x),
        (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x
    )

    inf = torch.finfo(dtype)
    torch.clamp(sign, min=inf.min, max=inf.max, out=sign)
    return sign

def stochastic_rounding(value, dtype, seed=0):
    if dtype in [torch.float32, torch.float16, torch.bfloat16]:
        return value.to(dtype=dtype)
    if dtype in[getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)]:
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        output = torch.empty_like(value, dtype=dtype)
        num_slices = max(1, int(value.numel() / (4096 * 4096)))
        slice_size = max(1, round(value.shape[0] / num_slices))
        for i in range(0, value.shape[0], slice_size):
            output[i:i+slice_size].copy_(manual_stochastic_round_to_float8(value[i:i+slice_size], dtype, generator=generator))
        return output
    if hasattr(torch, "float4_e2m1fn") and dtype == torch.float4_e2m1fn:
        return value.to(dtype)
    return value.to(dtype=dtype)

def _dequantize_nf4(weight_packed, scales, orig_shape, group_size, lut, compute_dtype):
    """Оптимизированная распаковка NF4 без использования тяжелых Int64 (long)"""
    w_flat = weight_packed.view(torch.uint8)
    
    unpacked = torch.empty((w_flat.shape[0] * 2,), dtype=torch.int32, device=w_flat.device)
    unpacked[0::2] = (w_flat >> 4).to(torch.int32)
    unpacked[1::2] = (w_flat & 0x0F).to(torch.int32)
    
    padded_in_features = unpacked.shape[0] // orig_shape[0]
    blocks_per_row = padded_in_features // group_size
    
    w_dequant = lut[unpacked].view(orig_shape[0], blocks_per_row, group_size)
    w_dequant = (w_dequant * scales).view(orig_shape[0], padded_in_features)
    
    if padded_in_features > orig_shape[1]:
        w_dequant = w_dequant[:, :orig_shape[1]]
        
    return w_dequant.to(compute_dtype)

class DequantizeFP8Linear(torch.autograd.Function):
    """ZeRO-VRAM Autograd для FP8. Запрещает кешировать входы X и 16-битные веса."""
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, x, weight_fp8, bias, compute_dtype):
        w_dequant = weight_fp8.to(compute_dtype)
        # Сохраняем ТОЛЬКО 8-битный вес! Мы не сохраняем X, т.к. градиент для слоя не нужен
        ctx.save_for_backward(weight_fp8)
        ctx.compute_dtype = compute_dtype
        output = F.linear(x, w_dequant, bias)
        del w_dequant # Обязательно освобождаем память
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        weight_fp8, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            w_dequant = weight_fp8.to(ctx.compute_dtype)
            grad_input = grad_output.matmul(w_dequant)
            del w_dequant
        return grad_input, None, None, None

class QuantizedFP8Linear(nn.Linear):
    @property
    def weight(self):
        w_dequant = self.weight_fp8.to(self.compute_dtype)
        return nn.Parameter(w_dequant, requires_grad=False)

    def forward(self, x):
        # БАГФИКС: Обходим autograd.Function во время генерации (инференса)
        if not torch.is_grad_enabled():
            w_dequant = self.weight_fp8.to(self.compute_dtype)
            # Фикс несовпадения типов (Float32 vs BFloat16)
            if w_dequant.dtype != x.dtype:
                w_dequant = w_dequant.to(x.dtype)
            return F.linear(x, w_dequant, self.bias)
            
        return DequantizeFP8Linear.apply(x, self.weight_fp8, self.bias, self.compute_dtype)

class DequantizeNF4Linear(torch.autograd.Function):
    """ZeRO-VRAM Autograd для NF4. Запрещает кешировать входы X и 16-битные веса."""
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, x, weight_packed, scales, bias, orig_shape, group_size, lut, compute_dtype):
        w_dequant = _dequantize_nf4(weight_packed, scales, orig_shape, group_size, lut, compute_dtype)
        ctx.save_for_backward(weight_packed, scales, lut)
        ctx.orig_shape = orig_shape
        ctx.group_size = group_size
        ctx.compute_dtype = compute_dtype
        output = F.linear(x, w_dequant, bias)
        del w_dequant
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        weight_packed, scales, lut = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            w_dequant = _dequantize_nf4(weight_packed, scales, ctx.orig_shape, ctx.group_size, lut, ctx.compute_dtype)
            grad_input = grad_output.matmul(w_dequant)
            del w_dequant
        return grad_input, None, None, None, None, None, None, None

class QuantizedNF4Linear(nn.Linear):
    @property
    def weight(self):
        w_dequant = _dequantize_nf4(
            self.weight_packed, self.scales, self.orig_shape, 
            self.group_size, self.nf4_lut, self.scales.dtype
        )
        return nn.Parameter(w_dequant, requires_grad=False)

    def forward(self, x):
        # БАГФИКС: Обходим autograd.Function во время генерации (инференса)
        if not torch.is_grad_enabled():
            w_dequant = _dequantize_nf4(
                self.weight_packed, self.scales, self.orig_shape, 
                self.group_size, self.nf4_lut, self.scales.dtype
            )
            # Фикс несовпадения типов (Float32 vs BFloat16)
            if w_dequant.dtype != x.dtype:
                w_dequant = w_dequant.to(x.dtype)
            return F.linear(x, w_dequant, self.bias)
            
        return DequantizeNF4Linear.apply(
            x, self.weight_packed, self.scales, self.bias, 
            self.orig_shape, self.group_size, self.nf4_lut, self.scales.dtype
        )

def apply_quantization_base_model(model, compute_dtype, quant_type="none", seed=42):
    if quant_type == "none":
        return

    converted_count = 0
    visited = set() # ЗАЩИТА: хранит id() обработанных модулей, чтобы не сжимать их дважды

    if quant_type == "fp8":
        if not hasattr(torch, "float8_e4m3fn"):
            print("⚠️ [Quantization] Your PyTorch version does not support float8_e4m3fn. Skipping.")
            return

        print(f"🪄 [Quantization] Quantizing frozen layers to FP8 (Stochastic Rounding)...")
        for name, module in model.named_modules():
            # Достаем оригинальный слой из-под обертки PEFT, если она есть
            if hasattr(module, 'base_layer') and isinstance(module.base_layer, nn.Linear):
                target_module = module.base_layer
            elif isinstance(module, nn.Linear):
                target_module = module
            else:
                continue

            # ЗАЩИТА от двойной обработки генератором
            if id(target_module) in visited:
                continue

            # Убеждаемся, что мы не пытаемся сжать обучаемые веса (LoRA A/B и т.д.)
            if hasattr(target_module, 'weight') and target_module.weight is not None and not target_module.weight.requires_grad:
                if "lora_" in name or "lycoris" in target_module.__class__.__name__.lower() or "lora" in target_module.__class__.__name__.lower():
                    continue
                
                visited.add(id(target_module))
                
                fp8_weight = stochastic_rounding(target_module.weight.data, torch.float8_e4m3fn, seed=seed)
                target_module.weight_fp8 = nn.Parameter(fp8_weight, requires_grad=False)
                target_module.compute_dtype = compute_dtype
                
                if target_module.bias is not None:
                    target_module.bias.data = target_module.bias.data.to(compute_dtype)
                    
                # Безопасное удаление оригинального 16-битного веса
                if hasattr(target_module, 'weight') and 'weight' in target_module._parameters:
                    del target_module._parameters['weight']
                elif hasattr(target_module, 'weight'):
                    delattr(target_module, 'weight')
                    
                target_module.__class__ = QuantizedFP8Linear 
                converted_count += 1
                
        print(f"✅ [Quantization] Successfully converted {converted_count} layers to FP8.")

    elif quant_type == "fp4":
        print(f"🪄 [Quantization] Quantizing frozen layers to NF4 (Block-64, OOM-Safe)...")
        group_size = 64

        NF4_LUT = torch.tensor([
            -1.0, -0.6961928, -0.52507305, -0.39491749,
            -0.28444138, -0.18477343, -0.091050036, 0.0,
            0.07958029, 0.1609302, 0.2461123, 0.33791524,
            0.44070983, 0.562617, 0.72295684, 1.0
        ], dtype=compute_dtype)

        for name, module in model.named_modules():
            if hasattr(module, 'base_layer') and isinstance(module.base_layer, nn.Linear):
                target_module = module.base_layer
            elif isinstance(module, nn.Linear):
                target_module = module
            else:
                continue

            if id(target_module) in visited:
                continue

            if hasattr(target_module, 'weight') and target_module.weight is not None and not target_module.weight.requires_grad:
                if "lora_" in name or "lycoris" in target_module.__class__.__name__.lower() or "lora" in target_module.__class__.__name__.lower():
                    continue

                visited.add(id(target_module))

                orig_shape = target_module.weight.shape
                device = target_module.weight.device
                
                pad_len = (group_size - (orig_shape[1] % group_size)) % group_size
                w_padded = target_module.weight.data.float() 
                if pad_len > 0:
                    w_padded = F.pad(w_padded, (0, pad_len))
                    
                w_grouped = w_padded.view(orig_shape[0], -1, group_size)
                blocks_per_row = w_grouped.shape[1]
                
                scales = torch.empty((orig_shape[0], blocks_per_row, 1), dtype=compute_dtype, device=device)
                quantized_indices = torch.empty((orig_shape[0], blocks_per_row, group_size), dtype=torch.uint8, device=device)
                
                CHUNK_SIZE = 128
                lut_device = NF4_LUT.to(device).view(1, 1, 1, 16)
                
                for i in range(0, orig_shape[0], CHUNK_SIZE):
                    end_i = min(i + CHUNK_SIZE, orig_shape[0])
                    w_chunk = w_grouped[i:end_i]
                    
                    chunk_scales = w_chunk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
                    scales[i:end_i] = chunk_scales.to(compute_dtype)
                    w_norm = w_chunk / chunk_scales
                    
                    distances = torch.abs(w_norm.unsqueeze(-1) - lut_device)
                    quantized_indices[i:end_i] = torch.argmin(distances, dim=-1).to(torch.uint8)

                flat_indices = quantized_indices.view(-1)
                high = flat_indices[0::2] << 4
                low = flat_indices[1::2]
                packed_weights = (high | low).contiguous()
                
                target_module.weight_packed = nn.Parameter(packed_weights, requires_grad=False)
                target_module.scales = nn.Parameter(scales, requires_grad=False) 
                target_module.register_buffer("nf4_lut", NF4_LUT.to(device))
                target_module.orig_shape = orig_shape
                target_module.group_size = group_size
                
                if target_module.bias is not None:
                    target_module.bias.data = target_module.bias.data.to(compute_dtype)
                    
                # Безопасное удаление оригинального 16-битного веса
                if hasattr(target_module, 'weight') and 'weight' in target_module._parameters:
                    del target_module._parameters['weight']
                elif hasattr(target_module, 'weight'):
                    delattr(target_module, 'weight')
                    
                target_module.__class__ = QuantizedNF4Linear 
                converted_count += 1
                
        print(f"✅ [Quantization] Successfully converted {converted_count} layers to NF4 Block-64.")

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# ======================================================================
# BLOCKSWAP MANAGER (ZeRO OFFLOAD)
# ======================================================================
class AdvancedBlockSwapManager:
    def __init__(self, model: nn.Module, device: torch.device, offload_device: str = "cpu"):
        self.model = model
        self.device = device
        self.offload_device = torch.device(offload_device)
        self.hook_handles =[]
        
        self.pinned_cpu_state = {}
        self.pinned_grad_state = {}
        self.pinned_opt_state = {} 
        self.is_loaded = {}
        
        self.compute_stream = torch.cuda.current_stream()
        self.transfer_stream = torch.cuda.Stream()
        
        self.load_events = {}
        self.compute_events = {}
        self.offloaded_layers =[]

    def _find_transformer_layers(self) -> nn.ModuleList:
        search_targets =["layers", "h", "blocks", "transformer_blocks"]
        root = self.model
        if hasattr(root, "base_model"): root = root.base_model
        if hasattr(root, "model"): root = root.model
        if hasattr(root, "decoder"): root = root.decoder

        queue =[root]
        visited = set()
        while queue:
            current = queue.pop(0)
            if id(current) in visited: continue
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
        layers = self._find_transformer_layers()
        if not layers: return
        total_layers = len(layers)
        num_to_swap = int(total_layers * offload_ratio)
        if num_to_swap == 0: return

        print(f"🚀 [ZeRO-Offload] Pipelining {num_to_swap}/{total_layers} layers. VRAM cache flush enabled.")

        self.offloaded_layers = [layers[i] for i in range(num_to_swap)]

        for idx, layer in enumerate(self.offloaded_layers):
            layer_id = id(layer)
            self.pinned_cpu_state[layer_id] = {}
            self.pinned_grad_state[layer_id] = {}
            self.load_events[layer_id] = torch.cuda.Event()
            self.compute_events[layer_id] = torch.cuda.Event()
            self.is_loaded[layer_id] = True
            
            for name, param in layer.named_parameters():
                pinned_t = torch.empty(param.shape, dtype=param.dtype, pin_memory=True)
                pinned_t.copy_(param.data)
                self.pinned_cpu_state[layer_id][name] = pinned_t
                param.data = pinned_t 
                
            for name, buf in layer.named_buffers():
                pinned_t = torch.empty(buf.shape, dtype=buf.dtype, pin_memory=True)
                pinned_t.copy_(buf.data)
                self.pinned_cpu_state[layer_id][name] = pinned_t
                buf.data = pinned_t

            self.is_loaded[layer_id] = False

        def _async_load(layer_idx):
            if layer_idx < 0 or layer_idx >= len(self.offloaded_layers): return
            layer = self.offloaded_layers[layer_idx]
            layer_id = id(layer)
            if self.is_loaded[layer_id]: return
            self.is_loaded[layer_id] = True
            
            with torch.cuda.stream(self.transfer_stream):
                for name, param in layer.named_parameters():
                    gpu_tensor = torch.empty_like(self.pinned_cpu_state[layer_id][name], device=self.device)
                    gpu_tensor.copy_(self.pinned_cpu_state[layer_id][name], non_blocking=True)
                    param.data = gpu_tensor
                    if name in self.pinned_grad_state[layer_id]:
                        if param.grad is None or param.grad.device != self.device:
                            param.grad = torch.empty_like(self.pinned_grad_state[layer_id][name], device=self.device)
                        param.grad.data.copy_(self.pinned_grad_state[layer_id][name], non_blocking=True)
                for name, buf in layer.named_buffers():
                    gpu_tensor = torch.empty_like(self.pinned_cpu_state[layer_id][name], device=self.device)
                    gpu_tensor.copy_(self.pinned_cpu_state[layer_id][name], non_blocking=True)
                    buf.data = gpu_tensor
                self.load_events[layer_id].record(self.transfer_stream)

        def _async_offload(layer_idx, is_backward=False):
            if layer_idx < 0 or layer_idx >= len(self.offloaded_layers): return
            layer = self.offloaded_layers[layer_idx]
            layer_id = id(layer)
            if not self.is_loaded[layer_id]: return
            self.is_loaded[layer_id] = False
            
            with torch.cuda.stream(self.transfer_stream):
                self.transfer_stream.wait_event(self.compute_events[layer_id])
                for name, param in layer.named_parameters():
                    if param.device != torch.device("cpu"):
                        self.pinned_cpu_state[layer_id][name].copy_(param.data, non_blocking=True)
                        param.data.record_stream(self.transfer_stream)
                        param.data = self.pinned_cpu_state[layer_id][name]
                    if is_backward and param.grad is not None and param.grad.device != torch.device("cpu"):
                        if name not in self.pinned_grad_state[layer_id]:
                            self.pinned_grad_state[layer_id][name] = torch.empty_like(param.grad.data, device="cpu", pin_memory=True)
                        self.pinned_grad_state[layer_id][name].copy_(param.grad.data, non_blocking=True)
                        param.grad.data.record_stream(self.transfer_stream)
                        param.grad.data = self.pinned_grad_state[layer_id][name]
                for name, buf in layer.named_buffers():
                    if buf.device != torch.device("cpu"):
                        self.pinned_cpu_state[layer_id][name].copy_(buf.data, non_blocking=True)
                        buf.data.record_stream(self.transfer_stream)
                        buf.data = self.pinned_cpu_state[layer_id][name]

        def pre_forward_hook(module, args):
            layer_idx = self.offloaded_layers.index(module)
            layer_id = id(module)
            _async_load(layer_idx)
            current_stream = torch.cuda.current_stream()
            current_stream.wait_event(self.load_events[layer_id])
            if not torch.is_grad_enabled(): _async_load(layer_idx + 1)
            return args

        def post_forward_hook(module, args, output):
            layer_idx = self.offloaded_layers.index(module)
            layer_id = id(module)
            current_stream = torch.cuda.current_stream()
            self.compute_events[layer_id].record(current_stream)
            if not torch.is_grad_enabled(): _async_offload(layer_idx, is_backward=False)
            return output

        def pre_backward_hook(module, grad_output):
            layer_idx = self.offloaded_layers.index(module)
            layer_id = id(module)
            _async_load(layer_idx)
            current_stream = torch.cuda.current_stream()
            current_stream.wait_event(self.load_events[layer_id])
            _async_load(layer_idx - 1)
            return grad_output

        def post_backward_hook(module, grad_input, grad_output):
            layer_idx = self.offloaded_layers.index(module)
            layer_id = id(module)
            current_stream = torch.cuda.current_stream()
            self.compute_events[layer_id].record(current_stream)
            _async_offload(layer_idx, is_backward=True)

        for layer in self.offloaded_layers:
            self.hook_handles.append(layer.register_forward_pre_hook(pre_forward_hook))
            self.hook_handles.append(layer.register_forward_hook(post_forward_hook))
            self.hook_handles.append(layer.register_full_backward_pre_hook(pre_backward_hook))
            self.hook_handles.append(layer.register_full_backward_hook(post_backward_hook))

    def prepare_for_optimizer(self, optimizer):
        for i in range(len(self.offloaded_layers)):
            layer = self.offloaded_layers[i]
            layer_id = id(layer)
            # FIX: Only load parameters that require gradients to save VRAM
            for name, param in layer.named_parameters():
                if not param.requires_grad: continue
                if param.data.device.type == "cpu":
                    param.data = param.data.to(self.device, non_blocking=True)
                if name in self.pinned_grad_state[layer_id] and param.grad is not None:
                    if param.grad.data.device.type == "cpu":
                        param.grad.data = param.grad.data.to(self.device, non_blocking=True)
                
                if param in optimizer.state:
                    for k, v in optimizer.state[param].items():
                        if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                            optimizer.state[param][k] = v.to(self.device, non_blocking=True)
        torch.cuda.current_stream().synchronize()

    def cleanup_after_optimizer(self, optimizer):
        for i in range(len(self.offloaded_layers)):
            layer = self.offloaded_layers[i]
            layer_id = id(layer)
            for name, param in layer.named_parameters():
                if not param.requires_grad: continue
                if param.data.device.type != "cpu":
                    self.pinned_cpu_state[layer_id][name].copy_(param.data, non_blocking=True)
                    param.data = self.pinned_cpu_state[layer_id][name]
                
                if param in optimizer.state:
                    pid = id(param)
                    if pid not in self.pinned_opt_state:
                        self.pinned_opt_state[pid] = {}
                    for k, v in optimizer.state[param].items():
                        if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                            if k not in self.pinned_opt_state[pid]:
                                self.pinned_opt_state[pid][k] = torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
                            self.pinned_opt_state[pid][k].copy_(v, non_blocking=True)
                            optimizer.state[param][k] = self.pinned_opt_state[pid][k] 
        torch.cuda.current_stream().synchronize()
        torch.cuda.empty_cache()

    def remove(self):
        for handle in self.hook_handles: handle.remove()
        self.hook_handles.clear()
        self.pinned_cpu_state.clear()
        self.pinned_grad_state.clear()
        self.pinned_opt_state.clear()
        self.offloaded_layers.clear()
        torch.cuda.empty_cache()

# ======================================================================
# V2 FULL FINETUNE MODULE & UTILS
# ======================================================================
CURRENT_REG_WEIGHT = 1.0

def _save_complete_model(trained_model_state, output_dir, source_model_dir, variant="turbo", extra_pnginfo=None, prompt_info=None):
    os.makedirs(output_dir, exist_ok=True)
    try:
        from safetensors.torch import load_file as st_load, save_file as st_save
        original_path = os.path.join(source_model_dir, "model.safetensors")
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"No model.safetensors found in {source_model_dir}")
        
        print(f"🔄 Слияние обученной модели с оригинальными весами из {source_model_dir}...")
        original_state = st_load(original_path)
        
        cleaned_state = {}
        for key, value in trained_model_state.items():
            clean_key = key
            for prefix in ("model.", "_forward_module.", "module."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            cleaned_state[clean_key] = value

        merged_state = {}
        merged_count = 0
        for orig_key, orig_val in original_state.items():
            if orig_key in cleaned_state:
                trained_tensor = cleaned_state[orig_key]
                if trained_tensor.shape == orig_val.shape:
                    bf16_tensor = trained_tensor.to(dtype=torch.bfloat16) if trained_tensor.is_floating_point() else trained_tensor
                    merged_state[orig_key] = bf16_tensor
                    merged_count += 1
                else:
                    merged_state[orig_key] = orig_val
            else:
                merged_state[orig_key] = orig_val

        metadata = {"format": "pt"}
        if extra_pnginfo and "workflow" in extra_pnginfo:
            metadata["workflow"] = json.dumps(extra_pnginfo["workflow"])
            try:
                with open(os.path.join(output_dir, "workflow.json"), "w", encoding="utf-8") as f:
                    json.dump(extra_pnginfo["workflow"], f, indent=2)
            except Exception as e:
                print(f"⚠️ Failed to save workflow.json: {e}")
                
        if prompt_info:
            metadata["prompt"] = json.dumps(prompt_info)

        print(f"✅ Успешно обновлено {merged_count} слоёв. Сохранение в {output_dir}...")
        st_save(merged_state, os.path.join(output_dir, "model.safetensors"), metadata=metadata)
        
        src_silence = os.path.join(source_model_dir, "silence_latent.pt")
        if os.path.exists(src_silence):
            shutil.copy2(src_silence, os.path.join(output_dir, "silence_latent.pt"))
            
        code_files =["config.json", "configuration_acestep_v15.py", "apg_guidance.py"]
        if variant == "turbo": code_files.append("modeling_acestep_v15_turbo.py")
        elif variant == "base": code_files.append("modeling_acestep_v15_base.py")
        elif variant == "sft": code_files.append("modeling_acestep_v15_base.py")
        elif variant == "xl_turbo": code_files.append("modeling_acestep_v15_xl_turbo.py")
        elif variant in ["xl_base", "xl_sft"]: code_files.append("modeling_acestep_v15_xl_base.py")
        
        for f in code_files:
            src = os.path.join(source_model_dir, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, f))
                
    except Exception as e:
        print(f"❌ Ошибка при слиянии весов: {e}")
        torch.save(trained_model_state, os.path.join(output_dir, "model_state_dict.pt"))


class FullFinetuneModuleV2(torch.nn.Module):
    def __init__(self, model, training_config, device, precision, encoder_train_mode="none", train_null_emb=True):
        super().__init__()
        self.training_config = training_config
        self.device = device
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else torch.float32)
        self.transfer_non_blocking = self.device_type in ("cuda", "xpu")
        
        self.timestep_mu = getattr(model.config, 'timestep_mu', -0.4)
        self.timestep_sigma = getattr(model.config, 'timestep_sigma', 1.0)
        self.data_proportion = getattr(model.config, 'data_proportion', 0.5)
        self.cfg_ratio = getattr(training_config, "cfg_ratio", 0.15)
        self.encoder_train_mode = encoder_train_mode
        
        if hasattr(model, "null_condition_emb"):
            self._null_cond_emb = model.null_condition_emb
        else:
            self._null_cond_emb = None
            
        self.model = model 
        self.force_input_grads_for_checkpointing = False

        for param in self.model.decoder.parameters():
            param.requires_grad = True
            
        if self.encoder_train_mode != "none" and hasattr(self.model, "encoder") and self.model.encoder is not None:
            if self.encoder_train_mode == "all":
                for param in self.model.encoder.parameters():
                    param.requires_grad = True
            elif self.encoder_train_mode == "projectors_only":
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                for name, param in self.model.encoder.named_parameters():
                    if "projector" in name or "projection" in name:
                        param.requires_grad = True
                        
        if train_null_emb and self._null_cond_emb is not None:
            self._null_cond_emb.requires_grad = True

        if getattr(training_config, "gradient_checkpointing", False):
            try:
                if hasattr(self.model.decoder, "gradient_checkpointing_enable"):
                    self.model.decoder.gradient_checkpointing_enable()
                elif hasattr(self.model.decoder, "gradient_checkpointing"):
                    self.model.decoder.gradient_checkpointing = True
                    
                if self.encoder_train_mode != "none" and hasattr(self.model.encoder, "gradient_checkpointing_enable"):
                    self.model.encoder.gradient_checkpointing_enable()
            except: pass

    def training_step(self, batch: dict) -> torch.Tensor:
        global CURRENT_REG_WEIGHT
        if self.device_type in ("cuda", "xpu", "mps"):
            autocast_ctx = torch.autocast(device_type=self.device_type, dtype=self.dtype)
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            nb = self.transfer_non_blocking
            target_latents = batch["target_latents"].to(self.device, dtype=self.dtype, non_blocking=nb)
            attention_mask = batch["attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)
            context_latents = batch["context_latents"].to(self.device, dtype=self.dtype, non_blocking=nb)
            bsz = target_latents.shape[0]

            if self.encoder_train_mode != "none" and "text_hidden_states" in batch and batch["text_hidden_states"].dim() > 1:
                ths = batch["text_hidden_states"].to(self.device, dtype=self.dtype, non_blocking=nb)
                tmask = batch["text_attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)
                lhs = batch["lyric_hidden_states"].to(self.device, dtype=self.dtype, non_blocking=nb)
                lmask = batch["lyric_attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)
                
                ra_hid = torch.zeros(bsz, 1, 64, device=self.device, dtype=self.dtype)
                ra_mask = torch.zeros(bsz, device=self.device, dtype=torch.long)
                
                encoder_hidden_states, encoder_attention_mask = self.model.encoder(
                    text_hidden_states=ths, text_attention_mask=tmask,
                    lyric_hidden_states=lhs, lyric_attention_mask=lmask,
                    refer_audio_acoustic_hidden_states_packed=ra_hid,
                    refer_audio_order_mask=ra_mask
                )
                del ths, tmask, lhs, lmask, ra_hid, ra_mask
            else:
                encoder_hidden_states = batch["encoder_hidden_states"].to(self.device, dtype=self.dtype, non_blocking=nb)
                encoder_attention_mask = batch["encoder_attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)

            if self._null_cond_emb is not None and self.cfg_ratio > 0.0:
                encoder_hidden_states = ts.apply_cfg_dropout(
                    encoder_hidden_states, self._null_cond_emb, cfg_ratio=self.cfg_ratio
                )

            x1 = torch.randn_like(target_latents)
            x0 = target_latents

            t, r = sample_timesteps(
                batch_size=bsz, device=self.device, dtype=self.dtype,
                data_proportion=self.data_proportion, timestep_mu=self.timestep_mu,
                timestep_sigma=self.timestep_sigma, use_meanflow=False
            )
            t_ = t.unsqueeze(-1).unsqueeze(-1)
            xt = t_ * x1 + (1.0 - t_) * x0

            if self.force_input_grads_for_checkpointing:
                xt = xt.requires_grad_(True)

            decoder_outputs = self.model.decoder(
                hidden_states=xt, timestep=t, timestep_r=t,
                attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask, context_latents=context_latents
            )

            flow = x1 - x0
            # Упрощенный расчет MSE для экономии памяти
            diff = decoder_outputs[0] - flow
            loss_per_sample = diff.pow(2).reshape(bsz, -1).mean(dim=1)

            metadata = batch.get("metadata",[])
            weights = torch.ones(bsz, device=self.device, dtype=self.dtype)
            for i in range(bsz):
                meta = metadata[i] if i < len(metadata) else {}
                if meta.get("is_reg", False): weights[i] = CURRENT_REG_WEIGHT

            diffusion_loss = (loss_per_sample * weights).mean()

        # Explicit cleanup to release heavy tensors from graph immediately
        del target_latents, attention_mask, encoder_hidden_states, encoder_attention_mask, context_latents
        del x1, x0, t, r, t_, xt, decoder_outputs, flow, diff, loss_per_sample, weights

        return diffusion_loss.float()

def custom_training_step_lora(self, batch: dict) -> torch.Tensor:
    global CURRENT_REG_WEIGHT
    if self.device_type in ("cuda", "xpu", "mps"):
        autocast_ctx = torch.autocast(device_type=self.device_type, dtype=self.dtype)
    else: autocast_ctx = nullcontext()

    with autocast_ctx:
        nb = self.transfer_non_blocking
        target_latents = batch["target_latents"].to(self.device, dtype=self.dtype, non_blocking=nb)
        attention_mask = batch["attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)
        encoder_hidden_states = batch["encoder_hidden_states"].to(self.device, dtype=self.dtype, non_blocking=nb)
        encoder_attention_mask = batch["encoder_attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)
        context_latents = batch["context_latents"].to(self.device, dtype=self.dtype, non_blocking=nb)

        bsz = target_latents.shape[0]

        if self._null_cond_emb is not None and self._cfg_ratio > 0.0:
            encoder_hidden_states = ts.apply_cfg_dropout(encoder_hidden_states, self._null_cond_emb, cfg_ratio=self._cfg_ratio)

        x1 = torch.randn_like(target_latents)
        x0 = target_latents

        t, r = sample_timesteps(
            batch_size=bsz, device=self.device, dtype=self.dtype,
            data_proportion=self._data_proportion, timestep_mu=self._timestep_mu,
            timestep_sigma=self._timestep_sigma, use_meanflow=False
        )
        t_ = t.unsqueeze(-1).unsqueeze(-1)
        xt = t_ * x1 + (1.0 - t_) * x0

        if self.force_input_grads_for_checkpointing: xt = xt.requires_grad_(True)

        decoder_outputs = self.model.decoder(
            hidden_states=xt, timestep=t, timestep_r=t,
            attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask, context_latents=context_latents
        )
        flow = x1 - x0
        diff = decoder_outputs[0] - flow
        loss_per_sample = diff.pow(2).reshape(bsz, -1).mean(dim=1)

        metadata = batch.get("metadata",[])
        weights = torch.ones(bsz, device=self.device, dtype=self.dtype)
        for i in range(bsz):
            meta = metadata[i] if i < len(metadata) else {}
            if meta.get("is_reg", False): weights[i] = CURRENT_REG_WEIGHT

        diffusion_loss = (loss_per_sample * weights).mean()
        unweighted_loss = loss_per_sample.mean()

    self.training_losses.append(unweighted_loss.item())

    # Explicit cleanup
    del target_latents, attention_mask, encoder_hidden_states, encoder_attention_mask, context_latents
    del x1, x0, t, r, t_, xt, decoder_outputs, flow, diff, loss_per_sample, weights
    
    return diffusion_loss.float()

try: FixedLoRAModule.training_step = custom_training_step_lora
except NameError: pass 


# ======================================================================
# NODES (Config, Estimator, Lora, Trainer, etc.)
# ======================================================================
class ACEStepDatasetConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data_source": ("STRING", {"default": ""}),
                "tensor_root": ("STRING", {"default": "./datasets/preprocessed_tensors"}),
                "output_dir": ("STRING", {"default": "./lora_output/my_run"}),
                "checkpoint_dir": ("STRING", {"default": "./checkpoints"}),
                "resume_from": ("STRING", {"default": ""}),
                "epochs": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 256}),
                "grad_accum": ("INT", {"default": 4, "min": 1, "max": 256}),
                "max_duration": ("FLOAT", {"default": 240.0, "min": 1.0}),
                "save_every": ("INT", {"default": 10, "min": 0}),
                "save_start": ("INT", {"default": 0, "min": 0}),
                "save_loss_limit": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "save_final": ("BOOLEAN", {"default": False}),
                "save_state": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No", "tooltip": "Сохранять training_state.pt (optimizer, scheduler) для возобновления тренировки. Включите только если планируете продолжать тренировку. Экономит ~50-200 МБ на чекпоинт."}),
            },
            "optional": {
                "reg_data_source": ("STRING", {"default": ""}),
                "reg_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("ACESTEP_DATASET",)
    FUNCTION = "get_config"
    CATEGORY = "ACE-Step/Configs"
    def get_config(self, **kwargs): return (kwargs,)

class ACEStepModelConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (["turbo", "base", "sft", "xl_turbo", "xl_base", "xl_sft"], {"default": "turbo"}),
                "vae_variant": (["official", "scragvae"], {"default": "official"}),
                "rank": ("INT", {"default": 64, "min": 1, "max": 1024}),
                "alpha": ("INT", {"default": 128, "min": 1, "max": 2048}),
                "dropout": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target_modules": ("STRING", {"default": "q_proj k_proj v_proj o_proj", "multiline": True}),
                "attn_type": (["both", "self", "cross"], {"default": "both"}),
                "bias": (["none", "lora_only", "all"], {"default": "none"}),
                "inf_steps": ("INT", {"default": 8, "min": 1}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "step": 0.1}),
                "cfg_ratio": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_dora": ("BOOLEAN", {"default": True, "tooltip": "DoRA improves quality but costs VRAM if base_quantization is used."}),
            }
        }
    RETURN_TYPES = ("ACESTEP_MODEL",)
    FUNCTION = "get_config"
    CATEGORY = "ACE-Step/Configs"
    def get_config(self, **kwargs):
        kwargs["adapter_type"] = "lora"
        return (kwargs,)

class ACEStepLoKRConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (["turbo", "base", "sft", "xl_turbo", "xl_base", "xl_sft"], {"default": "turbo"}),
                "vae_variant": (["official", "scragvae"], {"default": "official"}),
                "linear_dim": ("INT", {"default": 64, "min": 1, "max": 1024}),
                "linear_alpha": ("INT", {"default": 128, "min": 1, "max": 2048}),
                "factor": ("INT", {"default": -1, "min": -1, "max": 256}),
                "decompose_both": ("BOOLEAN", {"default": False}),
                "use_tucker": ("BOOLEAN", {"default": False}),
                "use_scalar": ("BOOLEAN", {"default": False}),
                "weight_decompose": ("BOOLEAN", {"default": False}),
                "target_modules": ("STRING", {"default": "q_proj k_proj v_proj o_proj", "multiline": True}),
                "attn_type": (["both", "self", "cross"], {"default": "both"}),
                "inf_steps": ("INT", {"default": 8, "min": 1}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "step": 0.1}),
                "cfg_ratio": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("ACESTEP_MODEL",)
    FUNCTION = "get_config"
    CATEGORY = "ACE-Step/Configs"
    def get_config(self, **kwargs):
        kwargs["adapter_type"] = "lokr"
        return (kwargs,)

def get_base_opt_params(default_lr=1e-4):
    return {
        "scheduler": (["cosine", "cosine_restarts", "linear", "constant"], {"default": "cosine"}),
        "learning_rate": ("FLOAT", {"default": default_lr, "min": 0.0, "max": 10.0, "step": 1e-6, "precision": 6}),
        "weight_decay": ("FLOAT", {"default": 0.01, "min": 0.0, "step": 0.001}),
        "warmup_steps": ("INT", {"default": 100, "min": 0}),
        "max_grad_norm": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.1}),
    }

class ACEStepAdamWConfig:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"optimizer": (["adamw", "adamw8bit"], {"default": "adamw"})}
        inputs.update(get_base_opt_params(1e-4))
        return {"required": inputs}
    RETURN_TYPES = ("ACESTEP_OPTIMIZER",)
    FUNCTION = "get_config"
    CATEGORY = "ACE-Step/Optimizers"
    def get_config(self, **kwargs):
        kwargs["optimizer_kwargs"] = {}
        return (kwargs,)

class ACEStepProdigyConfig:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = get_base_opt_params(1.0)
        inputs.update({
            "d_coef": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
            "d0": ("FLOAT", {"default": 1e-6, "min": 1e-8, "step": 1e-7, "precision": 8}),
            "use_bias_correction": ("BOOLEAN", {"default": False}),
            "safeguard_warmup": ("BOOLEAN", {"default": True}),
        })
        return {"required": inputs}
    RETURN_TYPES = ("ACESTEP_OPTIMIZER",)
    FUNCTION = "get_config"
    CATEGORY = "ACE-Step/Optimizers"
    def get_config(self, **kwargs):
        opt_kwargs = {"d_coef": kwargs.pop("d_coef"), "d0": kwargs.pop("d0"), "use_bias_correction": kwargs.pop("use_bias_correction"), "safeguard_warmup": kwargs.pop("safeguard_warmup")}
        kwargs["optimizer"] = "prodigy"
        kwargs["optimizer_kwargs"] = opt_kwargs
        return (kwargs,)

class ACEStepProdigyPlusConfig:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = get_base_opt_params(1.0)
        inputs.update({
            "d_coef": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
            "d0": ("FLOAT", {"default": 1e-6, "min": 1e-8, "step": 1e-7, "precision": 8}),
            "beta3": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 0.99, "step": 0.01}), 
            "prodigy_steps": ("INT", {"default": 0, "min": 0}),
            "schedulefree_c": ("FLOAT", {"default": 0.0, "min": 0.0}),
            "weight_decay_by_lr": ("BOOLEAN", {"default": True}),
            "factored": ("BOOLEAN", {"default": True}),
            "use_stableadamw": ("BOOLEAN", {"default": True}),
            "use_schedulefree": ("BOOLEAN", {"default": True}),
            "split_groups": ("BOOLEAN", {"default": True}),
            "d_limiter": ("BOOLEAN", {"default": True}),
            "factored_fp32": ("BOOLEAN", {"default": True}),
            "stochastic_rounding": ("BOOLEAN", {"default": True}),
            "use_cautious": ("BOOLEAN", {"default": False}),
            "use_adopt": ("BOOLEAN", {"default": False}),
            "use_grams": ("BOOLEAN", {"default": False}),
            "use_orthograd": ("BOOLEAN", {"default": False}),
            "use_bias_correction": ("BOOLEAN", {"default": False}),
            "use_focus": ("BOOLEAN", {"default": False}),
            "use_speed": ("BOOLEAN", {"default": False}),
            "fused_back_pass": ("BOOLEAN", {"default": False}),
            "split_groups_mean": ("BOOLEAN", {"default": False}),
        })
        return {"required": inputs}
    RETURN_TYPES = ("ACESTEP_OPTIMIZER",)
    FUNCTION = "get_config"
    CATEGORY = "ACE-Step/Optimizers"
    def get_config(self, **kwargs):
        opt_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k not in["scheduler", "learning_rate", "weight_decay", "warmup_steps", "max_grad_norm"]}
        if opt_kwargs.get("beta3", -1.0) < 0: opt_kwargs["beta3"] = None
        kwargs["optimizer"] = "prodigy_plus"
        kwargs["optimizer_kwargs"] = opt_kwargs
        return (kwargs,)

class ACEStepAdemamixConfig:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = get_base_opt_params(1e-4)
        inputs.update({"alpha": ("FLOAT", {"default": 5.0}), "beta3": ("FLOAT", {"default": 0.9})})
        return {"required": inputs}
    RETURN_TYPES = ("ACESTEP_OPTIMIZER",)
    FUNCTION = "get_config"
    CATEGORY = "ACE-Step/Optimizers"
    def get_config(self, **kwargs):
        opt_kwargs = {"ademamix_alpha": kwargs.pop("alpha"), "ademamix_beta3": kwargs.pop("beta3"), "ademamix_t_alpha_beta3": None}
        kwargs["optimizer"] = "ademamix"
        kwargs["optimizer_kwargs"] = opt_kwargs
        return (kwargs,)

class ACEStepAdafactorConfig:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = get_base_opt_params(1e-4)
        inputs.update({"scale_parameter": ("BOOLEAN", {"default": True}), "relative_step": ("BOOLEAN", {"default": True}), "warmup_init": ("BOOLEAN", {"default": True})})
        return {"required": inputs}
    RETURN_TYPES = ("ACESTEP_OPTIMIZER",)
    FUNCTION = "get_config"
    CATEGORY = "ACE-Step/Optimizers"
    def get_config(self, **kwargs):
        opt_kwargs = {"scale_parameter": kwargs.pop("scale_parameter"), "relative_step": kwargs.pop("relative_step"), "warmup_init": kwargs.pop("warmup_init")}
        kwargs["optimizer"] = "adafactor"
        kwargs["optimizer_kwargs"] = opt_kwargs
        return (kwargs,)

class ACEStepPreviewConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gen_preview": ("BOOLEAN", {"default": False}),
                "offload_dit_prev": ("BOOLEAN", {"default": False}),
                "sample_idx": ("INT", {"default": 0, "min": 0}),
                "prev_caption": ("STRING", {"multiline": True, "default": ""}),
                "prev_lyrics": ("STRING", {"multiline": True, "default": ""}),
                "prev_bpm": ("STRING", {"default": ""}),
                "prev_key": ("STRING", {"default": ""}),
                "prev_ts": ("STRING", {"default": ""}),
            },
            "optional": {
                "dataset_config": ("ACESTEP_DATASET",),
            }
        }
    RETURN_TYPES = ("ACESTEP_PREVIEW", "STRING")
    RETURN_NAMES = ("preview_config", "dataset_indices")
    FUNCTION = "get_config"
    CATEGORY = "ACE-Step/Configs"
    
    def get_config(self, **kwargs):
        dataset_config = kwargs.pop("dataset_config", None)
        indices_str = "⚠️ Please connect 'Dataset Config' to see the predicted list of files and indices."
        if dataset_config is not None:
            clean_source = dataset_config.get("data_source", "").strip('"')
            if not clean_source or not os.path.exists(clean_source):
                indices_str = f"⚠️ Data source not found:\n{clean_source}"
            else:
                basenames =[]
                if clean_source.lower().endswith('.json'):
                    try:
                        with open(clean_source, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            samples_list = data.get("samples",[]) if isinstance(data, dict) else (data if isinstance(data, list) else[])
                            for item in samples_list:
                                if isinstance(item, dict):
                                    path_key = next((k for k in["audio_path", "audio", "path", "file", "filename"] if k in item), None)
                                    if path_key and item[path_key]:
                                        basenames.append(os.path.basename(item[path_key]))
                    except Exception as e:
                        indices_str = f"⚠️ Error reading JSON: {e}"
                        return (kwargs, indices_str)
                elif os.path.isdir(clean_source):
                    valid_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.opus'}
                    try:
                        for f in os.listdir(clean_source):
                            if os.path.isfile(os.path.join(clean_source, f)):
                                ext = os.path.splitext(f)[1].lower()
                                if ext in valid_exts: basenames.append(f)
                    except Exception as e:
                        indices_str = f"⚠️ Error reading directory: {e}"
                        return (kwargs, indices_str)
                if not basenames:
                    indices_str = f"⚠️ No valid audio files found"
                else:
                    pt_names = sorted(list(set([os.path.splitext(b)[0] + ".pt" for b in basenames])))
                    lines =[f"🔮 Predicted Training Order (Files: {len(pt_names)})", "="*60]
                    for idx, pt in enumerate(pt_names): lines.append(f"[{idx}] ➔ {pt}")
                    indices_str = "\n".join(lines)
        return (kwargs, indices_str)


class ACEStepEstimator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_config": ("ACESTEP_DATASET",),
                "model_config": ("ACESTEP_MODEL",),
                "est_batches": ("INT", {"default": 5, "min": 1}),
                "top_k": ("INT", {"default": 20, "min": 1}),
                "granularity": (["module", "layer"], {"default": "module"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("target_modules",)
    FUNCTION = "estimate"
    CATEGORY = "ACE-Step/Training"

    def _check_and_prepare_cache(self, source_path, tensor_dir):
        if not os.path.exists(tensor_dir): return False
        pt_files = glob.glob(os.path.join(tensor_dir, "*.pt"))
        if not pt_files: return False
        if not source_path or not os.path.exists(source_path): return True

        oldest_tensor_time = min(os.path.getmtime(f) for f in pt_files)
        source_mtime = 0
        if os.path.isfile(source_path):
            source_mtime = os.path.getmtime(source_path)
        elif os.path.isdir(source_path):
            source_mtime = os.path.getmtime(source_path)
            for root, _, files in os.walk(source_path):
                for file in files:
                    if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.opus')):
                        source_mtime = max(source_mtime, os.path.getmtime(os.path.join(root, file)))

        if source_mtime > oldest_tensor_time:
            print(f"⚠️ [Cache] Source data was modified. Invalidating old tensors...")
            for f in pt_files:
                try: os.remove(f)
                except: pass
            if os.path.exists(os.path.join(tensor_dir, "manifest.json")):
                try: os.remove(os.path.join(tensor_dir, "manifest.json"))
                except: pass
            return False
        return True

    def estimate(self, dataset_config, model_config, est_batches, top_k, granularity):
        print("\n" + "="*60)
        print("🔍 ACE-Step Estimator Started")
        print("="*60)
        
        with torch.enable_grad():
            clean_source = dataset_config["data_source"].strip('"')
            tensor_root = dataset_config["tensor_root"].strip('"')
            checkpoint_dir = dataset_config["checkpoint_dir"].strip('"')
            
            dataset_name = "default_dataset"
            if clean_source:
                dataset_name = os.path.splitext(os.path.basename(os.path.normpath(clean_source)))[0]
                dataset_name = re.sub(r'[\\/*?:"<>|]', "", dataset_name).replace(" ", "_")
            
            final_tensor_dir = os.path.join(tensor_root, dataset_name)
            gpu_info = detect_gpu("auto", "auto") if detect_gpu else None

            if not self._check_and_prepare_cache(clean_source, final_tensor_dir):
                if not clean_source: raise ValueError(f"❌ Tensors not found and no Data Source provided!")
                print(f"🔨 Preprocessing required for estimation...")
                try:
                    is_json = clean_source.lower().endswith('.json')
                    preprocess_audio_files(
                        audio_dir=None if is_json else clean_source,
                        dataset_json=clean_source if is_json else None,
                        output_dir=final_tensor_dir,
                        checkpoint_dir=checkpoint_dir,
                        variant=model_config["model_variant"],
                        vae_variant=model_config.get("vae_variant", "official"),
                        max_duration=dataset_config["max_duration"],
                        device=gpu_info.device if gpu_info else "cuda",
                        precision=gpu_info.precision if gpu_info else "bf16",
                        progress_callback=lambda c,t,m: print(f"[PRE] {m}")
                    )
                except Exception as e:
                    raise RuntimeError(f"❌ Preprocessing failed: {e}")

            print("📊 Running Gradient Estimation...")
            try:
                results = run_estimation(
                    checkpoint_dir=checkpoint_dir,
                    variant=model_config["model_variant"],
                    dataset_dir=final_tensor_dir,
                    num_batches=est_batches,
                    batch_size=dataset_config["batch_size"],
                    top_k=top_k,
                    granularity=granularity,
                    cfg_ratio=model_config["cfg_ratio"]
                )
                
                top_modules = [item["module"].replace("decoder.", "") for item in results]
                result_str = " ".join(top_modules)
                
                print(f"✅ Estimation Complete. Top {top_k} modules found.")
                print(f"📋 Result: {result_str}")
                
                mm.soft_empty_cache()
                return (result_str,)
            except Exception as e:
                raise RuntimeError(f"❌ Estimation failed: {e}")

class ACEStepLoRAResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": ""}),
                "output_path": ("STRING", {"default": ""}),
                "new_rank": ("INT", {"default": 32, "min": 1}),
                "dynamic_method": (["None", "sv_ratio", "sv_fro", "safe"], {"default": "None"}),
                "dynamic_param": ("FLOAT", {"default": 0.9, "step": 0.05}),
                "precision": (["float32", "fp16", "bf16"], {"default": "float32"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "resize"
    CATEGORY = "ACE-Step/Tools"

    def perform_svd(self, weights: torch.Tensor, device: str):
        w_float = weights.to(device).float()
        try: U, S, Vh = torch.linalg.svd(w_float, full_matrices=False)
        except Exception as e:
            print(f"Error during SVD: {e}. Falling back to CPU.")
            U, S, Vh = torch.linalg.svd(w_float.cpu(), full_matrices=False)
        return U, S, Vh

    def calculate_new_rank(self, S: torch.Tensor, method: str, param: float, fixed_rank: int) -> int:
        if method == "None" or method == "safe": return min(fixed_rank, len(S))
        elif method == "sv_ratio":
            keep_indices = torch.nonzero(S >= S[0] * param).flatten()
            return keep_indices[-1].item() + 1 if len(keep_indices) > 0 else 1
        elif method == "sv_fro":
            S_sq = S.pow(2)
            keep_indices = torch.nonzero(torch.cumsum(S_sq, dim=0) >= param * torch.sum(S_sq)).flatten()
            return keep_indices[0].item() + 1 if len(keep_indices) > 0 else len(S)
        return fixed_rank

    def resize(self, input_path, output_path, new_rank, dynamic_method, dynamic_param, precision, device):
        print(f"\n" + "="*60)
        print(f"📉 LoRA Resize Started")
        print(f"="*60)
        
        in_path = input_path.strip('"')
        out_path = output_path.strip('"')

        if not os.path.exists(in_path): raise FileNotFoundError(f"Input path not found: {in_path}")
        if not out_path: out_path = in_path + "_resized"
        
        os.makedirs(out_path, exist_ok=True)
        config_path = os.path.join(in_path, "adapter_config.json")
        model_path = os.path.join(in_path, "adapter_model.safetensors")
        
        use_safetensors = True
        if not os.path.exists(model_path):
             model_path = os.path.join(in_path, "adapter_model.bin")
             use_safetensors = False
             if not os.path.exists(model_path): raise FileNotFoundError("Could not find adapter_model")

        with open(config_path, 'r') as f: config = json.load(f)
        state_dict = load_file(model_path) if use_safetensors else torch.load(model_path, map_location="cpu")

        old_r = config.get("r", 8)
        scaling_factor = config.get("lora_alpha", 8) / old_r if old_r > 0 else 1.0
        
        save_dtype = torch.float32
        if precision == "fp16": save_dtype = torch.float16
        elif precision == "bf16": save_dtype = torch.bfloat16

        pairs = {}
        for key, tensor in state_dict.items():
            if "lora_A" in key:
                base = key.replace("lora_A", "").replace(".weight", "")
                if base not in pairs: pairs[base] = {}
                pairs[base]['A'] = tensor
            elif "lora_B" in key:
                base = key.replace("lora_B", "").replace(".weight", "")
                if base not in pairs: pairs[base] = {}
                pairs[base]['B'] = tensor

        new_state_dict = {}
        rank_pattern = {}
        pbar = ProgressBar(len(pairs))
        print(f"🚀 Processing {len(pairs)} layers...")
        
        for base_key, mats in pairs.items():
            pbar.update(1)
            if 'A' not in mats or 'B' not in mats:
                for k, v in mats.items(): new_state_dict[f"{base_key}lora_{k}.weight"] = v.to(save_dtype)
                continue

            W = (mats['B'].to(device) @ mats['A'].to(device)) * scaling_factor
            U, S, Vh = self.perform_svd(W, device)
            
            target_rank = min(self.calculate_new_rank(S, dynamic_method, dynamic_param, new_rank), old_r)
            if dynamic_method in["sv_fro", "sv_ratio"]: rank_pattern[base_key.replace("base_model.model.", "").rstrip(".")] = target_rank

            U_r, S_r, Vh_r = U[:, :target_rank], S[:target_rank], Vh[:target_rank, :]
            sqrt_S = torch.sqrt(S_r)
            
            new_state_dict[f"{base_key}lora_A.weight"] = (torch.diag(sqrt_S) @ Vh_r).to(save_dtype).cpu()
            new_state_dict[f"{base_key}lora_B.weight"] = (U_r @ torch.diag(sqrt_S)).to(save_dtype).cpu()
            
        import copy
        new_config = copy.deepcopy(config)
        new_config["peft_type"] = "LORA" 
        
        if dynamic_method in ["sv_fro", "sv_ratio"] and rank_pattern:
            new_config["rank_pattern"] = rank_pattern
            new_config["alpha_pattern"] = rank_pattern 
            new_config["r"] = max(rank_pattern.values()) 
            new_config["lora_alpha"] = new_config["r"]
        else:
            new_config["r"] = new_rank
            new_config["lora_alpha"] = new_rank
            new_config.pop("rank_pattern", None)
            new_config.pop("alpha_pattern", None)

        with open(os.path.join(out_path, "adapter_config.json"), 'w') as f: json.dump(new_config, f, indent=2)
        if use_safetensors: save_file(new_state_dict, os.path.join(out_path, "adapter_model.safetensors"))
        else: torch.save(new_state_dict, os.path.join(out_path, "adapter_model.bin"))

        print(f"✅ Resize complete! Saved to {out_path}")
        mm.soft_empty_cache()
        return (out_path,)


class ACEStepTrainer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_config": ("ACESTEP_DATASET",),
                "model_config": ("ACESTEP_MODEL",),
                "optimizer_config": ("ACESTEP_OPTIMIZER",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "grad_ckpt": ("BOOLEAN", {"default": True}),
                "offload_enc": ("BOOLEAN", {"default": False}),
                "vram_cleanup": ("BOOLEAN", {"default": True}),
                "base_quantization": (["none", "fp8", "fp4"], {"default": "none", "tooltip": "Quantize frozen base model/encoders to save VRAM"}),
            },
            "optional": {
                "preview_config": ("ACESTEP_PREVIEW",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_dir",)
    FUNCTION = "train_model"
    CATEGORY = "ACE-Step/Training"

    def _check_and_prepare_cache(self, source_path, tensor_dir):
        if not os.path.exists(tensor_dir): return False
        pt_files = glob.glob(os.path.join(tensor_dir, "*.pt"))
        if not pt_files: return False
        if not source_path or not os.path.exists(source_path): return True

        oldest_tensor_time = min(os.path.getmtime(f) for f in pt_files)
        source_mtime = 0
        if os.path.isfile(source_path):
            source_mtime = os.path.getmtime(source_path)
        elif os.path.isdir(source_path):
            source_mtime = os.path.getmtime(source_path)
            for root, _, files in os.walk(source_path):
                for file in files:
                    if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.opus')):
                        source_mtime = max(source_mtime, os.path.getmtime(os.path.join(root, file)))

        if source_mtime > oldest_tensor_time:
            for f in pt_files:
                try: os.remove(f)
                except: pass
            if os.path.exists(os.path.join(tensor_dir, "manifest.json")):
                try: os.remove(os.path.join(tensor_dir, "manifest.json"))
                except: pass
            return False
        return True

    def process_loss_graph(self, epochs, losses, emas, lrs, elapsed_str="", eta_str="", step_time_str="", epoch_time_str="", saved_epochs=None, node_id=None, output_dir=None):
        if saved_epochs is None: saved_epochs =[]
        fig = Figure(figsize=(4.8, 4.2), dpi=100, facecolor='#2b2b2b')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        ax.grid(True, linestyle='--', color='#444444', alpha=0.3)
        ax.tick_params(colors='#e0e0e0', labelsize=8)
        for spine in ax.spines.values(): spine.set_color('#444444')
        for se in saved_epochs: ax.axvline(x=se, color='#aaaaaa', linestyle=':', linewidth=1.2, alpha=0.7)
            
        ln1 = ax.plot(epochs, losses, color='#ff5252', linewidth=1.0, alpha=0.4, label='Step Loss')
        ln2 = ax.plot(epochs, emas, color='#4caf50', linewidth=2.0, label='EMA Loss')
        
        ax2 = ax.twinx()
        ax2.tick_params(colors='#2196f3', labelsize=8)
        for spine in ax2.spines.values(): spine.set_color('#444444')
        ln3 = ax2.plot(epochs, lrs, color='#2196f3', linestyle=':', linewidth=1.5, alpha=0.8, label='LR')

        last_loss = losses[-1] if losses else 0.0
        last_ema = emas[-1] if emas else 0.0
        last_lr = lrs[-1] if lrs else 0.0
        
        title_lines =[
            f"Loss: {last_loss:.4f} | EMA: {last_ema:.4f} | LR: {last_lr:.2e}",
            f"Time: {elapsed_str} (ETA: {eta_str})",
            f"Speed: {step_time_str} | {epoch_time_str}"
        ]
        ax.set_title("\n".join(title_lines), color='#e0e0e0', fontsize=9, pad=8)
        lns = ln1 + ln2 + ln3
        labs =[l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, facecolor='#2b2b2b', edgecolor='#444444', labelcolor='#e0e0e0', fontsize=8)
        fig.subplots_adjust(bottom=0.25, top=0.82, left=0.12, right=0.88)
        
        canvas = FigureCanvasAgg(fig)
        if node_id is not None:
            buf = BytesIO()
            canvas.print_png(buf)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            PromptServer.instance.send_sync("acestep_loss_update", {"node": node_id, "image": f"data:image/png;base64,{b64}"})
        if output_dir is not None:
            try: canvas.print_png(os.path.join(output_dir, "loss_graph.png"))
            except Exception as e: print(f"⚠️ Failed to save graph to disk: {e}")
        fig.clf()

    @torch.inference_mode(False)
    def train_model(self, dataset_config, model_config, optimizer_config, seed, grad_ckpt, offload_enc, vram_cleanup, base_quantization, preview_config=None, unique_id=None, prompt=None, extra_pnginfo=None):
        with torch.enable_grad():
            print("\n" + "="*60)
            print("🚀 ACE-Step LoRA/LoKR Training Node (ComfyUI)")
            print("="*60)

            clean_source = dataset_config["data_source"].strip('"')
            tensor_root = dataset_config["tensor_root"].strip('"')
            checkpoint_dir = dataset_config["checkpoint_dir"].strip('"')
            output_dir = dataset_config["output_dir"].strip('"')
            reg_source = dataset_config.get("reg_data_source", "").strip('"')
            reg_weight = dataset_config.get("reg_weight", 1.0)
            
            global CURRENT_REG_WEIGHT
            CURRENT_REG_WEIGHT = reg_weight

            os.makedirs(output_dir, exist_ok=True)
            
            # === СОХРАНЯЕМ WORKFLOW СРАЗУ В НАЧАЛЕ ТРЕНИРОВКИ ===
            print(f"🔍 [Workflow] extra_pnginfo keys: {list(extra_pnginfo.keys()) if extra_pnginfo else 'None'}")
            if extra_pnginfo and "workflow" in extra_pnginfo:
                workflow_path = os.path.join(output_dir, "workflow.json")
                print(f"💾 [Workflow] Сохранение workflow в {workflow_path}")
                try:
                    with open(workflow_path, "w", encoding="utf-8") as f:
                        json.dump(extra_pnginfo["workflow"], f, indent=2, ensure_ascii=False)
                    print(f"✅ [Workflow] Workflow успешно сохранён!")
                except Exception as e:
                    print(f"⚠️ [Workflow] Ошибка сохранения workflow: {e}")
            else:
                print("⚠️ [Workflow] Workflow данные не найдены в extra_pnginfo")

            dataset_name = "default_dataset"
            if clean_source:
                dataset_name = os.path.splitext(os.path.basename(os.path.normpath(clean_source)))[0]
                dataset_name = re.sub(r'[\\/*?:"<>|]', "", dataset_name).replace(" ", "_")

            main_tensor_dir = os.path.join(tensor_root, dataset_name)
            gpu_info = detect_gpu("auto", "auto") if detect_gpu else None
            device = gpu_info.device if gpu_info else "cuda"
            precision = gpu_info.precision if gpu_info else "bf16"

            if not self._check_and_prepare_cache(clean_source, main_tensor_dir):
                if not clean_source: raise ValueError(f"❌ Tensors not found and no Data Source provided!")
                print(f"🔨[Phase 1] Preprocessing Main Dataset...")
                is_json = clean_source.lower().endswith('.json')
                preprocess_audio_files(
                    audio_dir=None if is_json else clean_source,
                    dataset_json=clean_source if is_json else None,
                    output_dir=main_tensor_dir,
                    checkpoint_dir=checkpoint_dir,
                    variant=model_config["model_variant"],
                    vae_variant=model_config.get("vae_variant", "official"),
                    max_duration=dataset_config["max_duration"],
                    device=device,
                    precision=precision,
                    progress_callback=lambda c,t,m: print(f"[PRE-MAIN] {m}")
                )
            else:
                print(f"📂[Phase 1] Valid cache found. Using existing main tensors in: {main_tensor_dir}")

            reg_tensor_dir = None
            if reg_source and os.path.exists(reg_source):
                reg_dataset_name = os.path.splitext(os.path.basename(os.path.normpath(reg_source)))[0]
                reg_dataset_name = re.sub(r'[\\/*?:"<>|]', "", reg_dataset_name).replace(" ", "_")
                reg_tensor_dir = os.path.join(tensor_root, reg_dataset_name + "_REG")

                if not self._check_and_prepare_cache(reg_source, reg_tensor_dir):
                    print(f"🔨[Phase 1.5] Preprocessing Regularization Data...")
                    is_json = reg_source.lower().endswith('.json')
                    preprocess_audio_files(
                        audio_dir=None if is_json else reg_source,
                        dataset_json=reg_source if is_json else None,
                        output_dir=reg_tensor_dir,
                        checkpoint_dir=checkpoint_dir,
                        variant=model_config["model_variant"],
                        vae_variant=model_config.get("vae_variant", "official"),
                        max_duration=dataset_config["max_duration"],
                        device=device,
                        precision=precision,
                        progress_callback=lambda c,t,m: print(f"[PRE-REG] {m}")
                    )
                else:
                    print(f"📂[Phase 1.5] Valid cache found. Using existing REG tensors in: {reg_tensor_dir}")

            dataset_len = len(glob.glob(os.path.join(main_tensor_dir, "*.pt")))
            if dataset_len == 0: raise ValueError("❌ No .pt files found! Cannot proceed.")

            resolved_modules = resolve_target_modules(model_config["target_modules"].split(), model_config["attn_type"])
            adapter_type = model_config.get("adapter_type", "lora")

            if adapter_type == "lokr":
                adapter_cfg = LoKRConfigV2(
                    linear_dim=model_config["linear_dim"], linear_alpha=model_config["linear_alpha"],
                    factor=model_config["factor"], decompose_both=model_config["decompose_both"],
                    use_tucker=model_config["use_tucker"], use_scalar=model_config["use_scalar"],
                    weight_decompose=model_config["weight_decompose"], target_modules=resolved_modules,
                    attention_type=model_config["attn_type"]
                )
            else:
                adapter_cfg = LoRAConfigV2(
                    r=model_config["rank"], alpha=model_config["alpha"], dropout=model_config["dropout"], 
                    target_modules=resolved_modules, bias=model_config["bias"], attention_type=model_config["attn_type"],
                    use_dora=model_config.get("use_dora", True)
                )

            if base_quantization in ["fp4", "fp8"] and model_config.get("use_dora", True):
                print("\n⚠️ [VRAM WARNING] 'use_dora=True' combined with FP4/FP8 quantization forces PEFT to materialize 16-bit base weights during the forward pass! This significantly reduces VRAM savings. Disable 'use_dora' for maximum memory efficiency.\n")

            prev_cfg = preview_config if preview_config is not None else {}
            
            train_cfg = TrainingConfigV2(
                adapter_type=adapter_type, learning_rate=optimizer_config["learning_rate"], 
                batch_size=dataset_config["batch_size"], gradient_accumulation_steps=dataset_config["grad_accum"],
                max_epochs=dataset_config["epochs"], save_every_n_epochs=dataset_config["save_every"], 
                output_dir=output_dir, seed=seed, resume_from=dataset_config["resume_from"] if dataset_config["resume_from"] else None, 
                optimizer_type=optimizer_config["optimizer"], scheduler_type=optimizer_config["scheduler"],
                gradient_checkpointing=grad_ckpt, offload_encoder=offload_enc, cfg_ratio=model_config["cfg_ratio"],
                device=device, precision=precision, dataset_dir=main_tensor_dir,
                checkpoint_dir=checkpoint_dir, model_variant=model_config["model_variant"], 
                num_workers=0 if os.name == 'nt' else 4, log_every=1, log_heavy_every=10000, 
                weight_decay=optimizer_config["weight_decay"], max_grad_norm=optimizer_config["max_grad_norm"],
                warmup_steps=optimizer_config["warmup_steps"], save_start_epoch=dataset_config["save_start"], 
                save_loss_threshold=dataset_config["save_loss_limit"], optimizer_kwargs=optimizer_config["optimizer_kwargs"], 
                shift=model_config["shift"], num_inference_steps=model_config["inf_steps"],
                generate_preview=prev_cfg.get("gen_preview", False), offload_dit_for_preview=prev_cfg.get("offload_dit_prev", False), 
                preview_sample_index=prev_cfg.get("sample_idx", 0), preview_caption=prev_cfg.get("prev_caption", "").strip() or None, 
                preview_lyrics=prev_cfg.get("prev_lyrics", "").strip() or None, preview_bpm=prev_cfg.get("prev_bpm", "").strip() or None, 
                preview_keyscale=prev_cfg.get("prev_key", "").strip() or None, preview_timesig=prev_cfg.get("prev_ts", "").strip() or None,
            )
            train_cfg.save_state = dataset_config.get("save_state", True)
            train_cfg.save_final_model = dataset_config.get("save_final", False)
            
            # Информируем о режиме сохранения training_state
            if train_cfg.save_state:
                print("💾 [Save State] ✅ Сохранение training_state.pt ВКЛЮЧЕНО (можно возобновить тренировку)")
            else:
                print("💾 [Save State] ⏭️ Сохранение training_state.pt ОТКЛЮЧЕНО (экономия места, только адаптеры)")

            print(f"🧠[Phase 2] Loading {model_config['model_variant']} model on {device} ({precision})...")
            model = load_decoder_for_training(checkpoint_dir=checkpoint_dir, variant=model_config["model_variant"], device=device, precision=precision)

            if vram_cleanup:
                print(f"🧹[System] Aggressive VRAM Cleanup...")
                to_kill =["vae", "text_encoder", "tokenizer", "detokenizer", "music_encoder", "lyric_encoder", "timbre_encoder", "condition_projection"]
                for attr in to_kill:
                    if hasattr(model, attr):
                        m = getattr(model, attr)
                        if m is not None: setattr(model, attr, m.to("cpu"))
                    if hasattr(model, "encoder") and hasattr(model.encoder, attr):
                        m = getattr(model.encoder, attr)
                        if m is not None: setattr(model.encoder, attr, m.to("cpu"))
                if hasattr(model, "encoder") and hasattr(model.encoder, "text_projector"):
                    model.encoder.text_projector.to("cpu")

                model.decoder.to("cpu")
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.synchronize()
                
                model.decoder.to(device)
                if hasattr(model, "null_condition_emb") and model.null_condition_emb is not None:
                    model.null_condition_emb.data = model.null_condition_emb.data.to(device)

            model.train()

            print("\n🔥 Starting Training Loop...")
            trainer = FixedLoRATrainer(model, adapter_cfg, train_cfg)
            trainer.main_tensor_dir = main_tensor_dir

            # === ВСТРАИВАЕМ WORKFLOW В SAFETENSORS ===
            if extra_pnginfo and "workflow" in extra_pnginfo:
                try:
                    trainer.workflow_json = json.dumps(extra_pnginfo["workflow"], ensure_ascii=False)
                    print("✅ [Workflow] Workflow JSON передан тренеру для встраивания в safetensors")
                except Exception as e:
                    print(f"⚠️ [Workflow] Ошибка сериализации workflow: {e}")
            
            import acestep.training.data_module as dm_module
            original_setup = dm_module.PreprocessedDataModule.setup

            def balanced_setup(self_dm, stage=None):
                main_ds = dm_module.PreprocessedTensorDataset(main_tensor_dir)
                if reg_tensor_dir and os.path.exists(reg_tensor_dir):
                    reg_ds = dm_module.PreprocessedTensorDataset(reg_tensor_dir)
                    class BalancedWrapper(torch.utils.data.Dataset):
                        def __init__(self, m_ds, r_ds):
                            self.m_ds = m_ds; self.r_ds = r_ds; self.target_len = max(len(m_ds), len(r_ds))
                        def __len__(self): return self.target_len * 2
                        def __getitem__(self, idx):
                            if idx % 2 == 0: return self.m_ds[(idx // 2) % len(self.m_ds)]
                            else:
                                item = dict(self.r_ds[(idx // 2) % len(self.r_ds)])
                                item["metadata"] = dict(item.get("metadata", {}))
                                item["metadata"]["is_reg"] = True
                                return item
                    self_dm.train_dataset = BalancedWrapper(main_ds, reg_ds)
                    print(f"⚖️ In-Memory Dataset Balanced: Main ({len(main_ds)}) | Reg ({len(reg_ds)}) -> Total Epoch Size: {len(self_dm.train_dataset)}")
                else: self_dm.train_dataset = main_ds
                self_dm.val_dataset = None

            dm_module.PreprocessedDataModule.setup = balanced_setup

            actual_dataset_len = dataset_len
            if reg_tensor_dir and os.path.exists(reg_tensor_dir):
                actual_dataset_len = max(dataset_len, len(glob.glob(os.path.join(reg_tensor_dir, "*.pt")))) * 2

            eff_batch = max(1, dataset_config["batch_size"] * dataset_config["grad_accum"])
            steps_per_epoch = max(1, actual_dataset_len // eff_batch)
            total_steps_approx = steps_per_epoch * dataset_config["epochs"]

            pbar = ProgressBar(total_steps_approx)
            training_state = {"should_stop": False}
            start_time = time.time()

            epoch_history, loss_history, ema_history, lr_history = [], [], [],[]
            ema_loss, ema_alpha, saved_epochs = None, 0.1,[]
            elapsed_str, eta_str, step_time_str, epoch_time_str = "00:00:00", "00:00:00", "0s/it", "0s/ep"

            def fmt_time(secs):
                m, s = divmod(int(max(0, secs)), 60)
                h, m = divmod(m, 60)
                return f"{h:02d}:{m:02d}:{s:02d}"

            is_first_yield = True

            try:
                for update in trainer.train(training_state):
                    # В этот момент trainer.module уже создан и PEFT внедрил LoRA адаптеры!
                    if is_first_yield and trainer.module is not None:
                        print("🪄 [Phase 3] Applying VRAM optimizations (FP8/NF4) to wrapped base layers...")
                        target_dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else torch.float32)
                        apply_quantization_base_model(trainer.module.model, target_dtype, base_quantization, seed)
                        is_first_yield = False

                    if mm.processing_interrupted():
                        print("\n🛑 Training Interrupted by User via ComfyUI!")
                        training_state["should_stop"] = True
                        break

                    if update.kind == "step":
                        pbar.update(1)
                        loss = update.loss
                        current_lr = getattr(update, "lr", 0.0)
                        
                        if ema_loss is None: ema_loss = loss
                        else: ema_loss = ema_alpha * loss + (1 - ema_alpha) * ema_loss

                        current_epoch = update.step / steps_per_epoch
                        epoch_history.append(current_epoch)
                        loss_history.append(loss)
                        ema_history.append(ema_loss)
                        lr_history.append(current_lr)

                        elapsed = time.time() - start_time
                        if update.step > 0:
                            time_per_step = elapsed / update.step
                            remaining_steps = total_steps_approx - update.step
                            eta_secs = remaining_steps * time_per_step
                            step_time_str = f"{int(time_per_step * 1000)}ms/it" if time_per_step < 1.0 else f"{time_per_step:.2f}s/it"
                            time_per_epoch = time_per_step * steps_per_epoch
                            ep_m, ep_s = divmod(int(time_per_epoch), 60)
                            ep_h, ep_m = divmod(ep_m, 60)
                            epoch_time_str = f"{ep_h}h {ep_m}m/ep" if ep_h > 0 else (f"{ep_m}m {ep_s}s/ep" if ep_m > 0 else f"{ep_s}s/ep")
                        else:
                            eta_secs = 0
                            
                        elapsed_str = fmt_time(elapsed)
                        eta_str = fmt_time(eta_secs)

                        if update.step % 5 == 0 and unique_id is not None:
                            try:
                                node_id_str = unique_id[0] if isinstance(unique_id, list) else str(unique_id)
                                self.process_loss_graph(
                                    epoch_history, loss_history, ema_history, lr_history,
                                    elapsed_str, eta_str, step_time_str, epoch_time_str,
                                    saved_epochs, node_id=node_id_str
                                )
                            except Exception: pass
                            
                    elif update.kind == "epoch":
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    if update.msg:
                        if not (("Step" in update.msg and "Loss" in update.msg) or ("Epoch" in update.msg and "Loss" in update.msg)):
                            print(f"[LOG] {update.msg}")
                        msg_lower = update.msg.lower()
                        if "save" in msg_lower or "saving" in msg_lower or "saved" in msg_lower:
                            if epoch_history:
                                last_ep = epoch_history[-1]
                                if not saved_epochs or abs(saved_epochs[-1] - last_ep) > 0.05:
                                    saved_epochs.append(last_ep)
            finally:
                dm_module.PreprocessedDataModule.setup = original_setup
                if loss_history:
                    print(f"📊 Saving final training graph to: {output_dir}/loss_graph.png")
                    self.process_loss_graph(
                        epoch_history, loss_history, ema_history, lr_history,
                        elapsed_str, eta_str, step_time_str, epoch_time_str,
                        saved_epochs, node_id=None, output_dir=output_dir
                    )

            elapsed = time.time() - start_time
            print(f"\n🎉 Finished in {fmt_time(elapsed)}")
            mm.soft_empty_cache()
            return (output_dir,)


class ACEStepFinetuneTrainer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_config": ("ACESTEP_DATASET",),
                "model_variant": (["turbo", "base", "sft", "xl_turbo", "xl_base", "xl_sft"], {"default": "turbo"}),
                "vae_variant": (["official", "scragvae"], {"default": "official"}),
                "cfg_ratio": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "optimizer_config": ("ACESTEP_OPTIMIZER",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "grad_ckpt": ("BOOLEAN", {"default": True}),
                "offload_enc": ("BOOLEAN", {"default": False}),
                "vram_cleanup": ("BOOLEAN", {"default": True}),
                "block_swap_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Offload layers to CPU. 1.0 = Max VRAM saving."}),
                "encoder_train_mode": (["none", "projectors_only", "all"], {"default": "none", "tooltip": "none=frozen, projectors_only=low VRAM impact, all=OOM on 16GB unless adamw8bit"}),
                "train_null_emb": ("BOOLEAN", {"default": True, "tooltip": "Train null_condition_emb (CFG)"}),
                "base_quantization": (["none", "fp8", "fp4"], {"default": "none", "tooltip": "Quantize frozen encoders to save VRAM"}),
            },
            "optional": {
                "preview_config": ("ACESTEP_PREVIEW",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_dir",)
    FUNCTION = "train_finetune"
    CATEGORY = "ACE-Step/Training"

    def _check_and_prepare_cache(self, source_path, tensor_dir):
        if not os.path.exists(tensor_dir): return False
        pt_files = glob.glob(os.path.join(tensor_dir, "*.pt"))
        if not pt_files: return False
        if not source_path or not os.path.exists(source_path): return True

        oldest_tensor_time = min(os.path.getmtime(f) for f in pt_files)
        source_mtime = 0
        if os.path.isfile(source_path):
            source_mtime = os.path.getmtime(source_path)
        elif os.path.isdir(source_path):
            source_mtime = os.path.getmtime(source_path)
            for root, _, files in os.walk(source_path):
                for file in files:
                    if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.opus')):
                        source_mtime = max(source_mtime, os.path.getmtime(os.path.join(root, file)))

        if source_mtime > oldest_tensor_time:
            print(f"⚠️ [Cache] Source data was modified. Invalidating old tensors...")
            for f in pt_files:
                try: os.remove(f)
                except: pass
            if os.path.exists(os.path.join(tensor_dir, "manifest.json")):
                try: os.remove(os.path.join(tensor_dir, "manifest.json"))
                except: pass
            return False
        return True

    def process_loss_graph(self, epochs, losses, emas, lrs, elapsed_str="", eta_str="", step_time_str="", epoch_time_str="", saved_epochs=None, node_id=None, output_dir=None):
        if saved_epochs is None: saved_epochs =[]
        fig = Figure(figsize=(4.8, 4.2), dpi=100, facecolor='#2b2b2b')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        ax.grid(True, linestyle='--', color='#444444', alpha=0.3)
        ax.tick_params(colors='#e0e0e0', labelsize=8)
        for spine in ax.spines.values(): spine.set_color('#444444')
        
        for se in saved_epochs: ax.axvline(x=se, color='#aaaaaa', linestyle=':', linewidth=1.2, alpha=0.7)
            
        ln1 = ax.plot(epochs, losses, color='#ff5252', linewidth=1.0, alpha=0.4, label='Step Loss')
        ln2 = ax.plot(epochs, emas, color='#4caf50', linewidth=2.0, label='EMA Loss')
        
        ax2 = ax.twinx()
        ax2.tick_params(colors='#2196f3', labelsize=8)
        for spine in ax2.spines.values(): spine.set_color('#444444')
        ln3 = ax2.plot(epochs, lrs, color='#2196f3', linestyle=':', linewidth=1.5, alpha=0.8, label='LR')

        last_loss = losses[-1] if losses else 0.0
        last_ema = emas[-1] if emas else 0.0
        last_lr = lrs[-1] if lrs else 0.0
        
        title_lines =[
            f"Loss: {last_loss:.4f} | EMA: {last_ema:.4f} | LR: {last_lr:.2e}",
            f"Time: {elapsed_str} (ETA: {eta_str})",
            f"Speed: {step_time_str} | {epoch_time_str}"
        ]
        ax.set_title("\n".join(title_lines), color='#e0e0e0', fontsize=9, pad=8)
        lns = ln1 + ln2 + ln3
        labs =[l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, facecolor='#2b2b2b', edgecolor='#444444', labelcolor='#e0e0e0', fontsize=8)
        fig.subplots_adjust(bottom=0.25, top=0.82, left=0.12, right=0.88)
        
        canvas = FigureCanvasAgg(fig)
        if node_id is not None:
            buf = BytesIO()
            canvas.print_png(buf)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            PromptServer.instance.send_sync("acestep_loss_update", {"node": node_id, "image": f"data:image/png;base64,{b64}"})
        if output_dir is not None:
            try: canvas.print_png(os.path.join(output_dir, "loss_graph.png"))
            except Exception as e: print(f"⚠️ Failed to save graph to disk: {e}")
        fig.clf()

    def generate_preview(self, model, preview_config, checkpoint_dir, output_dir, step, device, precision, variant, vae_variant, main_tensor_dir, offload_enc=False, vram_cleanup=False, encoder_train_mode="none"):
        if not preview_config.get("gen_preview", False): return
            
        print(f"[LOG] 🎵 Generating preview (Step: {step})...")
        do_offload_dit = preview_config.get("offload_dit_prev", False)
        
        try:
            import torchaudio
            from acestep.training_v2.model_loader import load_vae, load_text_encoder, load_silence_latent, unload_models
            from acestep.core.generation.handler.vae_decode import VaeDecodeMixin
            from acestep.core.generation.handler.vae_decode_chunks import VaeDecodeChunksMixin
            from acestep.core.generation.handler.memory_utils import MemoryUtilsMixin
            
            target_idx = preview_config.get("sample_idx", 0)
            pt_files = sorted(glob.glob(os.path.join(main_tensor_dir, "*.pt")))
            pt_files =[f for f in pt_files if not f.endswith("manifest.json")]
            
            caption, lyrics, tag, pos = "Music", "[Instrumental]", "", "prepend"
            ds_bpm, ds_key, ds_ts = "N/A", "N/A", "N/A"
            
            if pt_files:
                chosen_file = pt_files[target_idx % len(pt_files)]
                try:
                    sample_data = torch.load(chosen_file, map_location="cpu", weights_only=False)
                    meta = sample_data.get("metadata", {})
                    caption = meta.get("caption", "Music")
                    lyrics = meta.get("lyrics", "[Instrumental]")
                    tag = meta.get("custom_tag", "")
                    pos = meta.get("tag_position", "prepend")
                    ds_bpm = meta.get("bpm", "N/A")
                    ds_key = meta.get("keyscale", "N/A")
                    ds_ts = meta.get("timesignature", "N/A")
                except Exception as e:
                    print(f"⚠️ Failed to load sample {chosen_file}: {e}")

            final_caption = preview_config.get("prev_caption", "").strip() or caption
            final_lyrics = preview_config.get("prev_lyrics", "").strip() or lyrics
            
            if not preview_config.get("prev_caption", "").strip() and tag:
                if pos == "prepend": test_prompt_raw = f"{tag}, {final_caption}" if final_caption else tag
                elif pos == "append": test_prompt_raw = f"{final_caption}, {tag}" if final_caption else tag
                elif pos == "replace": test_prompt_raw = tag
                else: test_prompt_raw = final_caption
            else:
                test_prompt_raw = final_caption

            final_bpm = preview_config.get("prev_bpm", "").strip() or str(ds_bpm)
            final_key = preview_config.get("prev_key", "").strip() or str(ds_key)
            final_ts = preview_config.get("prev_ts", "").strip() or str(ds_ts)

            meta_str = f"- bpm: {final_bpm}\n- timesignature: {final_ts}\n- keyscale: {final_key}\n- duration: 30 seconds\n"
            full_text_prompt = f"# Instruction\nFill the audio semantic mask based on the given conditions:\n\n# Caption\n{test_prompt_raw}\n\n# Metas\n{meta_str}<|endoftext|>"
            full_lyrics_prompt = f"# Languages\nunknown\n\n# Lyric\n{final_lyrics}<|endoftext|>"

            if do_offload_dit:
                model.to("cpu")
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            tokenizer, text_enc = load_text_encoder(checkpoint_dir, device=device, precision=precision)
            silence_lat = load_silence_latent(checkpoint_dir, device=device, precision=precision, variant=variant)
            dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else torch.float32)

            with torch.no_grad():
                text_inputs = tokenizer(full_text_prompt, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
                text_hs = text_enc(text_inputs.input_ids.to(device)).last_hidden_state.to(dtype)
                text_mask = text_inputs.attention_mask.to(device).to(dtype)
                
                lyric_inputs = tokenizer(full_lyrics_prompt, padding="max_length", max_length=2048, truncation=True, return_tensors="pt")
                lyric_hs = text_enc.embed_tokens(lyric_inputs.input_ids.to(device)).to(dtype)
                lyric_mask = lyric_inputs.attention_mask.to(device).to(dtype)

            unload_models(text_enc)
            model.to(device)
            model.eval()

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

            is_turbo = "turbo" in variant.lower()
            inf_steps = preview_config.get("inf_steps", 8 if is_turbo else 50)
            shift = preview_config.get("shift", 3.0 if is_turbo else 1.0)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=dtype):
                    outputs = model.generate_audio(
                        text_hidden_states=text_hs, text_attention_mask=text_mask,
                        lyric_hidden_states=lyric_hs, lyric_attention_mask=lyric_mask,
                        refer_audio_acoustic_hidden_states_packed=refer_audio,
                        refer_audio_order_mask=refer_mask, src_latents=src_latents,
                        chunk_masks=chunk_masks, silence_latent=current_silence,
                        attention_mask=attention_mask, is_covers=is_covers,
                        infer_steps=inf_steps, diffusion_guidance_sale=7.5,
                        shift=shift, use_cache=True, infer_method="ode", use_progress_bar=False
                    )
            generated_latents = outputs["target_latents"]

            if do_offload_dit:
                model.to("cpu")
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            vae = load_vae(checkpoint_dir, device=device, precision=precision, vae_variant=vae_variant)
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
                def _get_auto_decode_chunk_size(self): return 256 
                def _should_offload_wav_to_cpu(self): return False

            vae_wrapper = VaeInferenceWrapper(vae, str(device))
            latents_to_decode = generated_latents.to(device).to(vae.dtype).transpose(1, 2)
            
            with torch.no_grad():
                audio = vae_wrapper.tiled_decode(latents_to_decode, chunk_size=256, overlap=32)
            
            preview_dir = os.path.join(output_dir, "previews")
            os.makedirs(preview_dir, exist_ok=True)
            save_path = os.path.join(preview_dir, f"sample_epoch_{step}_idx{target_idx}.mp3")
            
            audio_data = audio[0].detach().cpu().float()
            max_val = audio_data.abs().max()
            if max_val > 0: audio_data = audio_data / max_val * 0.95
            
            try:
                torchaudio.save(save_path, audio_data, 48000, format="mp3")
                print(f"💾 Saved preview to: {save_path}")
            except Exception as e:
                save_path_wav = save_path.replace(".mp3", ".wav")
                torchaudio.save(save_path_wav, audio_data, 48000)
                print(f"💾 Saved preview to: {save_path_wav}")

        except Exception as e:
            print(f"❌ Preview generation failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try: unload_models(vae)
            except: pass
            
            model.to(device)
            if offload_enc or vram_cleanup:
                to_kill = ["vae", "text_encoder", "tokenizer", "detokenizer"]
                if encoder_train_mode == "none":
                    to_kill.extend(["music_encoder", "lyric_encoder", "timbre_encoder", "condition_projection"])
                
                for attr in to_kill:
                    if hasattr(model, attr):
                        m = getattr(model, attr)
                        if m is not None: setattr(model, attr, m.to("cpu"))
                    if hasattr(model, "encoder") and hasattr(model.encoder, attr):
                        m = getattr(model.encoder, attr)
                        if m is not None: setattr(model.encoder, attr, m.to("cpu"))
                        
                if encoder_train_mode == "none" and hasattr(model, "encoder") and hasattr(model.encoder, "text_projector"):
                    model.encoder.text_projector.to("cpu")

                if offload_enc and encoder_train_mode == "none" and hasattr(model, "encoder") and model.encoder is not None:
                    model.encoder.to("cpu")

            model.train()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    @torch.inference_mode(False)
    def train_finetune(self, dataset_config, model_variant, vae_variant, cfg_ratio, optimizer_config, seed, grad_ckpt, offload_enc, vram_cleanup, block_swap_ratio, encoder_train_mode, train_null_emb, base_quantization, preview_config=None, unique_id=None, prompt=None, extra_pnginfo=None):
        global CURRENT_REG_WEIGHT

        with torch.enable_grad():
            print("\n" + "="*60)
            print("🚀 ACE-Step Full Fine-Tuning Node (ComfyUI)")
            print("="*60)

            clean_source = dataset_config["data_source"].strip('"')
            tensor_root = dataset_config["tensor_root"].strip('"')
            checkpoint_dir = dataset_config["checkpoint_dir"].strip('"')
            output_dir = dataset_config["output_dir"].strip('"')
            reg_source = dataset_config.get("reg_data_source", "").strip('"')
            reg_weight = dataset_config.get("reg_weight", 1.0)
            CURRENT_REG_WEIGHT = reg_weight

            if encoder_train_mode != "none":
                offload_enc = False
                print(f"⚠️ Note: encoder_train_mode is '{encoder_train_mode}'. Encoders will NOT be offloaded.")
                if encoder_train_mode == "all" and optimizer_config["optimizer"] == "adamw":
                    print("⚠️ DANGER: You are training ALL encoder layers with standard AdamW!")
                    print("⚠️ Expect OOM on 16GB GPU. Use AdamW8bit or Adafactor, or set to 'projectors_only'.")

            os.makedirs(output_dir, exist_ok=True)
            
            # === СОХРАНЯЕМ WORKFLOW СРАЗУ В НАЧАЛЕ ТРЕНИРОВКИ ===
            print(f"🔍 [Workflow] extra_pnginfo keys: {list(extra_pnginfo.keys()) if extra_pnginfo else 'None'}")
            if extra_pnginfo and "workflow" in extra_pnginfo:
                workflow_path = os.path.join(output_dir, "workflow.json")
                print(f"💾 [Workflow] Сохранение workflow в {workflow_path}")
                try:
                    with open(workflow_path, "w", encoding="utf-8") as f:
                        json.dump(extra_pnginfo["workflow"], f, indent=2, ensure_ascii=False)
                    print(f"✅ [Workflow] Workflow успешно сохранён!")
                except Exception as e:
                    print(f"⚠️ [Workflow] Ошибка сохранения workflow: {e}")
            else:
                print("⚠️ [Workflow] Workflow данные не найдены в extra_pnginfo")

            dataset_name = "default_dataset"
            if clean_source:
                dataset_name = os.path.splitext(os.path.basename(os.path.normpath(clean_source)))[0]
                dataset_name = re.sub(r'[\\/*?:"<>|]', "", dataset_name).replace(" ", "_")

            main_tensor_dir = os.path.join(tensor_root, dataset_name)

            gpu_info = None
            try:
                from acestep.training_v2.gpu_utils import detect_gpu
                gpu_info = detect_gpu("auto", "auto")
            except: pass
            
            device = gpu_info.device if gpu_info else ("cuda" if torch.cuda.is_available() else "cpu")
            precision = gpu_info.precision if gpu_info else ("bf16" if "cuda" in device else "fp32")

            # ===[PHASE 1] PREPROCESSING ===
            if not self._check_and_prepare_cache(clean_source, main_tensor_dir):
                if not clean_source: raise ValueError(f"❌ Tensors not found and no Data Source provided!")
                print(f"🔨[Phase 1] Preprocessing Main Dataset...")
                try:
                    is_json = clean_source.lower().endswith('.json')
                    from acestep.training_v2.preprocess import preprocess_audio_files
                    preprocess_audio_files(
                        audio_dir=None if is_json else clean_source,
                        dataset_json=clean_source if is_json else None,
                        output_dir=main_tensor_dir,
                        checkpoint_dir=checkpoint_dir,
                        variant=model_variant,
                        vae_variant=vae_variant,
                        max_duration=dataset_config["max_duration"],
                        device=device,
                        precision=precision,
                        progress_callback=lambda c,t,m: print(f"[PRE-MAIN] {m}")
                    )
                except Exception as e:
                    raise RuntimeError(f"❌ Main Preprocessing failed: {e}")
            else:
                print(f"📂[Phase 1] Valid cache found. Using existing main tensors in: {main_tensor_dir}")

            reg_tensor_dir = None
            if reg_source and os.path.exists(reg_source):
                reg_dataset_name = os.path.splitext(os.path.basename(os.path.normpath(reg_source)))[0]
                reg_dataset_name = re.sub(r'[\\/*?:"<>|]', "", reg_dataset_name).replace(" ", "_")
                reg_tensor_dir = os.path.join(tensor_root, reg_dataset_name + "_REG")

                if not self._check_and_prepare_cache(reg_source, reg_tensor_dir):
                    print(f"🔨[Phase 1.5] Preprocessing Regularization Data...")
                    is_json = reg_source.lower().endswith('.json')
                    preprocess_audio_files(
                        audio_dir=None if is_json else reg_source,
                        dataset_json=reg_source if is_json else None,
                        output_dir=reg_tensor_dir,
                        checkpoint_dir=checkpoint_dir,
                        variant=model_variant,
                        vae_variant=vae_variant,
                        max_duration=dataset_config["max_duration"],
                        device=device,
                        precision=precision,
                        progress_callback=lambda c,t,m: print(f"[PRE-REG] {m}")
                    )
                else:
                    print(f"📂[Phase 1.5] Valid cache found. Using existing REG tensors in: {reg_tensor_dir}")

            tensor_files = glob.glob(os.path.join(main_tensor_dir, "*.pt"))
            dataset_len = len(tensor_files)
            if dataset_len == 0: raise ValueError("❌ No .pt files found! Cannot proceed.")

            from acestep.training_v2.configs import TrainingConfigV2
            training_cfg = TrainingConfigV2(
                learning_rate=optimizer_config["learning_rate"],
                batch_size=dataset_config["batch_size"],
                gradient_accumulation_steps=dataset_config["grad_accum"],
                max_epochs=dataset_config["epochs"],
                save_every_n_epochs=dataset_config["save_every"],
                save_start_epoch=dataset_config["save_start"],
                save_loss_threshold=dataset_config["save_loss_limit"],
                weight_decay=optimizer_config["weight_decay"],
                max_grad_norm=optimizer_config["max_grad_norm"],
                gradient_checkpointing=grad_ckpt,
                offload_encoder=offload_enc,
                seed=seed,
                output_dir=output_dir,
                optimizer_type=optimizer_config["optimizer"],
                optimizer_kwargs=optimizer_config.get("optimizer_kwargs", {}),
                scheduler_type=optimizer_config["scheduler"],
                warmup_steps=optimizer_config["warmup_steps"],
                num_workers=0 if os.name == 'nt' else 4,
                cfg_ratio=cfg_ratio,
                checkpoint_dir=checkpoint_dir,
                model_variant=model_variant,
                dataset_dir=main_tensor_dir
            )
            
            # Информируем о режиме сохранения training_state
            finetune_save_state = dataset_config.get("save_state", False)
            if finetune_save_state:
                print("💾 [Save State] ✅ Сохранение training_state.pt ВКЛЮЧЕНО (можно возобновить тренировку)")
            else:
                print("💾 [Save State] ⏭️ Сохранение training_state.pt ОТКЛЮЧЕНО (экономия места, только адаптеры)")

            print(f"🧠[Phase 2] Loading {model_variant} model on {device} ({precision})...")
            from acestep.training_v2.model_loader import load_decoder_for_training
            model = load_decoder_for_training(checkpoint_dir=checkpoint_dir, variant=model_variant, device=device, precision=precision)

            if vram_cleanup or offload_enc:
                print(f"🧹[System] Optimizing VRAM Usage (Cleanup/Offload)...")
                to_kill =["vae", "text_encoder", "tokenizer", "detokenizer"]
                if encoder_train_mode == "none":
                    to_kill.extend(["music_encoder", "lyric_encoder", "timbre_encoder", "condition_projection"])
                
                for attr in to_kill:
                    if hasattr(model, attr):
                        m = getattr(model, attr)
                        if m is not None: setattr(model, attr, m.to("cpu"))
                    if hasattr(model, "encoder") and hasattr(model.encoder, attr):
                        m = getattr(model.encoder, attr)
                        if m is not None: setattr(model.encoder, attr, m.to("cpu"))
                        
                if encoder_train_mode == "none" and hasattr(model, "encoder") and hasattr(model.encoder, "text_projector"):
                    model.encoder.text_projector.to("cpu")

                if offload_enc and encoder_train_mode == "none" and hasattr(model, "encoder") and model.encoder is not None:
                    print(f"🧹[System] Full Encoder offload enabled. Moving to CPU.")
                    model.encoder.to("cpu")

                model.decoder.to("cpu")
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.synchronize()
                
                model.decoder.to(device)
                if hasattr(model, "null_condition_emb") and model.null_condition_emb is not None:
                    model.null_condition_emb.data = model.null_condition_emb.data.to(device)

            model.train()

            variant_dir = f"acestep-v15-{model_variant}"
            source_model_dir = os.path.join(checkpoint_dir, variant_dir)
            if not os.path.isdir(source_model_dir): source_model_dir = os.path.join(checkpoint_dir, model_variant)
            if not os.path.isdir(source_model_dir): source_model_dir = checkpoint_dir

            print("\n🔥 Starting Full Fine-Tuning Loop...")
            
            block_swap = None
            if block_swap_ratio > 0.0:
                block_swap = AdvancedBlockSwapManager(model.decoder, device=device, offload_device="cpu")
                block_swap.apply(offload_ratio=block_swap_ratio)

            module = FullFinetuneModuleV2(model, training_cfg, device, precision, encoder_train_mode, train_null_emb)
            
            target_dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else torch.float32)
            apply_quantization_base_model(module.model, target_dtype, base_quantization, seed)

            trainable_params =[p for p in module.parameters() if p.requires_grad]
            print(f"🎯 Training {sum(p.numel() for p in trainable_params):,} parameters")

            import acestep.training.data_module as dm_module
            original_setup = dm_module.PreprocessedDataModule.setup

            def balanced_setup(self_dm, stage=None):
                main_ds = dm_module.PreprocessedTensorDataset(main_tensor_dir)
                if reg_tensor_dir and os.path.exists(reg_tensor_dir):
                    reg_ds = dm_module.PreprocessedTensorDataset(reg_tensor_dir)
                    class BalancedWrapper(torch.utils.data.Dataset):
                        def __init__(self, m_ds, r_ds):
                            self.m_ds = m_ds; self.r_ds = r_ds; self.target_len = max(len(m_ds), len(r_ds))
                        def __len__(self): return self.target_len * 2
                        def __getitem__(self, idx):
                            if idx % 2 == 0: return self.m_ds[(idx // 2) % len(self.m_ds)]
                            else:
                                item = dict(self.r_ds[(idx // 2) % len(self.r_ds)])
                                item["metadata"] = dict(item.get("metadata", {}))
                                item["metadata"]["is_reg"] = True
                                return item
                    self_dm.train_dataset = BalancedWrapper(main_ds, reg_ds)
                    print(f"⚖️ In-Memory Dataset Balanced: Main ({len(main_ds)}) | Reg ({len(reg_ds)}) -> Total Epoch Size: {len(self_dm.train_dataset)}")
                else: self_dm.train_dataset = main_ds
                self_dm.val_dataset = None

            dm_module.PreprocessedDataModule.setup = balanced_setup

            data_module = dm_module.PreprocessedDataModule(
                tensor_dir=main_tensor_dir,
                batch_size=dataset_config["batch_size"],
                num_workers=0 if os.name == 'nt' else 4
            )
            data_module.setup('fit')
            train_loader = data_module.train_dataloader()

            from acestep.training_v2.optim import build_optimizer, build_scheduler

            if base_quantization in["fp4", "fp8"]:
                training_cfg.optimizer_kwargs["eps"] = 1e-6
                training_cfg.max_grad_norm = 0.5 

            optimizer = build_optimizer(
                params=trainable_params, optimizer_type=training_cfg.optimizer_type,
                lr=training_cfg.learning_rate, weight_decay=training_cfg.weight_decay,
                device_type=module.device_type, optimizer_kwargs=training_cfg.optimizer_kwargs
            )

            actual_dataset_len = dataset_len
            if reg_tensor_dir and os.path.exists(reg_tensor_dir):
                actual_dataset_len = max(dataset_len, len(glob.glob(os.path.join(reg_tensor_dir, "*.pt")))) * 2

            eff_batch = max(1, dataset_config["batch_size"] * dataset_config["grad_accum"])
            steps_per_epoch = max(1, len(train_loader) // dataset_config["grad_accum"])
            total_steps_approx = steps_per_epoch * dataset_config["epochs"]

            scheduler = build_scheduler(
                optimizer=optimizer, scheduler_type=training_cfg.scheduler_type,
                total_steps=total_steps_approx, warmup_steps=training_cfg.warmup_steps,
                lr=training_cfg.learning_rate, optimizer_type=training_cfg.optimizer_type
            )

            epoch_history, loss_history, ema_history, lr_history = [], [], [],[]
            ema_loss, ema_alpha, saved_epochs = None, 0.1,[]
            
            # ==============================================================
            # БЛОК RESUME
            # ==============================================================
            start_epoch = 0
            global_step = 0
            resume_from = dataset_config.get("resume_from", "").strip('"')

            if resume_from and os.path.exists(resume_from):
                print(f"\n🔄 [Resume] Загрузка чекпоинта файнтюна из {resume_from}...")
                model_path = os.path.join(resume_from, "model.safetensors")
                state_path = os.path.join(resume_from, "training_state.pt")
                
                if os.path.exists(model_path):
                    from safetensors.torch import load_file
                    state_dict = load_file(model_path)
                    module.model.load_state_dict(state_dict, strict=False)
                    print("📦 [Resume] Веса модели загружены.")
                else:
                    print("⚠️ [Resume] Файл model.safetensors не найден!")

                if os.path.exists(state_path):
                    state = torch.load(state_path, map_location="cpu", weights_only=False)
                    start_epoch = state.get("epoch", 0)
                    global_step = state.get("global_step", 0)
                    
                    if "optimizer_state_dict" in state: 
                        optimizer.load_state_dict(state["optimizer_state_dict"])
                        print("⚙️ [Resume] Состояние оптимизатора загружено.")
                    if "scheduler_state_dict" in state: 
                        scheduler.load_state_dict(state["scheduler_state_dict"])
                        print("⏱️ [Resume] Состояние шедулера загружено.")
                    
                    loss_history = state.get("loss_history", loss_history)
                    ema_history = state.get("ema_history", ema_history)
                    lr_history = state.get("lr_history", lr_history)
                    epoch_history = state.get("epoch_history", epoch_history)
                    saved_epochs = state.get("saved_epochs", saved_epochs)
                    
                    if ema_history: ema_loss = ema_history[-1]
                    print(f"✅ [Resume] Возобновление с Эпохи {start_epoch}, Шага {global_step}")
                else:
                    print("⚠️ [Resume] training_state.pt не найден! Начинаем новую эпоху, но со старыми весами.")
            # ==============================================================

            pbar = ProgressBar(total_steps_approx)
            if global_step > 0: pbar.update(global_step)
            
            start_time = time.time()
            elapsed_str, eta_str, step_time_str, epoch_time_str = "00:00:00", "00:00:00", "0s/it", "0s/ep"

            def fmt_time(secs):
                m, s = divmod(int(max(0, secs)), 60)
                h, m = divmod(m, 60)
                return f"{h:02d}:{m:02d}:{s:02d}"

            accum_step = 0
            accum_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            module.train()

            try:
                for epoch in range(start_epoch, training_cfg.max_epochs):
                    epoch_loss = 0.0
                    num_updates = 0
                    
                    for batch in train_loader:
                        if mm.processing_interrupted():
                            print("\n🛑 Training Interrupted by User via ComfyUI!")
                            break

                        loss = module.training_step(batch)
                        loss = loss / training_cfg.gradient_accumulation_steps
                        
                        loss.backward()
                        accum_loss += loss.item()
                        accum_step += 1

                        if accum_step >= training_cfg.gradient_accumulation_steps:
                            if block_swap is not None:
                                block_swap.prepare_for_optimizer(optimizer)

                            torch.nn.utils.clip_grad_norm_(trainable_params, training_cfg.max_grad_norm)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)

                            if block_swap is not None:
                                block_swap.cleanup_after_optimizer(optimizer)
                            
                            global_step += 1
                            avg_loss = accum_loss / accum_step
                            
                            pbar.update(1)
                            try:
                                from acestep.training_v2.fixed_lora_module import _get_effective_lr
                                current_lr = _get_effective_lr(optimizer, scheduler)
                            except ImportError:
                                current_lr = scheduler.get_last_lr()[0]
                                pg = optimizer.param_groups[0]
                                if "d" in pg:
                                    d_val = pg["d"]
                                    if hasattr(d_val, "item"): d_val = d_val.item()
                                    current_lr *= d_val
                            
                            if ema_loss is None: ema_loss = avg_loss
                            else: ema_loss = ema_alpha * avg_loss + (1 - ema_alpha) * ema_loss

                            current_epoch = global_step / steps_per_epoch
                            epoch_history.append(current_epoch)
                            loss_history.append(avg_loss)
                            ema_history.append(ema_loss)
                            lr_history.append(current_lr)

                            elapsed = time.time() - start_time
                            if global_step > 0:
                                time_per_step = elapsed / (global_step - state.get("global_step", 0) if resume_from and 'state' in locals() else global_step)
                                remaining_steps = total_steps_approx - global_step
                                eta_secs = remaining_steps * time_per_step
                                step_time_str = f"{int(time_per_step * 1000)}ms/it" if time_per_step < 1.0 else f"{time_per_step:.2f}s/it"
                                time_per_epoch = time_per_step * steps_per_epoch
                                ep_m, ep_s = divmod(int(time_per_epoch), 60)
                                ep_h, ep_m = divmod(ep_m, 60)
                                epoch_time_str = f"{ep_h}h {ep_m}m/ep" if ep_h > 0 else (f"{ep_m}m {ep_s}s/ep" if ep_m > 0 else f"{ep_s}s/ep")
                            else:
                                eta_secs = 0
                                
                            elapsed_str = fmt_time(elapsed)
                            eta_str = fmt_time(eta_secs)

                            if global_step % 5 == 0 and unique_id is not None:
                                try:
                                    node_id_str = unique_id[0] if isinstance(unique_id, list) else str(unique_id)
                                    self.process_loss_graph(
                                        epoch_history, loss_history, ema_history, lr_history,
                                        elapsed_str, eta_str, step_time_str, epoch_time_str,
                                        saved_epochs, node_id=node_id_str
                                    )
                                except: pass

                            epoch_loss += avg_loss
                            num_updates += 1
                            accum_loss = 0.0
                            accum_step = 0

                    if mm.processing_interrupted(): break

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    should_save = (epoch + 1) % training_cfg.save_every_n_epochs == 0
                    
                    if should_save and (epoch + 1) < training_cfg.save_start_epoch:
                        should_save = False
                        
                    if should_save and training_cfg.save_loss_threshold > 0.0:
                        avg_epoch_loss = epoch_loss / max(num_updates, 1)
                        loss_to_check = ema_loss if ema_loss is not None else avg_epoch_loss
                        if loss_to_check >= training_cfg.save_loss_threshold:
                            should_save = False
                            print(f"⏭️ Checkpoint skipped at epoch {epoch+1}: EMA Loss ({loss_to_check:.4f}) >= Limit ({training_cfg.save_loss_threshold:.4f})")

                    if should_save:
                        ckpt_dir = os.path.join(output_dir, "checkpoints", f"epoch_{epoch+1}")

                        _save_complete_model(module.model.state_dict(), ckpt_dir, source_model_dir, model_variant, extra_pnginfo, prompt)

                        if finetune_save_state:
                            state_save = {
                                "epoch": epoch + 1,
                                "global_step": global_step,
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "loss_history": loss_history,
                                "ema_history": ema_history,
                                "lr_history": lr_history,
                                "epoch_history": epoch_history,
                                "saved_epochs": saved_epochs,
                            }
                            torch.save(state_save, os.path.join(ckpt_dir, "training_state.pt"))
                            print(f"💾 Checkpoint and training state saved at epoch {epoch+1}")
                        else:
                            print(f"💾 Checkpoint saved at epoch {epoch+1}")
                        
                        if epoch_history:
                            last_ep = epoch_history[-1]
                            if not saved_epochs or abs(saved_epochs[-1] - last_ep) > 0.05:
                                saved_epochs.append(last_ep)

                        if preview_config and preview_config.get("gen_preview", False):
                            self.generate_preview(model, preview_config, checkpoint_dir, output_dir, epoch + 1, device, precision, model_variant, vae_variant, main_tensor_dir, offload_enc, vram_cleanup, encoder_train_mode)

            finally:
                dm_module.PreprocessedDataModule.setup = original_setup
                if block_swap is not None: block_swap.remove()

                if loss_history:
                    print(f"📊 Saving final training graph to: {output_dir}/loss_graph.png")
                    self.process_loss_graph(
                        epoch_history, loss_history, ema_history, lr_history,
                        elapsed_str, eta_str, step_time_str, epoch_time_str,
                        saved_epochs, node_id=None, output_dir=output_dir
                    )
                    
                if dataset_config.get("save_final", True):
                    print(f"💾 Saving final model weights and workflow...")
                    final_dir = os.path.join(output_dir, "final")
                    _save_complete_model(module.model.state_dict(), final_dir, source_model_dir, model_variant, extra_pnginfo, prompt)
                else: print(f"⏭️ Skipping final save as requested.")

            elapsed = time.time() - start_time
            print(f"\n🎉 Finished in {fmt_time(elapsed)}")
            mm.soft_empty_cache()
            return (output_dir,)

class ACEStepLoRAExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {"default": "", "placeholder": "Путь к базовой папке (acestep-v15-turbo, acestep-v15-xl-base и т.д.)"}),
                "finetuned_model_path": ("STRING", {"default": "", "placeholder": "Путь к папке файнтюна"}),
                "output_path": ("STRING", {"default": "./extracted_lora"}),
                "rank": ("INT", {"default": 32, "min": 1}),
                "alpha": ("INT", {"default": 32, "min": 1}),
                "target_modules": ("STRING", {"default": "q_proj k_proj v_proj o_proj", "multiline": True}),
                "dynamic_method": (["None", "sv_ratio", "sv_fro", "safe"], {"default": "None"}),
                "dynamic_param": ("FLOAT", {"default": 0.9, "step": 0.05}),
                "precision": (["float32", "fp16", "bf16"], {"default": "float32"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "extract"
    CATEGORY = "ACE-Step/Tools"

    def _resolve_model_path(self, path):
        if os.path.isfile(path): return path
        for name in["model.safetensors", "model_state_dict.pt", "model.bin"]:
            p = os.path.join(path, name)
            if os.path.exists(p): return p
        raise FileNotFoundError(f"Не найден файл модели (model.safetensors) в {path}")

    def extract(self, base_model_path, finetuned_model_path, output_path, rank, alpha, target_modules, dynamic_method, dynamic_param, precision, device):
        print(f"\n" + "="*60)
        print(f"🧬 LoRA Extractor Started")
        print(f"="*60)
        base_file = self._resolve_model_path(base_model_path.strip('"'))
        ft_file = self._resolve_model_path(finetuned_model_path.strip('"'))
        out_path = output_path.strip('"')

        base_sd = load_file(base_file) if base_file.endswith('.safetensors') else torch.load(base_file, map_location="cpu", weights_only=True)
        ft_sd = load_file(ft_file) if ft_file.endswith('.safetensors') else torch.load(ft_file, map_location="cpu", weights_only=True)

        targets = target_modules.replace(',', ' ').split()
        new_state_dict = {}
        rank_pattern = {}
        
        save_dtype = torch.float32
        if precision == "fp16": save_dtype = torch.float16
        elif precision == "bf16": save_dtype = torch.bfloat16

        scale_factor = rank / alpha if alpha > 0 else 1.0
        valid_keys =[k for k in base_sd.keys() if k in ft_sd and any(t in k for t in targets) and base_sd[k].shape == ft_sd[k].shape and base_sd[k].ndim == 2 and k.startswith("decoder.")]

        from comfy.utils import ProgressBar
        pbar = ProgressBar(len(valid_keys))
        print(f"🚀 Extracting {len(valid_keys)} layers...")

        for key in valid_keys:
            pbar.update(1)
            W_base = base_sd[key].to(device).float()
            W_ft = ft_sd[key].to(device).float()
            delta_W = W_ft - W_base

            if torch.allclose(delta_W, torch.zeros_like(delta_W), atol=1e-6): continue
            M = delta_W * scale_factor

            try: U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            except Exception:
                U, S, Vh = torch.linalg.svd(M.cpu(), full_matrices=False)
                U, S, Vh = U.to(device), S.to(device), Vh.to(device)

            target_rank = min(rank, len(S))
            if dynamic_method == "sv_ratio":
                keep_indices = torch.nonzero(S >= S[0] * dynamic_param).flatten()
                target_rank = keep_indices[-1].item() + 1 if len(keep_indices) > 0 else 1
            elif dynamic_method == "sv_fro":
                S_sq = S.pow(2)
                keep_indices = torch.nonzero(torch.cumsum(S_sq, dim=0) >= dynamic_param * torch.sum(S_sq)).flatten()
                target_rank = keep_indices[0].item() + 1 if len(keep_indices) > 0 else len(S)

            target_rank = min(target_rank, rank, len(S))
            layer_name = key.replace("decoder.", "").replace(".weight", "")
            if dynamic_method in["sv_fro", "sv_ratio"]: rank_pattern[layer_name] = target_rank

            U_r, S_r, Vh_r = U[:, :target_rank], S[:target_rank], Vh[:target_rank, :]
            sqrt_S = torch.sqrt(S_r)

            new_B = U_r @ torch.diag(sqrt_S)
            new_A = torch.diag(sqrt_S) @ Vh_r

            lora_base = key.replace("decoder.", "base_model.model.").replace(".weight", "")
            new_state_dict[lora_base + ".lora_A.weight"] = new_A.to(save_dtype).cpu()
            new_state_dict[lora_base + ".lora_B.weight"] = new_B.to(save_dtype).cpu()

        os.makedirs(out_path, exist_ok=True)
        
        config = {
            "peft_type": "LORA",
            "r": rank, 
            "lora_alpha": alpha, 
            "lora_dropout": 0.0,
            "target_modules": targets, 
            "bias": "none", 
            "task_type": "FEATURE_EXTRACTION",
            "rank_pattern": {},
            "alpha_pattern": {}
        }
        
        if dynamic_method in ["sv_fro", "sv_ratio"] and rank_pattern:
            config["rank_pattern"] = rank_pattern
            config["alpha_pattern"] = rank_pattern 
            config["r"] = max(rank_pattern.values()) if rank_pattern else rank
            config["lora_alpha"] = config["r"]
            avg_rank = sum(rank_pattern.values()) / len(rank_pattern) if rank_pattern else rank
            print(f"📊 Extracted with dynamic rank. Avg Rank: {avg_rank:.2f}")

        with open(os.path.join(out_path, "adapter_config.json"), 'w') as f: 
            json.dump(config, f, indent=2)
            
        save_file(new_state_dict, os.path.join(out_path, "adapter_model.safetensors"))

        print(f"✅ Extraction complete! Saved to {out_path}")
        mm.soft_empty_cache()
        return (out_path,)
    

NODE_CLASS_MAPPINGS = {
    "ACEStepDatasetConfig": ACEStepDatasetConfig,
    "ACEStepModelConfig": ACEStepModelConfig,
    "ACEStepLoKRConfig": ACEStepLoKRConfig,
    "ACEStepAdamWConfig": ACEStepAdamWConfig,
    "ACEStepProdigyConfig": ACEStepProdigyConfig,
    "ACEStepProdigyPlusConfig": ACEStepProdigyPlusConfig,
    "ACEStepAdemamixConfig": ACEStepAdemamixConfig,
    "ACEStepAdafactorConfig": ACEStepAdafactorConfig,
    "ACEStepPreviewConfig": ACEStepPreviewConfig,
    "ACEStepEstimator": ACEStepEstimator,
    "ACEStepLoRAResize": ACEStepLoRAResize,
    "ACEStepTrainer": ACEStepTrainer,
    "ACEStepFinetuneTrainer": ACEStepFinetuneTrainer,
    "ACEStepLoRAExtractor": ACEStepLoRAExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACEStepDatasetConfig": "📦 Dataset Config",
    "ACEStepModelConfig": "🧠 Model & LoRA Config",
    "ACEStepLoKRConfig": "🧠 Model & LoKR Config",
    "ACEStepAdamWConfig": "🟢 Optimizer: AdamW",
    "ACEStepProdigyConfig": "🟠 Optimizer: Prodigy",
    "ACEStepProdigyPlusConfig": "🔴 Optimizer: Prodigy PLUS",
    "ACEStepAdemamixConfig": "🔵 Optimizer: Ademamix",
    "ACEStepAdafactorConfig": "🟣 Optimizer: Adafactor",
    "ACEStepPreviewConfig": "🎵 Preview Config",
    "ACEStepEstimator": "📊 ACE-Step Estimator",
    "ACEStepLoRAResize": "📉 ACE-Step LoRA Resize",
    "ACEStepTrainer": "▶️ ACE-Step Trainer",
    "ACEStepFinetuneTrainer": "▶️ ACE-Step Full Finetune",
    "ACEStepLoRAExtractor": "🧬 ACE-Step LoRA Extractor",
}