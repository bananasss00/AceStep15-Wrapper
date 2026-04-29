import os
import json
import torch
import hashlib
import folder_paths
import tempfile
import comfy.utils
from concurrent.futures import ThreadPoolExecutor, as_completed

import acestep.core.generation.handler.lora.lifecycle as lora_lifecycle
from safetensors.torch import load_file, save_file

# ============================================================================
# Вспомогательные функции (Алгоритмы SVD)
# ============================================================================
def apply_rsvd(W: torch.Tensor, new_dim: int, niter: int = 2, oversample: int = 4):
    rank = min(new_dim, min(W.shape))
    q = min(rank + oversample, min(W.shape))
    U, S, V = torch.svd_lowrank(W, q=q, niter=niter)
    
    U = U[:, :rank]
    S = S[:rank]
    V = V[:, :rank]
    S_sqrt = torch.sqrt(S)
    
    up_new = U * S_sqrt.unsqueeze(0)
    down_new = (V * S_sqrt.unsqueeze(0)).T
    
    if new_dim > up_new.shape[1]:
        pad_up = torch.zeros((up_new.shape[0], new_dim - up_new.shape[1]), dtype=W.dtype, device=W.device)
        pad_down = torch.zeros((new_dim - down_new.shape[0], down_new.shape[1]), dtype=W.dtype, device=W.device)
        up_new = torch.cat([up_new, pad_up], dim=1)
        down_new = torch.cat([down_new, pad_down], dim=0)
    return down_new, up_new

def apply_energy_rsvd(W: torch.Tensor, new_dim: int, energy_keep_ratio: float = 1.5, niter: int = 1, oversample: int = 2):
    rank = min(new_dim, min(W.shape))
    q = min(rank + oversample, min(W.shape))
    U, S, V = torch.svd_lowrank(W, q=q, niter=niter)
    
    energy = torch.cumsum(S, dim=0) / torch.sum(S)
    target_idx = torch.searchsorted(energy, 0.95).item() + 1
    k = min(target_idx, int(new_dim * energy_keep_ratio))
    k = min(k, rank)
    
    U = U[:, :k]
    S = S[:k]
    V = V[:, :k]
    S_sqrt = torch.sqrt(S)
    up_new = U * S_sqrt.unsqueeze(0)            
    down_new = (V * S_sqrt.unsqueeze(0)).T      
    
    if new_dim > up_new.shape[1]:
        pad_up = torch.zeros((up_new.shape[0], new_dim - up_new.shape[1]), dtype=W.dtype, device=W.device)
        pad_down = torch.zeros((new_dim - down_new.shape[0], down_new.shape[1]), dtype=W.dtype, device=W.device)
        up_new = torch.cat([up_new, pad_up], dim=1)
        down_new = torch.cat([down_new, pad_down], dim=0)
    return down_new, up_new


# ============================================================================
# АЛГОРИТМЫ СЛИЯНИЯ (ЧИСТЫЙ PYTORCH, БЕЗ ЗАВИСИМОСТЕЙ)
# ============================================================================
def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """Сферическая линейная интерполяция между двумя тензорами"""
    v0_flat = v0.flatten()
    v1_flat = v1.flatten()
    
    v0_norm = v0_flat / (torch.norm(v0_flat) + 1e-8)
    v1_norm = v1_flat / (torch.norm(v1_flat) + 1e-8)
    
    dot = torch.sum(v0_norm * v1_norm)
    
    if dot > DOT_THRESHOLD:
        return (1.0 - t) * v0 + t * v1
        
    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta_0 = torch.sin(theta_0)
    
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    return s0 * v0 + s1 * v1

def drop_and_rescale(tensor, density, method="random"):
    """Удаление слабых/случайных весов и рескейлинг оставшихся (основа DARE/TIES)"""
    if density >= 1.0:
        return tensor
        
    if method == "random":
        # DARE: Случайный дропаут весов
        mask = torch.rand_like(tensor) < density
    else:
        # TIES: Удаление слабейших по амплитуде
        k = max(1, int(tensor.numel() * density))
        threshold = torch.topk(tensor.abs().flatten(), k).values[-1]
        mask = tensor.abs() >= threshold
        
    return (tensor * mask) / density

def ties_consensus_merge(tensors, strengths):
    """Вычисляет знак консенсуса и усредняет только согласные веса (TIES)"""
    stacked = torch.stack(tensors)
    signs = torch.sign(stacked)
    
    # 1. Считаем взвешенный консенсус знаков
    sum_signs = torch.zeros_like(tensors[0])
    for t, s in zip(tensors, strengths):
        sum_signs += torch.sign(t) * s
    
    consensus_sign = torch.sign(sum_signs)
    consensus_sign[consensus_sign == 0] = 1.0 # Дефолтный знак при ничьей
    
    # 2. Оставляем только те веса, знак которых совпадает с консенсусом
    merged = torch.zeros_like(tensors[0])
    weight_sum = torch.zeros_like(tensors[0])
    
    for t, w in zip(tensors, strengths):
        agree_mask = (torch.sign(t) == consensus_sign).float()
        merged += t * w * agree_mask
        weight_sum += w * agree_mask
        
    return merged / (weight_sum + 1e-8)


def execute_pure_pytorch_merge(w_list, strengths, method_name, density, global_scale):
    """Единая точка входа для продвинутых математических алгоритмов мержа"""
    
    # 1. Linear (Обычное взвешенное сложение)
    if method_name == "linear":
        res = torch.zeros_like(w_list[0])
        for w, s in zip(w_list, strengths):
            res += w * s
        return res * global_scale

    # 2. SLERP (Сферическая интерполяция)
    if method_name == "slerp":
        if len(w_list) != 2:
            print("⚠️ SLERP требует строго 2 LoRA! Переключаемся на Linear.")
            return execute_pure_pytorch_merge(w_list, strengths, "linear", density, global_scale)
        t = strengths[1] / (strengths[0] + strengths[1] + 1e-8)
        return slerp(t, w_list[0], w_list[1]) * global_scale

    # 3. TIES / DARE / DELLA
    if method_name in ["dare_ties", "ties", "della"]:
        processed_tensors =[]
        for w in w_list:
            if method_name == "ties":
                # TIES обнуляет самые слабые веса (по модулю)
                pt = drop_and_rescale(w, density, method="magnitude")
            else: # dare_ties или della
                # DARE обнуляет веса случайно, сохраняя общую статистику
                pt = drop_and_rescale(w, density, method="random")
            processed_tensors.append(pt)
        
        if method_name == "della":
            # DELLA = DARE Dropout + обычное Linear слияние
            res = torch.zeros_like(w_list[0])
            for w, s in zip(processed_tensors, strengths):
                res += w * s
        else:
            # DARE-TIES и TIES = Dropout + Сохранение только весов, согласных по знаку
            res = ties_consensus_merge(processed_tensors, strengths)
            
        return res * global_scale


# ============================================================================
# 1. Нода: AceStep Lora Stack Entry
# ============================================================================
class AceStepLoraStackEntry:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_path": ("STRING", {"default": "", "multiline": False, "placeholder": "Полный абсолютный путь к ПАПКЕ LoRA"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "Сила влияния этой LoRA в итоговом мерже"}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", {"tooltip": "Стек от предыдущей ноды (Daisy-chaining)"}),
            }
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "add_to_stack"
    CATEGORY = "ACE-Step/Smart Merger"

    def add_to_stack(self, lora_path, strength, lora_stack=None):
        stack = lora_stack.copy() if lora_stack is not None else[]
        final_path = lora_path.strip()
        if final_path:
            if not os.path.exists(final_path):
                print(f"[Smart Lora Stack] ⚠️ Предупреждение: Путь не найден: {final_path}")
            stack.append({"path": final_path, "strength": strength})
        return (stack,)


# ============================================================================
# 2. Нода: AceStep Smart Lora Merger
# ============================================================================
class AceStepSmartLoraMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ACESTEP_MODEL",),
                "lora_stack": ("LORA_STACK",),
                "merge_method": (["concat (Lossless)", "linear", "dare_ties", "ties", "della", "slerp"], {"default": "concat (Lossless)", "tooltip": "Concat: Идеально для музыки. Склеивает без потерь и шума. Остальные методы - математические."}),
                "target_rank": ("INT", {"default": 128, "min": 8, "max": 512, "step": 8, "tooltip": "Игнорируется в режиме concat. В остальных: Ранг сжатия (SVD)."}),
                "density": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Доля сохраняемых весов для DARE/TIES. (1.0 = выключить обнуление)"}),
                "svd_method": (["rSVD", "energy_rSVD"], {"default": "rSVD"}),
                "ignore_bias": ("BOOLEAN", {"default": True, "tooltip": "Игнорировать смещения (Bias). Спасает от гула."}),
                "global_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Общий множитель Лоры."}),
                "normalize_magnitudes": ("BOOLEAN", {"default": True, "tooltip": "Автоматически уравнивает силу лор, чтобы ни одна не 'съела' другую."}),
            }
        }

    RETURN_TYPES = ("ACESTEP_MODEL", "MERGED_LORA_TENSORS")
    RETURN_NAMES = ("model", "merged_tensors")
    FUNCTION = "smart_merge"
    CATEGORY = "ACE-Step/Smart Merger"

    def _load_lora_weights(self, path):
        if os.path.isdir(path):
            st_path = os.path.join(path, "adapter_model.safetensors")
            bin_path = os.path.join(path, "adapter_model.bin")
            if os.path.exists(st_path): return load_file(st_path)
            elif os.path.exists(bin_path): return torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            if path.endswith('.safetensors'): return load_file(path)
            else: return torch.load(path, map_location="cpu", weights_only=True)
        return None

    def smart_merge(self, model, lora_stack, merge_method, target_rank, density, svd_method, ignore_bias, global_scale, normalize_magnitudes):
        if not lora_stack:
            return (model, {})

        print(f"\n[Smart Lora Merger] 🧠 Слияние: {len(lora_stack)} LoRA | Метод: {merge_method} (Pure PyTorch)")
        
        loaded_sds =[]
        for lora in lora_stack:
            sd = self._load_lora_weights(lora["path"])
            if sd is None:
                raise FileNotFoundError(f"Не удалось загрузить веса из: {lora['path']}")
            
            scale = 1.0
            config_path = os.path.join(os.path.dirname(lora["path"]) if os.path.isfile(lora["path"]) else lora["path"], "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                    r = cfg.get("r", 1.0)
                    alpha = cfg.get("lora_alpha", r)
                    scale = alpha / r

            loaded_sds.append({"sd": sd, "scale": scale, "strength": lora["strength"], "name": os.path.basename(lora["path"])})

        base_keys = set()
        for item in loaded_sds:
            for k in item["sd"].keys():
                if any(x in k for x in[".lora_A", ".lora_B", ".lora_up", ".lora_down"]):
                    b_key = k.replace(".lora_A.weight", "").replace(".lora_B.weight", "") \
                             .replace(".lora_down.weight", "").replace(".lora_up.weight", "")
                    base_keys.add(b_key)

        merged_state_dict = {}
        pbar = comfy.utils.ProgressBar(len(base_keys))
        
        final_rank = target_rank if merge_method != "concat (Lossless)" else 0
        norms_log = {item["name"]:[] for item in loaded_sds}

        def process_layer(base_key):
            try:
                # -------------------------------------------------------------
                # РЕЖИМ 1: CONCAT (Без потерь, Идеально для аудио)
                # -------------------------------------------------------------
                if merge_method == "concat (Lossless)":
                    A_tensors =[]
                    B_tensors =[]
                    orig_dtype = torch.float32
                    
                    for item in loaded_sds:
                        sd = item["sd"]
                        A = sd.get(f"{base_key}.lora_A.weight")
                        if A is None: A = sd.get(f"{base_key}.lora_down.weight")
                        B = sd.get(f"{base_key}.lora_B.weight")
                        if B is None: B = sd.get(f"{base_key}.lora_up.weight")
                        
                        if A is not None and B is not None:
                            orig_dtype = A.dtype
                            A_gpu = A.cuda().float()
                            B_gpu = B.cuda().float()
                            
                            B_scaled = B_gpu * item["scale"] * item["strength"] * global_scale
                            A_tensors.append(A_gpu)
                            B_tensors.append(B_scaled)
                            
                    if not A_tensors:
                        return base_key, None, None, None, 0
                        
                    A_merged = torch.cat(A_tensors, dim=0).cpu().to(orig_dtype)
                    B_merged = torch.cat(B_tensors, dim=1).cpu().to(orig_dtype)
                    new_r = A_merged.shape[0]
                    return base_key, A_merged, B_merged, orig_dtype, new_r

                # -------------------------------------------------------------
                # РЕЖИМ 2: Классический (SVD + TIES/SLERP/Linear)
                # -------------------------------------------------------------
                W_list =[]
                valid_strengths =[]
                orig_dtype = torch.float32
                layer_norms =[]
                
                for idx, item in enumerate(loaded_sds):
                    sd = item["sd"]
                    A = sd.get(f"{base_key}.lora_A.weight")
                    if A is None: A = sd.get(f"{base_key}.lora_down.weight")
                    B = sd.get(f"{base_key}.lora_B.weight")
                    if B is None: B = sd.get(f"{base_key}.lora_up.weight")
                    
                    if A is not None and B is not None:
                        orig_dtype = A.dtype
                        A_gpu = A.cuda().float()
                        B_gpu = B.cuda().float()
                        
                        W = (B_gpu @ A_gpu) * item["scale"]
                        
                        current_norm = torch.norm(W).item()
                        layer_norms.append(current_norm)
                        norms_log[item["name"]].append(current_norm)
                        
                        W_list.append(W)
                        valid_strengths.append(item["strength"])
                        
                if not W_list:
                    return base_key, None, None, None, 0

                # Выравнивание Магнитуд (чтобы одна Лора не съела другую)
                if normalize_magnitudes and len(W_list) > 1:
                    target_norm = sum(layer_norms) / len(layer_norms)
                    for i in range(len(W_list)):
                        W_list[i] = W_list[i] * (target_norm / (layer_norms[i] + 1e-8))

                # Слияние чистым PyTorch-алгоритмом
                W_merged = execute_pure_pytorch_merge(W_list, valid_strengths, merge_method, density, global_scale)

                # Факторизация
                if svd_method == "energy_rSVD": A_merged, B_merged = apply_energy_rsvd(W_merged, target_rank)
                else: A_merged, B_merged = apply_rsvd(W_merged, target_rank)

                return base_key, A_merged.cpu().to(orig_dtype), B_merged.cpu().to(orig_dtype), orig_dtype, target_rank
                
            except Exception as e:
                print(f"[Smart Lora Merger] Ошибка в слое {base_key}: {e}")
                return base_key, None, None, None, 0

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_layer, key): key for key in base_keys}
            for future in as_completed(futures):
                base_key, A_merged, B_merged, orig_dtype, new_r = future.result()
                if A_merged is not None:
                    merged_state_dict[f"{base_key}.lora_A.weight"] = A_merged
                    merged_state_dict[f"{base_key}.lora_B.weight"] = B_merged
                    if new_r > final_rank:
                        final_rank = new_r
                pbar.update(1)

        torch.cuda.empty_cache()

        # Вывод аналитики по энергии
        if loaded_sds and merge_method != "concat (Lossless)":
            print("\n[Smart Lora Merger] 📊 Средняя норма матриц (Сила) до нормализации:")
            for name, n_list in norms_log.items():
                if n_list:
                    avg_norm = sum(n_list)/len(n_list)
                    print(f"   - {name}: {avg_norm:.2f}")

        # Обработка bias'ов
        if not ignore_bias:
            other_keys = set()
            for item in loaded_sds:
                for k in item["sd"].keys():
                    if not any(x in k for x in[".lora_A", ".lora_B", ".lora_up", ".lora_down"]):
                        other_keys.add(k)
                        
            for k in other_keys:
                t_list =[]
                valid_strengths =[]
                orig_dtype = torch.float32
                
                for item in loaded_sds:
                    val = item["sd"].get(k)
                    if val is not None:
                        orig_dtype = val.dtype
                        t_list.append(val.float())
                        valid_strengths.append(item["strength"])
                
                if not t_list: continue
                
                total_weight = sum(valid_strengths)
                if total_weight == 0: total_weight = 1.0 
                
                merged_val = torch.zeros_like(t_list[0])
                for t, s in zip(t_list, valid_strengths):
                    merged_val += t * s
                
                merged_val = (merged_val / total_weight) * global_scale
                merged_state_dict[k] = merged_val.to(orig_dtype)

        print(f"\n[Smart Lora Merger] ✅ Тензоры успешно слиты! Итоговый ранг: {final_rank}")

        # Инъекция
        dit_handler = model["dit_handler"]
        hash_str = hashlib.md5(str(merged_state_dict.keys()).encode()).hexdigest()[:8]
        adapter_name = f"smart_merged_{hash_str}"

        orig_load_st = lora_lifecycle.load_safetensors
        orig_torch_load = torch.load

        try:
            import peft
            orig_from_pretrained = peft.PeftConfig.from_pretrained
        except ImportError:
            orig_from_pretrained = None

        target_modules = list(set([k.split('.')[-3] for k in merged_state_dict.keys() if "lora_A" in k or "lora_down" in k]))

        def hooked_from_pretrained(pretrained_model_name_or_path, **kwargs):
            cfg = orig_from_pretrained(pretrained_model_name_or_path, **kwargs)
            cfg.r = final_rank
            cfg.lora_alpha = final_rank
            cfg.target_modules = target_modules
            return cfg

        def hooked_load_st(path, *args, **kwargs_st): return merged_state_dict
        def hooked_torch_load(path, *args, **kwargs_torch): return merged_state_dict

        lora_lifecycle.load_safetensors = hooked_load_st
        torch.load = hooked_torch_load
        
        if orig_from_pretrained:
            peft.PeftConfig.from_pretrained = hooked_from_pretrained

        try:
            dummy_path = lora_stack[0]["path"]
            load_msg = dit_handler.add_lora(dummy_path, adapter_name=adapter_name, ignore_bias=ignore_bias)
            
            decoder = getattr(dit_handler.model, "decoder", None)
            if decoder is not None and hasattr(decoder, "set_adapter"):
                try: decoder.set_adapter(adapter_name)
                except: pass
                
            if "❌" in load_msg and "already loaded" not in load_msg:
                print(f"[Smart Lora Merger] ❌ Ошибка инъекции: {load_msg}")
            else:
                new_active = model["active_adapters"].copy()
                new_active[adapter_name] = 1.0 
                model = model.copy()
                model["active_adapters"] = new_active
                print(f"[Smart Lora Merger] ✨ In-Memory инъекция '{adapter_name}' выполнена успешно!")
                
        finally:
            lora_lifecycle.load_safetensors = orig_load_st
            torch.load = orig_torch_load
            if orig_from_pretrained:
                peft.PeftConfig.from_pretrained = orig_from_pretrained

        return (model, merged_state_dict)

# ============================================================================
# 3. Нода: AceStep Save Merged Lora
# ============================================================================
class AceStepSaveMergedLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merged_tensors": ("MERGED_LORA_TENSORS",),
                "output_dir": ("STRING", {"default": folder_paths.get_output_directory(), "multiline": False}),
                "file_name": ("STRING", {"default": "My_Smart_Merge_LoRA", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_lora"
    CATEGORY = "ACE-Step/Smart Merger"
    OUTPUT_NODE = True

    def save_lora(self, merged_tensors, output_dir, file_name):
        if not merged_tensors:
            return ("",)

        save_path = os.path.join(output_dir, file_name)
        os.makedirs(save_path, exist_ok=True)

        sample_tensor = next((v for k, v in merged_tensors.items() if "lora_A" in k), None)
        rank = sample_tensor.shape[1] if sample_tensor is not None else 64
        target_modules = list(set([k.split('.')[-3] for k in merged_tensors.keys() if "lora_A" in k or "lora_down" in k]))
        
        has_bias = any("bias" in k for k in merged_tensors.keys())
        
        config_data = {
            "peft_type": "LORA",
            "r": rank,
            "lora_alpha": rank,
            "target_modules": target_modules,
            "bias": "all" if has_bias else "none"
        }

        with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
            json.dump(config_data, f, indent=2)

        weights_path = os.path.join(save_path, "adapter_model.safetensors")
        save_file(merged_tensors, weights_path)

        print(f"[Save Merged Lora] 💾 LoRA сохранена в: {save_path} (Ранг: {rank})")
        return (save_path,)

NODE_CLASS_MAPPINGS = {
    "AceStepLoraStackEntry": AceStepLoraStackEntry,
    "AceStepSmartLoraMerger": AceStepSmartLoraMerger,
    "AceStepSaveMergedLora": AceStepSaveMergedLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepLoraStackEntry": "ACE-Step LoRA Stack Entry 📚",
    "AceStepSmartLoraMerger": "ACE-Step Smart LoRA Merger 🧠",
    "AceStepSaveMergedLora": "ACE-Step Save Merged LoRA 💾",
}