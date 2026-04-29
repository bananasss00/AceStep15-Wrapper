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
# Ядро мержа с использованием Mergekit
# ============================================================================
def execute_mergekit_task(w_list, strengths, method_name, density, global_scale):
    try:
        from mergekit.architecture import WeightInfo
        from mergekit.common import ModelReference, ModelPath, ImmutableMap
        from mergekit.io.tasks import GatherTensors
        from mergekit.merge_methods.generalized_task_arithmetic import GTATask
        from mergekit.merge_methods.slerp import SlerpTask
        from mergekit.merge_methods import REGISTERED_MERGE_METHODS
        from mergekit.sparsify import RescaleNorm
        
        refs =[ModelReference(model=ModelPath(path=f"lora_{i}")) for i in range(len(w_list))]
        tensors = {refs[i]: w_list[i] for i in range(len(w_list))}
        
        param_map = {}
        for i, ref in enumerate(refs):
            param_map[ref] = ImmutableMap({"weight": float(strengths[i]), "density": float(density)})

        weight_info = WeightInfo(name="in_memory_tensor", dtype=w_list[0].dtype, is_embed=False)

        if method_name in["dare_ties", "ties", "della"]:
            base_ref = ModelReference(model=ModelPath(path="base"))
            tensors[base_ref] = torch.zeros_like(w_list[0]) 
            param_map[base_ref] = ImmutableMap({"weight": 0.0, "density": 1.0})
            tensor_params = ImmutableMap(param_map)

            mode_map = {"dare_ties": "dare_ties", "ties": "ties", "della": "della_linear"}
            mode = mode_map[method_name]
            method = REGISTERED_MERGE_METHODS[mode]
            
            all_refs = refs +[base_ref]
            weight_info_map = {ref: weight_info for ref in all_refs}
            gather_tensors = GatherTensors(weight_info=ImmutableMap(weight_info_map))
            
            rescale_norm = RescaleNorm.l1 if getattr(method, "default_rescale", False) else None

            task = GTATask(
                method=method,
                tensors=gather_tensors,
                base_model=base_ref,
                weight_info=weight_info,
                gather_tensors=gather_tensors,
                tensor_parameters=tensor_params,
                int8_mask=False,
                normalize=False,
                lambda_=float(global_scale), # Передаем общий масштаб сюда
                rescale_norm=rescale_norm
            )
            return task.execute(tensors=tensors).to(w_list[0].device)

        elif method_name == "slerp":
            if len(w_list) != 2:
                print("⚠️ SLERP требует ровно 2 LoRA! Переключаемся на Linear.")
            else:
                t = strengths[1] / (strengths[0] + strengths[1] + 1e-8)
                tensors_slerp = {refs[0]: w_list[0], refs[1]: w_list[1]}
                
                weight_info_map = {refs[0]: weight_info, refs[1]: weight_info}
                gather_tensors = GatherTensors(weight_info=ImmutableMap(weight_info_map))
                
                task = SlerpTask(
                    gather_tensors=gather_tensors,
                    base_model=refs[0],
                    weight_info=weight_info,
                    t=t
                )
                # Slerp не поддерживает lambda_ внутри, умножаем после
                return (task.execute(tensors=tensors_slerp) * global_scale).to(w_list[0].device)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️ Ошибка mergekit (переключаемся на Linear): {e}")

    # Fallback на Linear
    res = torch.zeros_like(w_list[0])
    for w, s in zip(w_list, strengths):
        res += w * s
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
            
            stack.append({
                "path": final_path,
                "strength": strength
            })
            
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
                "merge_method": (["dare_ties", "ties", "della", "slerp", "linear"], {"default": "dare_ties"}),
                "target_rank": ("INT", {"default": 128, "min": 8, "max": 512, "step": 8}),
                "density": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "svd_method": (["rSVD", "energy_rSVD"], {"default": "rSVD"}),
                "ignore_bias": ("BOOLEAN", {"default": True, "tooltip": "Для музыкальных Лор строго рекомендуется True во избежание гула."}),
                "normalize_magnitudes": ("BOOLEAN", {"default": True, "tooltip": "Уравнивает 'силу' лор. Включите, если одна Лора полностью перекрывает другую из-за разницы в обучении."}),
                "global_scale": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Множитель итоговых весов. Если есть шум/гул, снижайте. Обычно 2 лоры = 0.5-0.7."}),
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

    def smart_merge(self, model, lora_stack, merge_method, target_rank, density, svd_method, ignore_bias, normalize_magnitudes, global_scale):
        if not lora_stack:
            return (model, {})

        print(f"\n[Smart Lora Merger] 🧠 Слияние: {len(lora_stack)} LoRA | Ранг: {target_rank} | Масштаб: {global_scale} | Нормализация: {normalize_magnitudes}")
        
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
        
        # Для аналитики (чтобы показать пользователю в консоли)
        norms_log = {item["name"]:[] for item in loaded_sds}
        
        def process_layer(base_key):
            try:
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
                        
                        # Сбор статистики для консоли
                        current_norm = torch.norm(W).item()
                        layer_norms.append(current_norm)
                        norms_log[item["name"]].append(current_norm)
                        
                        W_list.append(W)
                        valid_strengths.append(item["strength"])
                        
                if not W_list:
                    return base_key, None, None, None

                # ============================================================
                # ИСПРАВЛЕНИЕ ДОМИНАЦИИ: Выравниваем магнитуды матриц
                # ============================================================
                if normalize_magnitudes and len(W_list) > 1:
                    # Вычисляем среднюю норму среди всех лор в этом слое
                    target_norm = sum(layer_norms) / len(layer_norms)
                    for i in range(len(W_list)):
                        # Приводим каждую матрицу к среднему знаменателю
                        W_list[i] = W_list[i] * (target_norm / (layer_norms[i] + 1e-8))

                if len(W_list) == 1:
                    W_merged = W_list[0] * valid_strengths[0] * global_scale
                else:
                    W_merged = execute_mergekit_task(W_list, valid_strengths, merge_method, density, global_scale)

                if svd_method == "energy_rSVD": A_merged, B_merged = apply_energy_rsvd(W_merged, target_rank)
                else: A_merged, B_merged = apply_rsvd(W_merged, target_rank)

                return base_key, A_merged.cpu().to(orig_dtype), B_merged.cpu().to(orig_dtype), orig_dtype
                
            except Exception as e:
                print(f"[Smart Lora Merger] Ошибка в слое {base_key}: {e}")
                return base_key, None, None, None

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_layer, key): key for key in base_keys}
            for future in as_completed(futures):
                base_key, A_merged, B_merged, orig_dtype = future.result()
                if A_merged is not None:
                    merged_state_dict[f"{base_key}.lora_A.weight"] = A_merged
                    merged_state_dict[f"{base_key}.lora_B.weight"] = B_merged
                pbar.update(1)

        torch.cuda.empty_cache()

        # Вывод аналитики по энергии Лор
        if loaded_sds:
            print("\n[Smart Lora Merger] 📊 Аналитика 'Силы' (Norm) ваших LoRA до нормализации:")
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
                
                if not t_list:
                    continue
                
                total_weight = sum(valid_strengths)
                if total_weight == 0: total_weight = 1.0 
                
                merged_val = torch.zeros_like(t_list[0])
                for t, s in zip(t_list, valid_strengths):
                    merged_val += t * s
                
                merged_val = (merged_val / total_weight) * global_scale
                merged_state_dict[k] = merged_val.to(orig_dtype)

        print(f"\n[Smart Lora Merger] ✅ Тензоры успешно слиты! Размер словаря: {len(merged_state_dict)}")

        # Инъекция
        dit_handler = model["dit_handler"]
        hash_str = hashlib.md5(str(merged_state_dict.keys()).encode()).hexdigest()[:8]
        adapter_name = f"smart_merged_{merge_method}_{hash_str}"

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
            cfg.r = target_rank
            cfg.lora_alpha = target_rank
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
                new_active[adapter_name] = 1.0 # Запечено
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

        print(f"[Save Merged Lora] 💾 LoRA сохранена в: {save_path} (Bias: {'Да' if has_bias else 'Нет'})")
        return (save_path,)

# ============================================================================
# Регистрация нод
# ============================================================================
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