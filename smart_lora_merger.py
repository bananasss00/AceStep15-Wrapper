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
# АЛГОРИТМЫ СЛИЯНИЯ И ПРОРЕЖИВАНИЯ (PURE PYTORCH)
# ============================================================================
def safe_float32(tensor):
    if tensor.dtype in[getattr(torch, 'float8_e4m3fn', None), getattr(torch, 'float8_e5m2', None)]:
        return tensor.to(torch.float32)
    return tensor.float()

def rand_mask(shape, density, device, dtype):
    return (torch.rand(shape, device=device, dtype=torch.float32) < density).to(dtype)

def ties_sparsify(tensor, density):
    if density >= 1.0: return tensor
    if density <= 0.0: return torch.zeros_like(tensor)
    tensor_f = safe_float32(tensor)
    temp = tensor_f.abs().flatten()
    k = max(1, int(temp.numel() * density))
    threshold = torch.topk(temp, k)[0][-1]
    mask = (tensor_f.abs() >= threshold).to(tensor.dtype)
    return tensor * mask

def dare_sparsify(tensor, density):
    if density >= 1.0: return tensor
    if density <= 0.0: return torch.zeros_like(tensor)
    mask = rand_mask(tensor.shape, density, tensor.device, tensor.dtype)
    return (tensor * mask) / density

def della_sparsify(tensor, density, epsilon=0.1):
    if density >= 1.0: return tensor
    if density <= 0.0: return torch.zeros_like(tensor)
    tensor_f = safe_float32(tensor)
    abs_t = tensor_f.abs()
    mag_max, mag_min = abs_t.max(), abs_t.min()
    if mag_max == mag_min:
        return dare_sparsify(tensor, density)
    norm_mag = (abs_t - mag_min) / (mag_max - mag_min)
    keep_prob = (density - epsilon) + 2 * epsilon * norm_mag
    keep_prob = torch.clamp(keep_prob, 0.0, 1.0)
    mask = (torch.rand(tensor.shape, device=tensor.device, dtype=torch.float32) < keep_prob).to(tensor.dtype)
    safe_prob = torch.clamp(keep_prob, min=1e-6).to(tensor.dtype)
    return (tensor * mask) / safe_prob

def breadcrumbs_sparsify(tensor, density, gamma=0.01):
    if density >= 1.0 and gamma <= 0.0: return tensor
    tensor_f = safe_float32(tensor)
    temp = tensor_f.abs().flatten()
    n = temp.numel()
    n_outliers = int(n * gamma)
    n_noise = int(n * max(0.0, 1.0 - density - gamma))
    if n_outliers + n_noise >= n:
        return torch.zeros_like(tensor)
    sorted_vals, _ = torch.sort(temp)
    noise_thresh = sorted_vals[n_noise] if n_noise > 0 else -1.0
    outlier_thresh = sorted_vals[n - n_outliers - 1] if n_outliers > 0 else float('inf')
    valid = (tensor_f.abs() > noise_thresh) & (tensor_f.abs() <= outlier_thresh)
    return tensor * valid.to(tensor.dtype)

def sign_consensus_merge(tensors, weights):
    tensors_f =[safe_float32(t) for t in tensors]
    sum_t = sum(t * w for t, w in zip(tensors_f, weights))
    consensus_sign = sum_t.sign()
    res = torch.zeros_like(sum_t)
    count = torch.zeros_like(sum_t)
    for t, t_orig, w in zip(tensors_f, tensors, weights):
        valid = (t.sign() == consensus_sign) & (t != 0)
        res += torch.where(valid, t_orig.to(res.dtype) * w, torch.zeros_like(res))
        count += valid.float()
    count = torch.clamp(count, min=1.0)
    return (res / count).to(tensors[0].dtype)

def linear_merge(tensors, weights):
    return sum(t * w for t, w in zip(tensors, weights))

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    v0_f, v1_f = safe_float32(v0), safe_float32(v1)
    v0_flat, v1_flat = v0_f.flatten(), v1_f.flatten()
    norm_v0, norm_v1 = torch.norm(v0_flat), torch.norm(v1_flat)
    if norm_v0 == 0.0 or norm_v1 == 0.0:
        return (1.0 - t) * v0 + t * v1
    v0_norm = v0_flat / norm_v0
    v1_norm = v1_flat / norm_v1
    dot = torch.sum(v0_norm * v1_norm)
    if dot.abs() > DOT_THRESHOLD:
        return (1.0 - t) * v0 + t * v1
    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    return (s0 * v0_f + s1 * v1_f).to(v0.dtype)

def slerp_merge(tensors, weights, explicit_t=None):
    if len(tensors) == 1: return tensors[0] * weights[0]
    res = tensors[0]
    current_w = weights[0]
    for i in range(1, len(tensors)):
        t = explicit_t if explicit_t is not None else (weights[i] / (current_w + weights[i]) if (current_w + weights[i]) > 0 else 0.5)
        current_w += weights[i]
        res = slerp(t, res, tensors[i])
    return res

def model_stock_merge(tensors, weights):
    if len(tensors) == 1: return tensors[0] * weights[0]
    tensors_f =[safe_float32(t) for t in tensors]
    w_sum = sum(weights)
    norm_weights = [w / w_sum for w in weights] if w_sum > 0 else weights
    D_avg = sum(t * w for t, w in zip(tensors_f, norm_weights))
    tr_avg_avg = (D_avg * D_avg).sum()
    tr_i_avg = sum(w * (t * D_avg).sum() for t, w in zip(tensors_f, norm_weights))
    t_opt = tr_i_avg / (tr_avg_avg + 1e-8)
    res = D_avg * t_opt * w_sum
    return res.to(tensors[0].dtype)


# ============================================================================
# НОДЫ НАСТРОЙКИ МЕТОДОВ МЕРЖА
# ============================================================================
class AceStep_MergeMethodConcat:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"normalize": ("BOOLEAN", {"default": True, "tooltip": "Авто-выравнивание силы эффектов лор."})}}
    RETURN_TYPES = ("ACESTEP_MERGE_METHOD",); FUNCTION = "get_method"; CATEGORY = "ACE-Step/Smart Merger/Methods"; TITLE = "ACE-Step Method: Concat (Lossless)"
    def get_method(self, normalize): return ({"name": "concat", "normalize": normalize},)

class AceStep_MergeMethodLinear:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"normalize": ("BOOLEAN", {"default": True, "tooltip": "Авто-выравнивание весов (предотвращает выгорание)."})}}
    RETURN_TYPES = ("ACESTEP_MERGE_METHOD",); FUNCTION = "get_method"; CATEGORY = "ACE-Step/Smart Merger/Methods"; TITLE = "ACE-Step Method: Linear"
    def get_method(self, normalize): return ({"name": "linear", "normalize": normalize},)

class AceStep_MergeMethodSLERP:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"t": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}), "use_stack_weights": ("BOOLEAN", {"default": True, "tooltip": "Если активно, слайдер 't' игнорируется и используются Strength из Stacker."})}}
    RETURN_TYPES = ("ACESTEP_MERGE_METHOD",); FUNCTION = "get_method"; CATEGORY = "ACE-Step/Smart Merger/Methods"; TITLE = "ACE-Step Method: SLERP"
    def get_method(self, t, use_stack_weights): return ({"name": "slerp", "t": None if use_stack_weights else t, "normalize": False},)

class AceStep_MergeMethodTIES:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"density": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}), "normalize": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("ACESTEP_MERGE_METHOD",); FUNCTION = "get_method"; CATEGORY = "ACE-Step/Smart Merger/Methods"; TITLE = "ACE-Step Method: TIES"
    def get_method(self, density, normalize): return ({"name": "ties", "density": density, "normalize": normalize},)

class AceStep_MergeMethodDARE:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"density": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}), "use_sign_consensus": ("BOOLEAN", {"default": True}), "normalize": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("ACESTEP_MERGE_METHOD",); FUNCTION = "get_method"; CATEGORY = "ACE-Step/Smart Merger/Methods"; TITLE = "ACE-Step Method: DARE"
    def get_method(self, density, use_sign_consensus, normalize): return ({"name": "dare", "density": density, "consensus": use_sign_consensus, "normalize": normalize},)

class AceStep_MergeMethodDELLA:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"density": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}), "epsilon": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}), "use_sign_consensus": ("BOOLEAN", {"default": True}), "normalize": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("ACESTEP_MERGE_METHOD",); FUNCTION = "get_method"; CATEGORY = "ACE-Step/Smart Merger/Methods"; TITLE = "ACE-Step Method: DELLA"
    def get_method(self, density, epsilon, use_sign_consensus, normalize): return ({"name": "della", "density": density, "epsilon": epsilon, "consensus": use_sign_consensus, "normalize": normalize},)

class AceStep_MergeMethodBreadcrumbs:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"density": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}), "gamma": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}), "use_sign_consensus": ("BOOLEAN", {"default": True}), "normalize": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("ACESTEP_MERGE_METHOD",); FUNCTION = "get_method"; CATEGORY = "ACE-Step/Smart Merger/Methods"; TITLE = "ACE-Step Method: Breadcrumbs"
    def get_method(self, density, gamma, use_sign_consensus, normalize): return ({"name": "breadcrumbs", "density": density, "gamma": gamma, "consensus": use_sign_consensus, "normalize": normalize},)

class AceStep_MergeMethodModelStock:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"normalize": ("BOOLEAN", {"default": True, "tooltip": "Математически рассчитывает идеальный баланс."})}}
    RETURN_TYPES = ("ACESTEP_MERGE_METHOD",); FUNCTION = "get_method"; CATEGORY = "ACE-Step/Smart Merger/Methods"; TITLE = "ACE-Step Method: Model Stock"
    def get_method(self, normalize): return ({"name": "model_stock", "normalize": normalize},)


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
                "merge_method": ("ACESTEP_MERGE_METHOD",),
                "target_rank": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8, "tooltip": "0 = Авто (использует макс. ранг из входных). Ранг сжатия (SVD) для математических методов."}),
                "svd_method": (["rSVD", "energy_rSVD"], {"default": "rSVD"}),
                "ignore_bias": ("BOOLEAN", {"default": True, "tooltip": "Игнорировать смещения (Bias). Спасает от гула."}),
                "global_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Общий множитель Лоры."}),
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

    def smart_merge(self, model, lora_stack, merge_method, target_rank, svd_method, ignore_bias, global_scale):
        if not lora_stack: return (model, {})
        
        meth_name = merge_method["name"]
        normalize = merge_method.get("normalize", True)

        print(f"\n[Smart Lora Merger] 🧠 Слияние: {len(lora_stack)} LoRA | Метод: {meth_name}")
        
        loaded_sds =[]
        for lora in lora_stack:
            sd = self._load_lora_weights(lora["path"])
            if sd is None: raise FileNotFoundError(f"Не удалось загрузить веса из: {lora['path']}")
            
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
        
        final_rank = 0
        norms_log = {item["name"]:[] for item in loaded_sds}

        def process_layer(base_key):
            try:
                # -------------------------------------------------------------
                # РЕЖИМ 1: CONCAT (Без потерь, Идеально для музыки)
                # -------------------------------------------------------------
                if meth_name == "concat":
                    A_tensors = []
                    B_tensors =[]
                    orig_dtype = torch.float32
                    layer_norms =[]
                    
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
                            
                            W = (B_gpu @ A_gpu) * item["scale"]
                            current_norm = torch.norm(W).item()
                            layer_norms.append(current_norm)
                            norms_log[item["name"]].append(current_norm)
                            
                            A_tensors.append(A_gpu)
                            B_tensors.append(B_gpu)
                        else:
                            layer_norms.append(0.0)
                            A_tensors.append(None)
                            B_tensors.append(None)
                            
                    valid_A =[]
                    valid_B = []
                    
                    active_norms =[n for n in layer_norms if n > 0]
                    target_norm = sum(active_norms) / len(active_norms) if active_norms else 1.0

                    for i, item in enumerate(loaded_sds):
                        if A_tensors[i] is not None:
                            A_gpu = A_tensors[i]
                            B_gpu = B_tensors[i]
                            
                            mult = 1.0
                            if normalize and len(active_norms) > 1:
                                raw_multiplier = target_norm / (layer_norms[i] + 1e-8)
                                mult = max(0.7, min(1.5, raw_multiplier))
                            
                            B_scaled = B_gpu * item["scale"] * item["strength"] * global_scale * mult
                            valid_A.append(A_gpu)
                            valid_B.append(B_scaled)
                            
                    if not valid_A: return base_key, None, None, None, 0
                        
                    A_merged = torch.cat(valid_A, dim=0).cpu().to(orig_dtype)
                    B_merged = torch.cat(valid_B, dim=1).cpu().to(orig_dtype)
                    return base_key, A_merged, B_merged, orig_dtype, A_merged.shape[0]

                # -------------------------------------------------------------
                # РЕЖИМ 2: Математический (SVD)
                # -------------------------------------------------------------
                W_list = []
                valid_strengths =[]
                orig_dtype = torch.float32
                layer_norms = []
                layer_ranks =[]
                
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
                        
                        layer_ranks.append(A.shape[0])
                        W_list.append(W)
                        valid_strengths.append(item["strength"])
                        
                if not W_list: return base_key, None, None, None, 0

                # Нормализация весов (защита от выгорания)
                if normalize and len(valid_strengths) > 1 and meth_name not in ["model_stock"]:
                    tw = sum(abs(w) for w in valid_strengths)
                    if tw > 0: valid_strengths =[w / tw for w in valid_strengths]

                # Выполнение мержа с помощью выбранного математического метода
                if meth_name == "linear":
                    W_merged = linear_merge(W_list, valid_strengths)
                elif meth_name == "slerp":
                    W_merged = slerp_merge(W_list, valid_strengths, merge_method.get("t"))
                elif meth_name == "model_stock":
                    W_merged = model_stock_merge(W_list, valid_strengths)
                else:
                    cons = merge_method.get("consensus", True)
                    if meth_name == "ties":
                        W_merged = sign_consensus_merge([ties_sparsify(t, merge_method.get("density", 0.9)) for t in W_list], valid_strengths)
                    elif meth_name == "dare":
                        sparse =[dare_sparsify(t, merge_method.get("density", 0.9)) for t in W_list]
                        W_merged = sign_consensus_merge(sparse, valid_strengths) if cons else linear_merge(sparse, valid_strengths)
                    elif meth_name == "della":
                        sparse =[della_sparsify(t, merge_method.get("density", 0.9), merge_method.get("epsilon", 0.1)) for t in W_list]
                        W_merged = sign_consensus_merge(sparse, valid_strengths) if cons else linear_merge(sparse, valid_strengths)
                    elif meth_name == "breadcrumbs":
                        sparse =[breadcrumbs_sparsify(t, merge_method.get("density", 0.9), merge_method.get("gamma", 0.01)) for t in W_list]
                        W_merged = sign_consensus_merge(sparse, valid_strengths) if cons else linear_merge(sparse, valid_strengths)
                    else:
                        W_merged = linear_merge(W_list, valid_strengths)

                W_merged = W_merged * global_scale

                # Динамическое вычисление ранка (target_rank = 0 означает использование максимального)
                calc_rank = target_rank if target_rank > 0 else (max(layer_ranks) if layer_ranks else 128)

                if svd_method == "energy_rSVD": A_merged, B_merged = apply_energy_rsvd(W_merged, calc_rank)
                else: A_merged, B_merged = apply_rsvd(W_merged, calc_rank)

                return base_key, A_merged.cpu().to(orig_dtype), B_merged.cpu().to(orig_dtype), orig_dtype, calc_rank
                
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
                    if new_r > final_rank: final_rank = new_r
                pbar.update(1)

        torch.cuda.empty_cache()

        if loaded_sds:
            print("\n[Smart Lora Merger] 📊 Сила Лоры (Средняя Норма матрицы) до мержа:")
            for name, n_list in norms_log.items():
                if n_list: print(f"   - {name}: {sum(n_list)/len(n_list):.2f}")

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
                
                if normalize and len(valid_strengths) > 1 and meth_name not in["model_stock", "concat"]:
                    tw = sum(abs(w) for w in valid_strengths)
                    if tw > 0: valid_strengths = [w / tw for w in valid_strengths]
                
                merged_val = linear_merge(t_list, valid_strengths) * global_scale
                merged_state_dict[k] = merged_val.to(orig_dtype)

        print(f"\n[Smart Lora Merger] ✅ Тензоры успешно слиты! Итоговый ранг (dim): {final_rank}")

        # Инъекция в систему
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
    "AceStep_MergeMethodConcat": AceStep_MergeMethodConcat,
    "AceStep_MergeMethodLinear": AceStep_MergeMethodLinear,
    "AceStep_MergeMethodSLERP": AceStep_MergeMethodSLERP,
    "AceStep_MergeMethodTIES": AceStep_MergeMethodTIES,
    "AceStep_MergeMethodDARE": AceStep_MergeMethodDARE,
    "AceStep_MergeMethodDELLA": AceStep_MergeMethodDELLA,
    "AceStep_MergeMethodBreadcrumbs": AceStep_MergeMethodBreadcrumbs,
    "AceStep_MergeMethodModelStock": AceStep_MergeMethodModelStock,
    "AceStepSmartLoraMerger": AceStepSmartLoraMerger,
    "AceStepSaveMergedLora": AceStepSaveMergedLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepLoraStackEntry": "ACE-Step LoRA Stack Entry 📚",
    "AceStep_MergeMethodConcat": "ACE-Step Method: Concat (Lossless) 🧲",
    "AceStep_MergeMethodLinear": "ACE-Step Method: Linear 📈",
    "AceStep_MergeMethodSLERP": "ACE-Step Method: SLERP 🧬",
    "AceStep_MergeMethodTIES": "ACE-Step Method: TIES ⚡",
    "AceStep_MergeMethodDARE": "ACE-Step Method: DARE 🎲",
    "AceStep_MergeMethodDELLA": "ACE-Step Method: DELLA 🧠",
    "AceStep_MergeMethodBreadcrumbs": "ACE-Step Method: Breadcrumbs 🍞",
    "AceStep_MergeMethodModelStock": "ACE-Step Method: Model Stock 🏗️",
    "AceStepSmartLoraMerger": "ACE-Step Smart LoRA Merger 🧠",
    "AceStepSaveMergedLora": "ACE-Step Save Merged LoRA 💾",
}