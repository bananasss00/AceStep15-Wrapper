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
import torch.nn.functional as F
from io import BytesIO

# Для графиков
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Для SVD Resize
from safetensors.torch import load_file, save_file

import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
from server import PromptServer

# Пытаемся импортировать библиотеку ACE-Step
try:
    from acestep.training_v2.configs import LoRAConfigV2, LoKRConfigV2, TrainingConfigV2
    from acestep.training_v2.trainer_fixed import FixedLoRATrainer
    from acestep.training_v2.model_loader import load_decoder_for_training
    from acestep.training_v2.preprocess import preprocess_audio_files
    from acestep.training_v2.gpu_utils import detect_gpu
    from acestep.training_v2.cli.validation import resolve_target_modules
    from acestep.training_v2.estimate import run_estimation
    from acestep.training_v2.fixed_lora_module import FixedLoRAModule
    from acestep.training_v2.timestep_sampling import apply_cfg_dropout, sample_timesteps
    from contextlib import nullcontext
except ImportError as e:
    print(f"⚠️[ACE-Step] Failed to import acestep modules: {e}")

# ============================================================================
try:
    torch.set_float32_matmul_precision('medium')
except:
    pass
# ============================================================================

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
# ============================================================================

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
# ============================================================================

# ======================================================================
# 1. DATASET CONFIG NODE
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
            },
            "optional": {
                "reg_data_source": ("STRING", {"default": "", "tooltip": "Path to regularization audio/json"}),
                "reg_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Loss weight for reg samples"}),
            }
        }
    RETURN_TYPES = ("ACESTEP_DATASET",)
    FUNCTION = "get_config"
    CATEGORY = "ACE-Step/Configs"

    def get_config(self, **kwargs):
        return (kwargs,)


# ======================================================================
# 2. MODEL & LORA CONFIG NODE
# ======================================================================
class ACEStepModelConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (["turbo", "base", "sft"], {"default": "turbo"}),
                "rank": ("INT", {"default": 64, "min": 1, "max": 1024}),
                "alpha": ("INT", {"default": 128, "min": 1, "max": 2048}),
                "dropout": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target_modules": ("STRING", {"default": "q_proj k_proj v_proj o_proj", "multiline": True}),
                "attn_type": (["both", "self", "cross"], {"default": "both"}),
                "bias": (["none", "lora_only", "all"], {"default": "none"}),
                "inf_steps": ("INT", {"default": 8, "min": 1}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "step": 0.1}),
                "cfg_ratio": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("ACESTEP_MODEL",)
    FUNCTION = "get_config"
    CATEGORY = "ACE-Step/Configs"

    def get_config(self, **kwargs):
        kwargs["adapter_type"] = "lora"
        return (kwargs,)

# ======================================================================
# 2.5 MODEL & LOKR CONFIG NODE
# ======================================================================
class ACEStepLoKRConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (["turbo", "base", "sft"], {"default": "turbo"}),
                "linear_dim": ("INT", {"default": 64, "min": 1, "max": 1024}),
                "linear_alpha": ("INT", {"default": 128, "min": 1, "max": 2048}),
                "factor": ("INT", {"default": -1, "min": -1, "max": 256, "tooltip": "-1 means auto factor"}),
                "decompose_both": ("BOOLEAN", {"default": False}),
                "use_tucker": ("BOOLEAN", {"default": False}),
                "use_scalar": ("BOOLEAN", {"default": False}),
                "weight_decompose": ("BOOLEAN", {"default": False, "tooltip": "DoRA-style weight decomposition"}),
                "target_modules": ("STRING", {"default": "q_proj k_proj v_proj o_proj", "multiline": True}),
                "attn_type": (["both", "self", "cross"], {"default": "both"}),
                "inf_steps": ("INT", {"default": 8, "min": 1}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "step": 0.1}),
                "cfg_ratio": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("ACESTEP_MODEL",) # Тот же тип, чтобы подключалось к тренеру
    FUNCTION = "get_config"
    CATEGORY = "ACE-Step/Configs"

    def get_config(self, **kwargs):
        kwargs["adapter_type"] = "lokr" # Указываем, что это LoKR
        return (kwargs,)
    
# ======================================================================
# 3. OPTIMIZER CONFIG NODES
# ======================================================================
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
            "beta3": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 0.99, "step": 0.01, "tooltip": "Set to -1.0 for Auto"}), 
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


# ======================================================================
# 4. PREVIEW CONFIG NODE
# ======================================================================
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
                indices_str = f"⚠️ Data source not found:\n{clean_source}\nPlease provide a valid directory or .json file in 'Dataset Config'."
            else:
                basenames =[]
                
                # Если передан JSON
                if clean_source.lower().endswith('.json'):
                    try:
                        with open(clean_source, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            # Умный поиск семплов в зависимости от структуры JSON
                            samples_list =[]
                            if isinstance(data, dict) and "samples" in data and isinstance(data["samples"], list):
                                # Формат ACE-Step V2: {"metadata": {...}, "samples": [{...}, {...}]}
                                samples_list = data["samples"]
                            elif isinstance(data, list):
                                # Простой массив: [{...}, {...}]
                                samples_list = data
                            elif isinstance(data, dict):
                                # Словарь где ключи это ID: {"id1": {"audio_path": "..."}, ...}
                                samples_list =[v for v in data.values() if isinstance(v, dict)]
                            
                            # Извлекаем пути к файлам
                            for item in samples_list:
                                if isinstance(item, dict):
                                    path_key = next((k for k in["audio_path", "audio", "path", "file", "filename"] if k in item), None)
                                    if path_key and item[path_key]:
                                        basenames.append(os.path.basename(item[path_key]))
                    except Exception as e:
                        indices_str = f"⚠️ Error reading JSON: {e}"
                        return (kwargs, indices_str)
                        
                # Если передана директория с аудио
                elif os.path.isdir(clean_source):
                    valid_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.opus'}
                    try:
                        for f in os.listdir(clean_source):
                            if os.path.isfile(os.path.join(clean_source, f)):
                                ext = os.path.splitext(f)[1].lower()
                                if ext in valid_exts:
                                    basenames.append(f)
                    except Exception as e:
                        indices_str = f"⚠️ Error reading directory: {e}"
                        return (kwargs, indices_str)

                if not basenames:
                    indices_str = f"⚠️ No valid audio files found in data source:\n{clean_source}"
                else:
                    # В тренере файлы считываются через sorted(glob(".../*.pt"))
                    # Эмулируем это: генерируем имена .pt файлов и сортируем по алфавиту
                    pt_names = [os.path.splitext(b)[0] + ".pt" for b in basenames]
                    pt_names = sorted(list(set(pt_names)))
                    
                    lines =[f"🔮 Predicted Training Order (Files: {len(pt_names)})", "="*60]
                    for idx, pt in enumerate(pt_names):
                        lines.append(f"[{idx}] ➔ {pt}")
                    
                    indices_str = "\n".join(lines)

        return (kwargs, indices_str)


# ======================================================================
# 5. ESTIMATOR NODE
# ======================================================================
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
        """Проверяет актуальность кеша. Если JSON/аудио новее тензоров, сносит старый кеш."""
        if not os.path.exists(tensor_dir):
            return False

        pt_files = glob.glob(os.path.join(tensor_dir, "*.pt"))
        if not pt_files:
            return False

        # Если источник не передан, просто верим кешу
        if not source_path or not os.path.exists(source_path):
            return True

        # Берем дату самого старого тензора (любое изменение ПОСЛЕ генерации первого тензора = кеш инвалид)
        oldest_tensor_time = min(os.path.getmtime(f) for f in pt_files)
        source_mtime = 0

        # Узнаем дату изменения исходника
        if os.path.isfile(source_path):
            source_mtime = os.path.getmtime(source_path)
        elif os.path.isdir(source_path):
            source_mtime = os.path.getmtime(source_path)
            for root, _, files in os.walk(source_path):
                for file in files:
                    if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.opus')):
                        fpath = os.path.join(root, file)
                        source_mtime = max(source_mtime, os.path.getmtime(fpath))

        # Если источник изменился позже, чем были созданы тензоры
        if source_mtime > oldest_tensor_time:
            print(f"⚠️ [Cache] Source data '{os.path.basename(source_path)}' was modified. Invalidating old tensors...")
            for f in pt_files:
                try: os.remove(f)
                except: pass
            # Удаляем манифест на всякий случай, если он остался от v1
            manifest_path = os.path.join(tensor_dir, "manifest.json")
            if os.path.exists(manifest_path):
                try: os.remove(manifest_path)
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
            gpu_info = detect_gpu("auto", "auto")

            if not self._check_and_prepare_cache(clean_source, final_tensor_dir):
                if not clean_source:
                    raise ValueError(f"❌ Tensors not found and no Data Source provided!")
                print(f"🔨 Preprocessing required for estimation...")
                try:
                    is_json = clean_source.lower().endswith('.json')
                    preprocess_audio_files(
                        audio_dir=None if is_json else clean_source,
                        dataset_json=clean_source if is_json else None,
                        output_dir=final_tensor_dir,
                        checkpoint_dir=checkpoint_dir,
                        variant=model_config["model_variant"],
                        max_duration=dataset_config["max_duration"],
                        device=gpu_info.device,
                        precision=gpu_info.precision,
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
                
                top_modules_raw = [item["module"] for item in results]
                top_modules = [m.replace("decoder.", "") for m in top_modules_raw]
                result_str = " ".join(top_modules)
                
                print(f"✅ Estimation Complete. Top {top_k} modules found.")
                print(f"📋 Result: {result_str}")
                
                mm.soft_empty_cache()
                return (result_str,)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"❌ Estimation failed: {e}")


# ======================================================================
# 6. LORA RESIZE NODE
# ======================================================================
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
        try:
            U, S, Vh = torch.linalg.svd(w_float, full_matrices=False)
        except Exception as e:
            print(f"Error during SVD: {e}. Falling back to CPU.")
            w_float = w_float.cpu()
            U, S, Vh = torch.linalg.svd(w_float, full_matrices=False)
        return U, S, Vh

    def calculate_new_rank(self, S: torch.Tensor, method: str, param: float, fixed_rank: int) -> int:
        if method == "None" or method == "safe":
            return min(fixed_rank, len(S))
        elif method == "sv_ratio":
            threshold = S[0] * param
            keep_indices = torch.nonzero(S >= threshold).flatten()
            if len(keep_indices) == 0: return 1
            return keep_indices[-1].item() + 1
        elif method == "sv_fro":
            S_sq = S.pow(2)
            sum_S_sq = torch.sum(S_sq)
            cumulative_S_sq = torch.cumsum(S_sq, dim=0)
            threshold = param * sum_S_sq
            keep_indices = torch.nonzero(cumulative_S_sq >= threshold).flatten()
            if len(keep_indices) == 0: return len(S) 
            return keep_indices[0].item() + 1
        return fixed_rank

    def resize(self, input_path, output_path, new_rank, dynamic_method, dynamic_param, precision, device):
        print(f"\n" + "="*60)
        print(f"📉 LoRA Resize Started")
        print(f"="*60)
        
        in_path = input_path.strip('"')
        out_path = output_path.strip('"')

        if not os.path.exists(in_path):
            raise FileNotFoundError(f"Input path not found: {in_path}")
        
        if not out_path:
            out_path = in_path + "_resized"
        
        settings_log = f"⚙️  Settings: Method={dynamic_method}"
        if dynamic_method in ["sv_fro", "sv_ratio"]:
            settings_log += f", Threshold/Ratio={dynamic_param}, MaxRank={new_rank}"
        else:
            settings_log += f", TargetRank={new_rank}"
        
        print(settings_log)
        print(f"🖥️  Device: {device}, Precision: {precision}")
        print(f"📂 Input: {in_path}")
        print(f"📂 Output: {out_path}")
        
        os.makedirs(out_path, exist_ok=True)
        
        config_path = os.path.join(in_path, "adapter_config.json")
        model_path = os.path.join(in_path, "adapter_model.safetensors")
        
        if not os.path.exists(model_path):
             model_path = os.path.join(in_path, "adapter_model.bin")
             use_safetensors = False
             if not os.path.exists(model_path):
                 raise FileNotFoundError("Could not find adapter_model.safetensors or .bin")
        else:
            use_safetensors = True

        with open(config_path, 'r') as f:
            config = json.load(f)

        if use_safetensors:
            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu")

        old_r = config.get("r", 8)
        old_alpha = config.get("lora_alpha", 8)
        scaling_factor = old_alpha / old_r if old_r > 0 else 1.0
        
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
        
        # ProgressBar
        total_steps = len(pairs)
        pbar = ProgressBar(total_steps)
        
        print(f"🚀 Processing {total_steps} layers...")
        
        for base_key, mats in pairs.items():
            pbar.update(1)
            
            if 'A' not in mats or 'B' not in mats:
                for k, v in mats.items():
                    new_state_dict[f"{base_key}lora_{k}.weight"] = v.to(save_dtype)
                continue

            A = mats['A'].to(device)
            B = mats['B'].to(device)
            
            W = (B @ A) * scaling_factor
            U, S, Vh = self.perform_svd(W, device)
            
            target_rank = self.calculate_new_rank(S, dynamic_method, dynamic_param, new_rank)
            target_rank = min(target_rank, old_r)
            
            layer_name = base_key.replace("base_model.model.", "").rstrip(".")
            if dynamic_method in ["sv_fro", "sv_ratio"]:
                rank_pattern[layer_name] = target_rank

            U_r = U[:, :target_rank]
            S_r = S[:target_rank]
            Vh_r = Vh[:target_rank, :]
            sqrt_S = torch.sqrt(S_r)
            
            new_B = U_r @ torch.diag(sqrt_S)
            new_A = torch.diag(sqrt_S) @ Vh_r
            
            new_state_dict[f"{base_key}lora_A.weight"] = new_A.to(save_dtype).cpu()
            new_state_dict[f"{base_key}lora_B.weight"] = new_B.to(save_dtype).cpu()
            
            del A, B, W, U, S, Vh, U_r, S_r, Vh_r, sqrt_S, new_A, new_B

        new_config = copy.deepcopy(config)
        if dynamic_method in ["sv_fro", "sv_ratio"] and rank_pattern:
            new_config["rank_pattern"] = rank_pattern
            new_config["alpha_pattern"] = rank_pattern 
            new_config["r"] = max(rank_pattern.values()) 
            new_config["lora_alpha"] = new_config["r"]
            avg_rank = sum(rank_pattern.values()) / len(rank_pattern)
            print(f"📊 Resized with dynamic rank. Avg Rank: {avg_rank:.2f}")
        else:
            new_config["r"] = new_rank
            new_config["lora_alpha"] = new_rank
            new_config.pop("rank_pattern", None)
            new_config.pop("alpha_pattern", None)

        with open(os.path.join(out_path, "adapter_config.json"), 'w') as f:
            json.dump(new_config, f, indent=2)
            
        if use_safetensors:
            save_file(new_state_dict, os.path.join(out_path, "adapter_model.safetensors"))
        else:
            torch.save(new_state_dict, os.path.join(out_path, "adapter_model.bin"))

        print(f"✅ Resize complete! Saved to {out_path}")
        mm.soft_empty_cache()
        return (out_path,)


# ======================================================================
# 7. MAIN TRAINER NODE
# ======================================================================
# ======================================================================
# ЗАПЛАТКА (MONKEY-PATCH) ДЛЯ ПОДДЕРЖКИ ВЕСА РЕГУЛЯРИЗАЦИИ
# ======================================================================
CURRENT_REG_WEIGHT = 1.0

def custom_training_step(self, batch: dict) -> torch.Tensor:
    global CURRENT_REG_WEIGHT
    
    if self.device_type in ("cuda", "xpu", "mps"):
        autocast_ctx = torch.autocast(device_type=self.device_type, dtype=self.dtype)
    else:
        autocast_ctx = nullcontext()

    with autocast_ctx:
        nb = self.transfer_non_blocking
        target_latents = batch["target_latents"].to(self.device, dtype=self.dtype, non_blocking=nb)
        attention_mask = batch["attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)
        encoder_hidden_states = batch["encoder_hidden_states"].to(self.device, dtype=self.dtype, non_blocking=nb)
        encoder_attention_mask = batch["encoder_attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)
        context_latents = batch["context_latents"].to(self.device, dtype=self.dtype, non_blocking=nb)

        bsz = target_latents.shape[0]

        if self._null_cond_emb is not None and self._cfg_ratio > 0.0:
            encoder_hidden_states = apply_cfg_dropout(
                encoder_hidden_states, self._null_cond_emb, cfg_ratio=self._cfg_ratio
            )

        x1 = torch.randn_like(target_latents)
        x0 = target_latents

        t, r = sample_timesteps(
            batch_size=bsz, device=self.device, dtype=self.dtype,
            data_proportion=self._data_proportion, timestep_mu=self._timestep_mu,
            timestep_sigma=self._timestep_sigma, use_meanflow=False
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

        # --- НАША КАСТОМНАЯ ЛОГИКА РЕГУЛЯРИЗАЦИИ ---
        # Вычисляем ошибку без усреднения (reduction='none')
        unreduced_loss = F.mse_loss(decoder_outputs[0], flow, reduction='none')
        
        # Усредняем по размерностям Sequence и Features, оставляя только размерность Batch -> [B]
        loss_per_sample = unreduced_loss.reshape(bsz, -1).mean(dim=1)

        # Достаем флаг is_reg из метадаты (если он есть)
        metadata = batch.get("metadata",[])
        weights = torch.ones(bsz, device=self.device, dtype=self.dtype)

        for i in range(bsz):
            meta = metadata[i] if i < len(metadata) else {}
            if meta.get("is_reg", False):
                weights[i] = CURRENT_REG_WEIGHT

        # Применяем веса регуляризации и усредняем по батчу
        diffusion_loss = (loss_per_sample * weights).mean()
        unweighted_loss = loss_per_sample.mean()
        # ---------------------------------------------

    diffusion_loss = diffusion_loss.float()
    self.training_losses.append(unweighted_loss.item())
    return diffusion_loss

# Подменяем функцию в классе:
try:
    FixedLoRAModule.training_step = custom_training_step
except NameError:
    pass # Защита, если импорт ACE-Step не удался

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
        """Проверяет актуальность кеша. Если JSON/аудио новее тензоров, сносит старый кеш."""
        if not os.path.exists(tensor_dir):
            return False

        pt_files = glob.glob(os.path.join(tensor_dir, "*.pt"))
        if not pt_files:
            return False

        if not source_path or not os.path.exists(source_path):
            return True

        oldest_tensor_time = min(os.path.getmtime(f) for f in pt_files)
        source_mtime = 0

        if os.path.isfile(source_path):
            source_mtime = os.path.getmtime(source_path)
        elif os.path.isdir(source_path):
            source_mtime = os.path.getmtime(source_path)
            for root, _, files in os.walk(source_path):
                for file in files:
                    if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.opus')):
                        fpath = os.path.join(root, file)
                        source_mtime = max(source_mtime, os.path.getmtime(fpath))

        if source_mtime > oldest_tensor_time:
            print(f"⚠️ [Cache] Source data '{os.path.basename(source_path)}' was modified. Invalidating old tensors...")
            for f in pt_files:
                try: os.remove(f)
                except: pass
            
            manifest_path = os.path.join(tensor_dir, "manifest.json")
            if os.path.exists(manifest_path):
                try: os.remove(manifest_path)
                except: pass
            return False

        return True

    def process_loss_graph(self, epochs, losses, emas, lrs, elapsed_str="", eta_str="", step_time_str="", epoch_time_str="", saved_epochs=None, node_id=None, output_dir=None):
        """Универсальная функция для отрисовки, отправки в UI и сохранения графика."""
        if saved_epochs is None:
            saved_epochs =[]
            
        fig = Figure(figsize=(4.8, 4.2), dpi=100, facecolor='#2b2b2b')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        ax.grid(True, linestyle='--', color='#444444', alpha=0.3)
        ax.tick_params(colors='#e0e0e0', labelsize=8)
        for spine in ax.spines.values(): spine.set_color('#444444')
        
        for se in saved_epochs:
            ax.axvline(x=se, color='#aaaaaa', linestyle=':', linewidth=1.2, alpha=0.7)
            
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
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, 
                  facecolor='#2b2b2b', edgecolor='#444444', labelcolor='#e0e0e0', fontsize=8)
        
        fig.subplots_adjust(bottom=0.25, top=0.82, left=0.12, right=0.88)
        
        canvas = FigureCanvasAgg(fig)
        
        # Отправка в UI
        if node_id is not None:
            buf = BytesIO()
            canvas.print_png(buf)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            PromptServer.instance.send_sync("acestep_loss_update", {"node": node_id, "image": f"data:image/png;base64,{b64}"})
            
        # Сохранение на диск
        if output_dir is not None:
            try:
                path = os.path.join(output_dir, "loss_graph.png")
                canvas.print_png(path)
            except Exception as e:
                print(f"⚠️ Failed to save graph to disk: {e}")
                
        fig.clf()

    @torch.inference_mode(False)
    def train_model(self, dataset_config, model_config, optimizer_config, seed, grad_ckpt, offload_enc, vram_cleanup, preview_config=None, unique_id=None, prompt=None, extra_pnginfo=None):
        with torch.enable_grad():
            print("\n" + "="*60)
            print("🚀 ACE-Step Training Native Node (ComfyUI)")
            print("="*60)

            clean_source = dataset_config["data_source"].strip('"')
            tensor_root = dataset_config["tensor_root"].strip('"')
            checkpoint_dir = dataset_config["checkpoint_dir"].strip('"')
            output_dir = dataset_config["output_dir"].strip('"')
            
            # --- ИЗВЛЕЧЕНИЕ НАСТРОЕК РЕГУЛЯРИЗАЦИИ ---
            reg_source = dataset_config.get("reg_data_source", "").strip('"')
            reg_weight = dataset_config.get("reg_weight", 1.0)
            
            global CURRENT_REG_WEIGHT
            CURRENT_REG_WEIGHT = reg_weight
            # -----------------------------------------

            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            if extra_pnginfo and "workflow" in extra_pnginfo:
                print(f"💾 Saving workflow to {output_dir}/workflow.json")
                try:
                    with open(os.path.join(output_dir, "workflow.json"), "w", encoding="utf-8") as f:
                        json.dump(extra_pnginfo["workflow"], f, indent=2)
                except Exception as e:
                    print(f"⚠️ Failed to save workflow: {e}")

            dataset_name = "default_dataset"
            if clean_source:
                dataset_name = os.path.splitext(os.path.basename(os.path.normpath(clean_source)))[0]
                dataset_name = re.sub(r'[\\/*?:"<>|]', "", dataset_name).replace(" ", "_")
            
            main_tensor_dir = os.path.join(tensor_root, dataset_name)
            gpu_info = detect_gpu("auto", "auto")

            # === [PHASE 1] PREPROCESSING MAIN DATASET ===
            if not self._check_and_prepare_cache(clean_source, main_tensor_dir):
                if not clean_source:
                    raise ValueError(f"❌ Tensors not found and no Data Source provided!")
                print(f"🔨 [Phase 1] Preprocessing Main Dataset...")
                try:
                    is_json = clean_source.lower().endswith('.json')
                    preprocess_audio_files(
                        audio_dir=None if is_json else clean_source,
                        dataset_json=clean_source if is_json else None,
                        output_dir=main_tensor_dir,
                        checkpoint_dir=checkpoint_dir,
                        variant=model_config["model_variant"],
                        max_duration=dataset_config["max_duration"],
                        device=gpu_info.device,
                        precision=gpu_info.precision,
                        progress_callback=lambda c,t,m: print(f"[PRE-MAIN] {m}")
                    )
                except Exception as e:
                    raise RuntimeError(f"❌ Main Preprocessing failed: {e}")
            else:
                print(f"📂 [Phase 1] Valid cache found. Using existing main tensors in: {main_tensor_dir}")

            # ===[PHASE 1.5] PREPROCESSING REG DATASET ===
            reg_tensor_dir = None
            if reg_source and os.path.exists(reg_source):
                reg_dataset_name = os.path.splitext(os.path.basename(os.path.normpath(reg_source)))[0]
                reg_dataset_name = re.sub(r'[\\/*?:"<>|]', "", reg_dataset_name).replace(" ", "_")
                reg_tensor_dir = os.path.join(tensor_root, reg_dataset_name + "_REG")

                if not self._check_and_prepare_cache(reg_source, reg_tensor_dir):
                    print(f"🔨 [Phase 1.5] Preprocessing Regularization Data...")
                    is_json = reg_source.lower().endswith('.json')
                    preprocess_audio_files(
                        audio_dir=None if is_json else reg_source,
                        dataset_json=reg_source if is_json else None,
                        output_dir=reg_tensor_dir,
                        checkpoint_dir=checkpoint_dir,
                        variant=model_config["model_variant"],
                        max_duration=dataset_config["max_duration"],
                        device=gpu_info.device,
                        precision=gpu_info.precision,
                        progress_callback=lambda c,t,m: print(f"[PRE-REG] {m}")
                    )
                else:
                    print(f"📂 [Phase 1.5] Valid cache found. Using existing REG tensors in: {reg_tensor_dir}")

            tensor_files = glob.glob(os.path.join(main_tensor_dir, "*.pt"))
            dataset_len = len(tensor_files)
            if dataset_len == 0:
                raise ValueError("❌ No .pt files found! Cannot proceed.")

            resolved_modules = resolve_target_modules(model_config["target_modules"].split(), model_config["attn_type"])
            adapter_type = model_config.get("adapter_type", "lora")

            if adapter_type == "lokr":
                adapter_cfg = LoKRConfigV2(
                    linear_dim=model_config["linear_dim"],
                    linear_alpha=model_config["linear_alpha"],
                    factor=model_config["factor"],
                    decompose_both=model_config["decompose_both"],
                    use_tucker=model_config["use_tucker"],
                    use_scalar=model_config["use_scalar"],
                    weight_decompose=model_config["weight_decompose"],
                    target_modules=resolved_modules,
                    attention_type=model_config["attn_type"]
                )
            else:
                adapter_cfg = LoRAConfigV2(
                    r=model_config["rank"], 
                    alpha=model_config["alpha"], 
                    dropout=model_config["dropout"], 
                    target_modules=resolved_modules, 
                    bias=model_config["bias"], 
                    attention_type=model_config["attn_type"]
                )

            prev_cfg = preview_config if preview_config is not None else {}
            
            train_cfg = TrainingConfigV2(
                adapter_type=adapter_type,
                learning_rate=optimizer_config["learning_rate"], 
                batch_size=dataset_config["batch_size"], 
                gradient_accumulation_steps=dataset_config["grad_accum"],
                max_epochs=dataset_config["epochs"], 
                save_every_n_epochs=dataset_config["save_every"], 
                output_dir=output_dir, 
                seed=seed,
                resume_from=dataset_config["resume_from"] if dataset_config["resume_from"] else None, 
                optimizer_type=optimizer_config["optimizer"], 
                scheduler_type=optimizer_config["scheduler"],
                gradient_checkpointing=grad_ckpt, 
                offload_encoder=offload_enc, 
                cfg_ratio=model_config["cfg_ratio"],
                device=gpu_info.device, 
                precision=gpu_info.precision, 
                dataset_dir=main_tensor_dir,
                checkpoint_dir=checkpoint_dir, 
                model_variant=model_config["model_variant"], 
                num_workers=0 if os.name == 'nt' else 4,
                log_every=1, 
                log_heavy_every=10000, 
                weight_decay=optimizer_config["weight_decay"], 
                max_grad_norm=optimizer_config["max_grad_norm"],
                warmup_steps=optimizer_config["warmup_steps"], 
                save_start_epoch=dataset_config["save_start"], 
                save_loss_threshold=dataset_config["save_loss_limit"],
                optimizer_kwargs=optimizer_config["optimizer_kwargs"], 
                shift=model_config["shift"], 
                num_inference_steps=model_config["inf_steps"],
                generate_preview=prev_cfg.get("gen_preview", False), 
                offload_dit_for_preview=prev_cfg.get("offload_dit_prev", False), 
                preview_sample_index=prev_cfg.get("sample_idx", 0),
                preview_caption=prev_cfg.get("prev_caption", "").strip() or None, 
                preview_lyrics=prev_cfg.get("prev_lyrics", "").strip() or None,
                preview_bpm=prev_cfg.get("prev_bpm", "").strip() or None, 
                preview_keyscale=prev_cfg.get("prev_key", "").strip() or None,
                preview_timesig=prev_cfg.get("prev_ts", "").strip() or None,
            )

            print(f"🧠[Phase 2] Loading {model_config['model_variant']} model on {gpu_info.device} ({gpu_info.precision})...")
            model = load_decoder_for_training(checkpoint_dir=checkpoint_dir, variant=model_config["model_variant"], device=gpu_info.device, precision=gpu_info.precision)

            if vram_cleanup:
                print(f"🧹 [System] Aggressive VRAM Cleanup...")
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
                
                model.decoder.to(gpu_info.device)
                if hasattr(model, "null_condition_emb") and model.null_condition_emb is not None:
                    model.null_condition_emb.data = model.null_condition_emb.data.to(gpu_info.device)

            model.train()

            print("\n🔥 Starting Training Loop...")
            trainer = FixedLoRATrainer(model, adapter_cfg, train_cfg)
            trainer.main_tensor_dir = main_tensor_dir
            
            import acestep.training.data_module as dm_module
            original_setup = dm_module.PreprocessedDataModule.setup

            def balanced_setup(self_dm, stage=None):
                main_ds = dm_module.PreprocessedTensorDataset(main_tensor_dir)
                if reg_tensor_dir and os.path.exists(reg_tensor_dir):
                    reg_ds = dm_module.PreprocessedTensorDataset(reg_tensor_dir)
                    
                    class BalancedWrapper(torch.utils.data.Dataset):
                        def __init__(self, m_ds, r_ds):
                            self.m_ds = m_ds
                            self.r_ds = r_ds
                            # размер = максимум из двух * 2
                            self.target_len = max(len(m_ds), len(r_ds))
                            
                        def __len__(self):
                            return self.target_len * 2
                            
                        def __getitem__(self, idx):
                            if idx % 2 == 0:
                                return self.m_ds[(idx // 2) % len(self.m_ds)]
                            else:
                                item = dict(self.r_ds[(idx // 2) % len(self.r_ds)])
                                item["metadata"] = dict(item.get("metadata", {}))
                                item["metadata"]["is_reg"] = True
                                return item
                                
                    self_dm.train_dataset = BalancedWrapper(main_ds, reg_ds)
                    print(f"⚖️ In-Memory Dataset Balanced: Main ({len(main_ds)}) | Reg ({len(reg_ds)}) -> Total Epoch Size: {len(self_dm.train_dataset)}")
                else:
                    self_dm.train_dataset = main_ds
                self_dm.val_dataset = None

            dm_module.PreprocessedDataModule.setup = balanced_setup
            # =================================================================

            # Считаем длину для статусбара с учетом балансировки
            actual_dataset_len = dataset_len
            if reg_tensor_dir and os.path.exists(reg_tensor_dir):
                actual_dataset_len = max(dataset_len, len(glob.glob(os.path.join(reg_tensor_dir, "*.pt")))) * 2

            eff_batch = max(1, dataset_config["batch_size"] * dataset_config["grad_accum"])
            steps_per_epoch = max(1, actual_dataset_len // eff_batch)
            total_steps_approx = steps_per_epoch * dataset_config["epochs"]

            pbar = ProgressBar(total_steps_approx)
            training_state = {"should_stop": False}
            start_time = time.time()

            epoch_history = []
            loss_history =[]
            ema_history = []
            lr_history =[]
            ema_loss = None
            ema_alpha = 0.1
            saved_epochs =[]
            
            elapsed_str = "00:00:00"
            eta_str = "00:00:00"
            step_time_str = "0s/it"
            epoch_time_str = "0s/ep"

            def fmt_time(secs):
                m, s = divmod(int(max(0, secs)), 60)
                h, m = divmod(m, 60)
                return f"{h:02d}:{m:02d}:{s:02d}"

            try:
                for update in trainer.train(training_state):
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
                            if ep_h > 0: epoch_time_str = f"{ep_h}h {ep_m}m/ep"
                            elif ep_m > 0: epoch_time_str = f"{ep_m}m {ep_s}s/ep"
                            else: epoch_time_str = f"{ep_s}s/ep"
                        else:
                            eta_secs = 0
                            step_time_str = "0s/it"
                            epoch_time_str = "0s/ep"
                            
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

# ======================================================================
# REGISTRATION
# ======================================================================
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
}