import sys
import os
import tempfile
import torch
import torchaudio
import folder_paths
import hashlib
import json
import shutil
import glob
import comfy.utils
import comfy.model_management as mm
import gc

# Добавляем путь к библиотеке
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from safetensors.torch import load_file, save_file
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import generate_music, GenerationParams, GenerationConfig, format_sample

# === НАСТРОЙКА ПУТЕЙ ===
ACESTEP_MODELS_DIR = os.path.join(folder_paths.models_dir, "acestep")
if not os.path.exists(ACESTEP_MODELS_DIR):
    os.makedirs(ACESTEP_MODELS_DIR)

GLOBAL_ACESTEP_HANDLERS = []

def cleanup_all_acestep():
    global GLOBAL_ACESTEP_HANDLERS
    if not GLOBAL_ACESTEP_HANDLERS:
        return
    
    print("\n[ACE-Step] Системная очистка памяти ComfyUI. Выгрузка ACE-Step из VRAM...\n")
    for dit_handler, llm_handler in GLOBAL_ACESTEP_HANDLERS:
        # 1. Выгрузка LLM
        try:
            if llm_handler:
                llm_handler.unload()
        except Exception as e:
            print(f"[ACE-Step] Ошибка выгрузки LLM: {e}")
            
        # 2. Тотальная зачистка DiT и LoKr/LoRA
        try:
            if dit_handler:
                # А. Заставляем модель снять с себя хуки LoKr и PEFT
                if getattr(dit_handler, 'lora_loaded', False) and hasattr(dit_handler, 'unload_lora'):
                    try:
                        dit_handler.unload_lora()
                    except:
                        pass
                
                # Б. УБИВАЕМ СКРЫТЫЕ ССЫЛКИ LoKr/LoRA (Именно они вызывали утечку!)
                if hasattr(dit_handler, '_lora_service') and dit_handler._lora_service is not None:
                    dit_handler._lora_service.decoder = None
                    if hasattr(dit_handler._lora_service, 'registry'):
                        dit_handler._lora_service.registry.clear()
                    dit_handler._lora_service = None
                
                if hasattr(dit_handler, '_lora_adapter_registry') and isinstance(dit_handler._lora_adapter_registry, dict):
                    dit_handler._lora_adapter_registry.clear()
                
                if hasattr(dit_handler, '_active_loras') and isinstance(dit_handler._active_loras, dict):
                    dit_handler._active_loras.clear()

                # В. Убиваем бэкап весов в RAM (иначе RAM будет переполняться)
                if hasattr(dit_handler, '_base_decoder'):
                    dit_handler._base_decoder = None

                # Г. Удаляем основные тензоры
                dit_handler.model = None
                dit_handler.vae = None
                dit_handler.text_encoder = None
                dit_handler.silence_latent = None 
                
                # Д. Очистка MLX (для пользователей Mac)
                dit_handler.mlx_decoder = None
                dit_handler.mlx_vae = None
        except Exception as e: 
            print(f"[ACE-Step] Ошибка выгрузки DiT: {e}")
    
    GLOBAL_ACESTEP_HANDLERS.clear()
    
    gc.collect()
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    print("[ACE-Step] Память успешно освобождена.")

if not hasattr(mm, "_original_unload_all_models_acestep"):
    mm._original_unload_all_models_acestep = mm.unload_all_models

    def hooked_unload_all_models(*args, **kwargs):
        cleanup_all_acestep()
        return mm._original_unload_all_models_acestep(*args, **kwargs)
    
    mm.unload_all_models = hooked_unload_all_models

# ============================================================================
# 1. Загрузчик основной модели
# ============================================================================
class AceStepModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config_path": (["acestep-v15-turbo", "acestep-v15-base", "acestep-v15-sft"], {"default": "acestep-v15-turbo"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "init_llm": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No"}),
                "lm_model_path": (["acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-4B"], {"default": "acestep-5Hz-lm-1.7B"}),
                "lm_backend": (["vllm", "pt", "mlx"], {"default": "vllm"}),
                "use_flash_attention": ("BOOLEAN", {"default": True}),
                "offload_to_cpu": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("ACESTEP_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "ACE-Step"

    def load_model(self, config_path, device, init_llm, lm_model_path, lm_backend, use_flash_attention, offload_to_cpu):
        print(f"[ACE-Step] Инициализация. Целевая папка моделей: {ACESTEP_MODELS_DIR}")
        
        cleanup_all_acestep()
        
        dit_handler = AceStepHandler()
        llm_handler = LLMHandler()

        dit_handler._get_project_root = lambda: ACESTEP_MODELS_DIR
        project_root = ACESTEP_MODELS_DIR

        status, enable_gen = dit_handler.initialize_service(
            project_root=project_root,
            config_path=config_path,
            device=device,
            use_flash_attention=use_flash_attention,
            compile_model=False,
            offload_to_cpu=offload_to_cpu,
            offload_dit_to_cpu=offload_to_cpu
        )

        if not enable_gen:
            raise RuntimeError(f"Ошибка инициализации DiT модели: {status}")

        if init_llm:
            print(f"[ACE-Step] Загрузка LLM {lm_model_path}...")
            checkpoint_dir = os.path.join(project_root, "checkpoints")
            
            from acestep.model_downloader import ensure_lm_model
            try:
                ensure_lm_model(model_name=lm_model_path, checkpoints_dir=checkpoint_dir)
            except Exception as e:
                print(f"[ACE-Step] Ошибка авто-скачивания LLM: {e}")

            lm_status, lm_success = llm_handler.initialize(
                checkpoint_dir=checkpoint_dir,
                lm_model_path=lm_model_path,
                backend=lm_backend,
                device=device,
                offload_to_cpu=offload_to_cpu,
            )
            if not lm_success:
                print(f"[ACE-Step] Warning LLM: {lm_status}")

        # Регистрируем хендлеры в глобальном массиве для последующей очистки
        GLOBAL_ACESTEP_HANDLERS.append((dit_handler, llm_handler if llm_handler.llm_initialized else None))

        return ({"dit_handler": dit_handler, "llm_handler": llm_handler if llm_handler.llm_initialized else None, "active_adapters": {}},)

# ============================================================================
# 2. Загрузчик LoRA
# ============================================================================
class AceStepLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ACESTEP_MODEL",),
                "lora_path": ("STRING", {"default": "", "multiline": False, "placeholder": "Полный путь к папке LoRA"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "enable_lora": ("BOOLEAN", {"default": True}),
                "ignore_bias_error": ("BOOLEAN", {"default": True, "tooltip": "Фильтрует bias из весов в памяти"}),
            },
            "optional": {
                 "adapter_name_override": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("ACESTEP_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "ACE-Step"

    def load_lora(self, model, lora_path, strength, enable_lora, ignore_bias_error, adapter_name_override=""):
        if not enable_lora or not lora_path.strip():
            return (model,)

        dit_handler = model["dit_handler"]
        final_path = lora_path.strip()
        
        if not os.path.exists(final_path):
             print(f"[ACE-Step] Warning: LoRA path not found: {final_path}")
             return (model,)

        suffix = "_nb" if ignore_bias_error else ""
        if adapter_name_override.strip():
            adapter_name = adapter_name_override.strip() + suffix
        else:
            path_hash = hashlib.md5(final_path.encode()).hexdigest()[:8]
            base_name = os.path.basename(final_path.rstrip(os.sep)).split('.')[0]
            adapter_name = f"{base_name}_{path_hash}{suffix}"

        load_msg = dit_handler.add_lora(final_path, adapter_name=adapter_name, ignore_bias=ignore_bias_error)

        if "❌" in load_msg and "already loaded" not in load_msg:
            print(f"[ACE-Step] Ошибка загрузки LoRA {adapter_name}: {load_msg}")
        else:
            new_active = model["active_adapters"].copy()
            new_active[adapter_name] = float(strength)
            new_model = model.copy()
            new_model["active_adapters"] = new_active
            return (new_model,)

        return (model,)
    
# ============================================================================
# Запекание (Merge) LoRA в модель
# ============================================================================
class AceStepLoraBaker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ACESTEP_MODEL",),
                "output_dir": ("STRING", {"default": folder_paths.get_output_directory(), "multiline": False}),
                "file_name": ("STRING", {"default": "acestep_merged_model.safetensors", "multiline": False}),
            }
        }

    RETURN_TYPES = ("ACESTEP_MODEL", "STRING")
    RETURN_NAMES = ("model", "saved_path")
    FUNCTION = "bake_loras"
    CATEGORY = "ACE-Step"

    def load_lora_sd(self, path):
        """Вспомогательная функция для загрузки весов LoRA"""
        if os.path.isdir(path):
            st_path = os.path.join(path, "adapter_model.safetensors")
            bin_path = os.path.join(path, "adapter_model.bin")
            if os.path.exists(st_path): 
                return load_file(st_path)
            elif os.path.exists(bin_path): 
                return torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            if path.endswith('.safetensors'): 
                return load_file(path)
            else: 
                return torch.load(path, map_location="cpu", weights_only=True)
        return None

    def bake_loras(self, model, output_dir, file_name):
        dit_handler = model["dit_handler"]
        active_adapters = model.get("active_adapters", {})
        
        if not active_adapters:
            print("[ACE-Step Baker] ⚠️ Нет активных LoRA для запекания. Подключите AceStepLoraLoader.")
            return (model, "")
            
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, file_name)
        
        print("\n[ACE-Step Baker] 🍳 Загрузка полной базовой модели с диска...")
        
        # === ИСПРАВЛЕНИЕ: Получаем имя модели из last_init_params ===
        config_path = None
        if hasattr(dit_handler, "last_init_params") and dit_handler.last_init_params:
            config_path = dit_handler.last_init_params.get("config_path")
        
        # Если вдруг не нашли, пробуем хардкод (обычно не требуется)
        if not config_path:
            config_path = "acestep-v15-turbo" # Fallback
            print(f"[ACE-Step Baker] ⚠️ Не удалось определить имя модели, используем {config_path}")

        base_model_dir = os.path.join(dit_handler._get_project_root(), "checkpoints", config_path)
        base_model_path = os.path.join(base_model_dir, "model.safetensors")
        
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"Не найден базовый файл модели: {base_model_path}")
            
        print(f"[ACE-Step Baker] Чтение базового файла: {base_model_path}")
        merged_sd = load_file(base_model_path)
        
        registry = getattr(dit_handler, "_lora_service", None).registry if hasattr(dit_handler, "_lora_service") else {}
        
        lora_prefix = "base_model.model."
        base_prefix = "decoder."

        for adapter_name, strength in active_adapters.items():
            if strength == 0.0:
                continue
                
            lora_path = registry.get(adapter_name, {}).get("path", "")
            if not lora_path or not os.path.exists(lora_path):
                print(f"[ACE-Step Baker] ❌ Предупреждение: не найден путь для адаптера {adapter_name}")
                continue
                
            print(f"[ACE-Step Baker] 💉 Запекание LoRA: {adapter_name} (Strength: {strength})")
            lora_sd = self.load_lora_sd(lora_path)
            
            if lora_sd is None:
                print(f"[ACE-Step Baker] ❌ Не удалось загрузить веса для {adapter_name}")
                continue
            
            peft_config = dit_handler.model.decoder.peft_config.get(adapter_name)
            if peft_config:
                lora_r = peft_config.r
                lora_alpha = peft_config.lora_alpha
            else:
                lora_r, lora_alpha = 64, 64 
                
            scale = (lora_alpha / lora_r) * strength
            
            merged_count = 0
            for lora_key, lora_tensor in lora_sd.items():
                if "lora_A" in lora_key:
                    # LoRA key: base_model.model.layers.0...
                    # Target key: decoder.layers.0...
                    base_key = lora_key.replace(lora_prefix, base_prefix).replace(".lora_A.", ".")
                    lora_b_key = lora_key.replace("lora_A", "lora_B")
                    
                    if base_key in merged_sd and lora_b_key in lora_sd:
                        W = merged_sd[base_key]
                        A = lora_tensor
                        B = lora_sd[lora_b_key]
                        
                        orig_dtype = W.dtype
                        
                        W_f32 = W.to(torch.float32)
                        A_f32 = A.to(torch.float32)
                        B_f32 = B.to(torch.float32)
                        
                        if W.ndim == 2 and A.ndim == 2 and B.ndim == 2:
                            delta_W = B_f32 @ A_f32
                            W_f32 += delta_W * scale
                            merged_sd[base_key] = W_f32.to(orig_dtype)
                            merged_count += 1
                            
                elif "lora_B" in lora_key:
                    continue
                    
                else:
                    # Bias и другие параметры
                    base_key = lora_key.replace(lora_prefix, base_prefix)
                        
                    if base_key in merged_sd:
                        W = merged_sd[base_key]
                        orig_dtype = W.dtype
                        
                        W_f32 = W.to(torch.float32)
                        L_f32 = lora_tensor.to(torch.float32)
                        
                        W_f32 = W_f32 + (L_f32 - W_f32) * strength
                        merged_sd[base_key] = W_f32.to(orig_dtype)
                        merged_count += 1

            print(f"[ACE-Step Baker] Успешно слито матриц: {merged_count}")

        print(f"[ACE-Step Baker] 💾 Сохранение итоговой модели в {save_path} ...")
        save_file(merged_sd, save_path)
        print(f"[ACE-Step Baker] ✨ Запекание завершено! Итоговый размер: {os.path.getsize(save_path) / (1024**3):.2f} ГБ")
        
        return (model, save_path)

# ============================================================================
# 3. Настройка параметров LLM
# ============================================================================
class AceStepLMConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lm_temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lm_cfg_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "lm_top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "lm_top_k": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "audio_cover_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Сила аудио-кодов / Cover strength"}),
                "lm_negative_prompt": ("STRING", {"default": "NO USER INPUT", "multiline": True}),
                "use_cot_metas": ("BOOLEAN", {"default": True, "tooltip": "Генерировать недостающие метаданные (BPM, тональность) через LLM"}),
                "use_cot_caption": ("BOOLEAN", {"default": True, "tooltip": "Разрешить LLM переписывать caption во время генерации (если не запрещено в самом генераторе)"}),
                "use_cot_language": ("BOOLEAN", {"default": True, "tooltip": "Определять язык через LLM"}),
                "allow_lm_batch": ("BOOLEAN", {"default": True, "tooltip": "Разрешить параллельную генерацию в LLM"}),
            }
        }

    RETURN_TYPES = ("ACESTEP_LM_CONFIG",)
    RETURN_NAMES = ("lm_config",)
    FUNCTION = "create_config"
    CATEGORY = "ACE-Step"

    def create_config(self, lm_temperature, lm_cfg_scale, lm_top_p, lm_top_k, audio_cover_strength, 
                      lm_negative_prompt, use_cot_metas, use_cot_caption, use_cot_language, allow_lm_batch):
        return ({
            "lm_temperature": lm_temperature,
            "lm_cfg_scale": lm_cfg_scale,
            "lm_top_p": lm_top_p,
            "lm_top_k": lm_top_k,
            "audio_cover_strength": audio_cover_strength,
            "lm_negative_prompt": lm_negative_prompt,
            "use_cot_metas": use_cot_metas,
            "use_cot_caption": use_cot_caption,
            "use_cot_language": use_cot_language,
            "allow_lm_batch": allow_lm_batch
        },)

# ============================================================================
# 4. Улучшение промптов через LLM
# ============================================================================
class AceStepPromptEnhancer:
    """
    Прогоняет базовые caption и lyrics через LLM (format_sample).
    Позволяет принудительно использовать оригинальный текст или задать метаданные вручную.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ACESTEP_MODEL",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "caption": ("STRING", {"multiline": True, "default": "pop song"}),
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "keep_orig_caption": ("BOOLEAN", {"default": False, "tooltip": "Отдать на выход оригинальный Caption, игнорируя результат LLM"}),
                "keep_orig_lyrics": ("BOOLEAN", {"default": False, "tooltip": "Отдать на выход оригинальные Lyrics, игнорируя результат LLM"}),
            },
            "optional": {
                "lm_config": ("ACESTEP_LM_CONFIG",),
                "bpm_override": ("INT", {"default": 0, "min": 0, "max": 300, "tooltip": "0 = использовать сгенерированный LLM"}),
                "keyscale_override": ("STRING", {"default": "", "tooltip": "Пусто = использовать сгенерированный LLM"}),
                "time_signature_override": ("STRING", {"default": "", "tooltip": "Пусто = использовать сгенерированный LLM"}),
                "language_override": (["none", "unknown", "en", "zh", "ja", "ru", "es", "fr", "de"], {"default": "none"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_caption", "enhanced_lyrics", "bpm", "key_scale", "time_signature", "vocal_language")
    FUNCTION = "enhance"
    CATEGORY = "ACE-Step"

    def enhance(self, model, seed, caption, lyrics, keep_orig_caption, keep_orig_lyrics,
                lm_config=None, bpm_override=0, keyscale_override="", 
                time_signature_override="", language_override="none"):
        
        llm_handler = model.get("llm_handler")
        
        # Если LLM нет, просто возвращаем оригиналы или оверрайды
        if not llm_handler or not getattr(llm_handler, "llm_initialized", False):
            print("[ACE-Step Enhancer] Warning: LLM не инициализирована. Возвращаем базовые данные.")
            return (
                caption, 
                lyrics, 
                bpm_override, 
                keyscale_override.strip(), 
                time_signature_override.strip(), 
                language_override if language_override != "none" else "unknown"
            )

        temp = 0.85
        top_k = None
        top_p = None

        if lm_config:
            temp = lm_config.get("lm_temperature", 0.85)
            tk = lm_config.get("lm_top_k", 0)
            tp = lm_config.get("lm_top_p", 0.9)
            top_k = tk if tk > 0 else None
            top_p = tp if tp < 1.0 else None

        print("[ACE-Step Enhancer] Обработка текста через LLM...")
        result = format_sample(
            llm_handler=llm_handler,
            caption=caption,
            lyrics=lyrics,
            user_metadata=None,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            use_constrained_decoding=True,
        )

        if not result.success:
            print(f"[ACE-Step Enhancer] Ошибка при улучшении промпта: {result.error or result.status_message}")
            return (caption, lyrics, bpm_override, keyscale_override, time_signature_override, language_override)

        # Выбираем, что отдать на выход: сгенерированное LLM или переопределенное юзером
        final_caption = caption if keep_orig_caption else (result.caption or caption)
        final_lyrics = lyrics if keep_orig_lyrics else (result.lyrics or lyrics)
        final_bpm = bpm_override if bpm_override > 0 else (result.bpm or 0)
        final_key = keyscale_override.strip() if keyscale_override.strip() else (result.keyscale or "")
        final_ts = time_signature_override.strip() if time_signature_override.strip() else (result.timesignature or "")
        final_lang = language_override if language_override != "none" else (result.language or "unknown")

        return (final_caption, final_lyrics, final_bpm, final_key, final_ts, final_lang)

# ============================================================================
# 5. Генератор Музыки
# ============================================================================
class AceStepMusicGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ACESTEP_MODEL",),
                "task_type": (["text2music", "cover", "repaint"], {"default": "text2music"}),
                "caption": ("STRING", {"multiline": True, "default": "piano solo"}),
                "lyrics": ("STRING", {"multiline": True, "default": "[Instrumental]"}),
                "duration": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 600.0}),
                "inference_steps": ("INT", {"default": 8}),
                "guidance_scale": ("FLOAT", {"default": 7.0}),
                "shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 5.0, "step": 0.1, "tooltip": "Смещение таймстепов (Timestep shift factor)"}),
                "thinking": ("BOOLEAN", {"default": True, "tooltip": "Генерация аудио-кодов через LLM"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "unload_unused_loras": ("BOOLEAN", {"default": True}),
                "merge_loras": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "source_audio": ("AUDIO",),
                "vocal_language": (["unknown", "en", "zh", "ja", "ru"], {"default": "unknown"}),
                "bpm": ("INT", {"default": 0}),
                "key_scale": ("STRING", {"default": ""}),
                "time_signature": ("STRING", {"default": ""}),
                "lm_config": ("ACESTEP_LM_CONFIG",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "ACE-Step"

    def _save_comfy_audio_to_temp(self, comfy_audio) -> str:
        if comfy_audio is None: return None
        waveform = comfy_audio["waveform"].squeeze(0)
        sample_rate = comfy_audio["sample_rate"]
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        torchaudio.save(temp_path, waveform, sample_rate)
        return temp_path

    def _sync_loras(self, dit_handler, requested_adapters, unload_unused):
        """
        Ручное управление слоями PEFT и множителями LyCORIS (LoKr).
        """
        if not dit_handler.lora_loaded:
            return

        loaded_adapters = list(dit_handler._active_loras.keys())
        decoder = getattr(dit_handler.model, "decoder", None)
        combo_name = "comfy_mixed_lora"
        is_lokr = getattr(dit_handler, "_adapter_type", None) == "lokr"
        lyco_net = getattr(decoder, "_lycoris_net", None) if is_lokr and decoder else None
        
        # Если нет запрашиваемых адаптеров (байпасс всех нод)
        if not requested_adapters:
            if unload_unused:
                print("[ACE-Step] Запрошенных LoRA нет. Полная выгрузка...")
                dit_handler.unload_lora()
            else:
                print("[ACE-Step] Запрошенных LoRA нет. Отключаем слои...")
                if not is_lokr and decoder and hasattr(decoder, "disable_adapter_layers"):
                    decoder.disable_adapter_layers()
                
                # Для LoKr сбрасываем веса в 0.0
                if is_lokr and lyco_net is not None:
                    if hasattr(lyco_net, "set_multiplier"):
                        lyco_net.set_multiplier(0.0)
                    else:
                        lyco_net.multiplier = 0.0
                        for m in lyco_net.modules():
                            if hasattr(m, "multiplier"): m.multiplier = 0.0
                    
                    for name in loaded_adapters:
                        dit_handler._active_loras[name] = 0.0
                        try: dit_handler.set_lora_scale(name, 0.0)
                        except: pass
                        
                dit_handler.use_lora = False
            return

        active_names =[]
        active_weights =[]
        
        for name in loaded_adapters:
            # Игнорируем наш временный адаптер для смешивания при переборе
            if name == combo_name:
                continue

            if name in requested_adapters:
                weight = float(requested_adapters[name])
                # Записываем вес для внутреннего стейта ACE-Step
                dit_handler._active_loras[name] = weight
                
                if weight > 0.0:
                    active_names.append(name)
                    active_weights.append(weight)
            else:
                if unload_unused:
                    print(f"[ACE-Step] Выгрузка неиспользуемой LoRA: {name}")
                    try:
                        dit_handler.remove_lora(name)
                    except Exception:
                        print(f"[ACE-Step] Сброс всех LoRA из-за ошибки выгрузки.")
                        dit_handler.unload_lora()
                        return
                else:
                    dit_handler._active_loras[name] = 0.0

        # Активация адаптеров
        if decoder is not None:
            if active_names:
                if is_lokr and lyco_net is not None:
                    adapter_name = active_names[0]
                    scale = active_weights[0]
                    
                    try:
                        dit_handler.set_lora_scale(adapter_name, scale)
                    except Exception:
                        pass
                        
                    # --- ПРИНУДИТЕЛЬНАЯ ЗАПИСЬ СИЛЫ (MULTIPLIER) В LYCORIS ---
                    if hasattr(lyco_net, "set_multiplier"):
                        lyco_net.set_multiplier(scale)
                    else:
                        lyco_net.multiplier = scale
                        # На всякий случай проталкиваем во все дочерние модули
                        for m in lyco_net.modules():
                            if hasattr(m, "multiplier"):
                                m.multiplier = scale
                                
                    print(f"[ACE-Step] Вес LoKr '{adapter_name}' установлен на {scale}")
                else:
                    # Для обычных PEFT адаптеров
                    if hasattr(decoder, "enable_adapter_layers"):
                        decoder.enable_adapter_layers()
                    
                    # Если адаптеров несколько, смешиваем их в один виртуальный
                    if len(active_names) > 1 and hasattr(decoder, "add_weighted_adapter"):
                        try:
                            if combo_name in decoder.peft_config:
                                decoder.delete_adapter(combo_name)
                            
                            # Линейное объединение весов
                            decoder.add_weighted_adapter(
                                adapters=active_names, 
                                weights=active_weights, 
                                adapter_name=combo_name, 
                                combination_type="linear"
                            )
                            decoder.set_adapter(combo_name)
                            print(f"[ACE-Step] Смешаны мульти-LoRA: {active_names} с весами {active_weights}")
                        except Exception as e:
                            print(f"[ACE-Step] Не удалось смешать адаптеры: {e}. Включаем только первый: {active_names[0]}")
                            try: decoder.set_adapter(active_names[0])
                            except Exception: pass
                    else:
                        # Если адаптер один, просто активируем его
                        adapter_name = active_names[0]
                        try:
                            decoder.set_adapter(active_names)
                        except TypeError:
                            decoder.set_adapter(adapter_name) # Fallback для старых PEFT
                        except ValueError as e:
                            print(f"[ACE-Step] Ошибка активации PEFT-адаптера: {e}")
                        
                        try:
                            dit_handler.set_lora_scale(adapter_name, active_weights[0])
                        except Exception as e:
                            print(f"[ACE-Step] Ошибка применения веса LoRA: {e}")
                        
                dit_handler.use_lora = True
                dit_handler.lora_scale = active_weights[0] if active_weights else 1.0
            else:
                # Если запрошенных активных адаптеров нет (все веса 0.0)
                if not is_lokr and hasattr(decoder, "disable_adapter_layers"):
                    decoder.disable_adapter_layers()
                    # ФИКС "ПРИЗРАЧНОЙ ЛОРЫ": Принудительно сбрасываем масштаб (scale) всех загруженных PEFT,
                    # так как disable_adapter_layers иногда пропускает остаточные веса.
                    for name in loaded_adapters:
                        try: dit_handler.set_lora_scale(name, 0.0)
                        except: pass
                    
                # Отключаем LoKr, если запрашиваемый вес был 0.0
                if is_lokr and lyco_net is not None:
                    if hasattr(lyco_net, "set_multiplier"):
                        lyco_net.set_multiplier(0.0)
                    else:
                        lyco_net.multiplier = 0.0
                        for m in lyco_net.modules():
                            if hasattr(m, "multiplier"): m.multiplier = 0.0
                            
                dit_handler.use_lora = False

    def generate(self, model, task_type, caption, lyrics, duration, inference_steps, 
                 guidance_scale, shift, thinking, seed, unload_unused_loras, merge_loras, 
                 reference_audio=None, source_audio=None, vocal_language="unknown", 
                 bpm=0, key_scale="", time_signature="", lm_config=None):
        
        dit_handler = model["dit_handler"]
        llm_handler = model["llm_handler"]
        active_adapters = model.get("active_adapters", {})

        self._sync_loras(dit_handler, active_adapters, unload_unused_loras)

        is_merged = False
        if merge_loras:
            # =================================================================
            # ОПТИМИЗАЦИЯ VRAM: Слияние LoRA с базовой моделью "на лету"
            # Это убирает накладные расходы на VRAM во время forward pass
            # =================================================================
            decoder = getattr(dit_handler.model, "decoder", None)
            if decoder is not None and hasattr(decoder, "merge_adapter") and getattr(dit_handler, "use_lora", False):
                try:
                    print("[ACE-Step] Оптимизация VRAM: Временное слияние (merge) LoRA...")
                    decoder.merge_adapter()
                    is_merged = True
                except Exception as e:
                    print(f"[ACE-Step] Предупреждение: не удалось выполнить merge_adapter: {e}")
            # =================================================================

        if thinking and llm_handler is None:
            thinking = False

        ref_path = self._save_comfy_audio_to_temp(reference_audio)
        src_path = self._save_comfy_audio_to_temp(source_audio)

        # Вытаскиваем параметры LLM (включая параметры CoT)
        lm_temp = 0.85
        lm_cfg = 2.0
        lm_tk = 0
        lm_tp = 0.9
        cover_str = 1.0
        lm_neg_prompt = "NO USER INPUT"
        use_cot_metas = True
        use_cot_caption = True
        use_cot_language = True
        allow_lm_batch = True

        if lm_config:
            lm_temp = lm_config.get("lm_temperature", 0.85)
            lm_cfg = lm_config.get("lm_cfg_scale", 2.0)
            lm_tk = lm_config.get("lm_top_k", 0)
            lm_tp = lm_config.get("lm_top_p", 0.9)
            cover_str = lm_config.get("audio_cover_strength", 1.0)
            lm_neg_prompt = lm_config.get("lm_negative_prompt", "NO USER INPUT")
            use_cot_metas = lm_config.get("use_cot_metas", True)
            use_cot_caption = lm_config.get("use_cot_caption", True)
            use_cot_language = lm_config.get("use_cot_language", True)
            allow_lm_batch = lm_config.get("allow_lm_batch", True)

        params = GenerationParams(
            task_type=task_type, caption=caption, lyrics=lyrics,
            bpm=bpm if bpm > 0 else None, keyscale=key_scale, timesignature=time_signature,
            duration=duration, vocal_language=vocal_language,
            inference_steps=inference_steps, guidance_scale=guidance_scale,
            shift=shift, seed=seed,
            thinking=thinking, reference_audio=ref_path,
            src_audio=src_path if task_type != "text2music" else None,
            use_cot_metas=use_cot_metas, 
            use_cot_caption=use_cot_caption, 
            use_cot_language=use_cot_language,
            audio_cover_strength=cover_str,
            lm_temperature=lm_temp, lm_cfg_scale=lm_cfg, lm_top_k=lm_tk, 
            lm_top_p=lm_tp, lm_negative_prompt=lm_neg_prompt
        )

        config = GenerationConfig(
            batch_size=1, 
            use_random_seed=(seed == -1), 
            audio_format="wav",
            allow_lm_batch=allow_lm_batch
        )

        # ==========================================
        # ПРОГРЕСС БАР COMFYUI
        # ==========================================
        pbar = comfy.utils.ProgressBar(100)
        last_percent = 0

        def progress_callback(value, desc=None, *args, **kwargs):
            nonlocal last_percent
            if isinstance(value, str):
                # print(f"[ACE-Step] {value}")
                return
                
            if isinstance(value, (int, float)):
                current_percent = min(100, max(0, int(value * 100)))
                if current_percent > last_percent:
                    pbar.update(current_percent - last_percent)
                    last_percent = current_percent
                    
            if desc:
                pass # print(f"[ACE-Step Progress] {desc} ({last_percent}%)")

        print("[ACE-Step] Начинаем генерацию...")
        try:
            result = generate_music(
                dit_handler=dit_handler, 
                llm_handler=llm_handler, 
                params=params, 
                config=config, 
                save_dir=None,
                progress=progress_callback
            )
            
            if not result.success:
                raise RuntimeError(f"Generation Failed: {result.error}")

            if last_percent < 100:
                pbar.update(100 - last_percent)

            output_audio_tensor = result.audios[0]["tensor"]
            output_sample_rate = result.audios[0]["sample_rate"]

            return ({"waveform": output_audio_tensor.unsqueeze(0), "sample_rate": output_sample_rate},)

        finally:

            # =================================================================
            # ВОЗВРАТ СОСТОЯНИЯ: Отменяем слияние после генерации
            # =================================================================
            if is_merged and hasattr(decoder, "unmerge_adapter"):
                try:
                    print("[ACE-Step] Отмена слияния (unmerge) LoRA...")
                    decoder.unmerge_adapter()
                except Exception as e:
                    print(f"[ACE-Step] Ошибка при unmerge_adapter: {e}")
            # =================================================================

            if ref_path and os.path.exists(ref_path): os.remove(ref_path)
            if src_path and os.path.exists(src_path): os.remove(src_path)

NODE_CLASS_MAPPINGS = {
    "AceStepModelLoader": AceStepModelLoader,
    "AceStepLoraLoader": AceStepLoraLoader,
    "AceStepLoraBaker": AceStepLoraBaker,
    "AceStepLMConfig": AceStepLMConfig,
    "AceStepPromptEnhancer": AceStepPromptEnhancer,
    "AceStepMusicGenerator": AceStepMusicGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepModelLoader": "ACE-Step Model Loader 🎵",
    "AceStepLoraLoader": "ACE-Step LoRA Loader 💊",
    "AceStepLoraBaker": "ACE-Step LoRA Baker 🍳",
    "AceStepLMConfig": "ACE-Step LM Config ⚙️",
    "AceStepPromptEnhancer": "ACE-Step Prompt Enhancer ✍️",
    "AceStepMusicGenerator": "ACE-Step Music Generator 🎵"
}