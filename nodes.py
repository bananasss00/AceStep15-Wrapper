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
        try:
            if llm_handler:
                llm_handler.unload()
        except: pass
        try:
            if dit_handler:
                dit_handler.model = None
                dit_handler.vae = None
                dit_handler.text_encoder = None
                dit_handler._base_decoder = None
        except: pass
    
    GLOBAL_ACESTEP_HANDLERS.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    print("[ACE-Step] Память успешно освобождена.")

# Обезьяний патч (Monkey Patch): подключаемся к функции выгрузки моделей ComfyUI
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
            print(f"[ACE-Step] LoRA {adapter_name}: OK (weight={strength})")
            new_active = model["active_adapters"].copy()
            new_active[adapter_name] = strength
            
            new_model = model.copy()
            new_model["active_adapters"] = new_active
            return (new_model,)

        return (model,)

# ============================================================================
# 3. Настройка параметров LLM
# ============================================================================
class AceStepLMConfig:
    """
    Узел для настройки параметров языковой модели и Chain-of-Thought (CoT)
    """
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

    def enhance(self, model, caption, lyrics, keep_orig_caption, keep_orig_lyrics,
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
                "thinking": ("BOOLEAN", {"default": True, "tooltip": "Генерация аудио-кодов через LLM"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "unload_unused_loras": ("BOOLEAN", {"default": True}),
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
        if not dit_handler.lora_loaded:
            return

        loaded_adapters = list(dit_handler._active_loras.keys())

        if not requested_adapters:
            if unload_unused:
                print("[ACE-Step] Все LoRA отключены. Полная выгрузка PEFT (unload_lora)...")
                dit_handler.unload_lora()
            else:
                print("[ACE-Step] Все LoRA отключены. Устанавливаем вес 0.0...")
                for name in loaded_adapters:
                    dit_handler.set_lora_scale(name, 0.0)
                dit_handler.set_use_lora(False)
            return

        # Если есть запрашиваемые адаптеры
        has_active = False
        for name in loaded_adapters:
            if name in requested_adapters:
                weight = requested_adapters[name]
                dit_handler.set_lora_scale(name, weight)
                has_active = True
            else:
                if unload_unused:
                    print(f"[ACE-Step] Выгрузка неиспользуемой LoRA: {name}")
                    try:
                        dit_handler.remove_lora(name)
                    except Exception as e:
                        print(f"[ACE-Step] Предупреждение PEFT при удалении: {e}. Выполняю полный сброс.")
                        dit_handler.unload_lora()
                        return # Прерываем цикл, так как все удалено.
                else:
                    if dit_handler._active_loras.get(name, 0.0) != 0.0:
                        dit_handler.set_lora_scale(name, 0.0)

        if dit_handler.lora_loaded:
            dit_handler.set_use_lora(has_active)

    def generate(self, model, task_type, caption, lyrics, duration, inference_steps, 
                 guidance_scale, thinking, seed, unload_unused_loras, 
                 reference_audio=None, source_audio=None, vocal_language="unknown", 
                 bpm=0, key_scale="", time_signature="", lm_config=None):
        
        dit_handler = model["dit_handler"]
        llm_handler = model["llm_handler"]
        active_adapters = model.get("active_adapters", {})

        self._sync_loras(dit_handler, active_adapters, unload_unused_loras)

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
            inference_steps=inference_steps, guidance_scale=guidance_scale, seed=seed,
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
            if ref_path and os.path.exists(ref_path): os.remove(ref_path)
            if src_path and os.path.exists(src_path): os.remove(src_path)

NODE_CLASS_MAPPINGS = {
    "AceStepModelLoader": AceStepModelLoader,
    "AceStepLoraLoader": AceStepLoraLoader,
    "AceStepLMConfig": AceStepLMConfig,
    "AceStepPromptEnhancer": AceStepPromptEnhancer,
    "AceStepMusicGenerator": AceStepMusicGenerator
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepModelLoader": "ACE-Step Model Loader 🎵",
    "AceStepLoraLoader": "ACE-Step LoRA Loader 💊",
    "AceStepLMConfig": "ACE-Step LM Config ⚙️",
    "AceStepPromptEnhancer": "ACE-Step Prompt Enhancer ✍️",
    "AceStepMusicGenerator": "ACE-Step Music Generator 🎵"
}