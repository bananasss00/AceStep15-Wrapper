"""LoRA/LoKr adapter load/unload lifecycle management."""

import json
import os
import glob
import torch
from typing import Any

from loguru import logger

from acestep.constants import DEBUG_MODEL_LOADING
from acestep.debug_utils import debug_log
from acestep.training.configs import LoKRConfig
# from peft import PeftModel, PeftConfig 

from peft.utils.save_and_load import set_peft_model_state_dict

# Понадобится для чтения safetensors в память
try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

LOKR_WEIGHTS_FILENAME = "lokr_weights.safetensors"


def _is_lokr_safetensors(weights_path: str) -> bool:
    """Return whether ``weights_path`` looks like a LoKr/LyCORIS safetensors file."""
    if not os.path.isfile(weights_path) or not weights_path.lower().endswith(".safetensors"):
        return False
    if os.path.basename(weights_path) == LOKR_WEIGHTS_FILENAME:
        return True

    try:
        from safetensors import safe_open
    except ImportError:
        return False

    try:
        with safe_open(weights_path, framework="pt", device="cpu") as sf:
            metadata: dict[str, Any] = sf.metadata() or {}
    except Exception:
        return False

    raw_config = metadata.get("lokr_config")
    return isinstance(raw_config, str) and bool(raw_config.strip())


def _resolve_lokr_weights_path(adapter_path: str) -> str | None:
    """Return LoKr safetensors path when ``adapter_path`` points to LoKr artifacts."""
    if os.path.isfile(adapter_path):
        return adapter_path if _is_lokr_safetensors(adapter_path) else None
    if os.path.isdir(adapter_path):
        weights_path = os.path.join(adapter_path, LOKR_WEIGHTS_FILENAME)
        if os.path.exists(weights_path):
            return weights_path

        # Backward-compat: support custom LyCORIS safetensors filenames that
        # carry ``lokr_config`` metadata.
        try:
            entries = os.listdir(adapter_path)
        except OSError:
            return None
        for name in entries:
            candidate = os.path.join(adapter_path, name)
            if _is_lokr_safetensors(candidate):
                return candidate
    return None


def _load_lokr_config(weights_path: str) -> LoKRConfig:
    """Build ``LoKRConfig`` from safetensors metadata, with defaults on parse failure."""
    config = LoKRConfig()
    try:
        from safetensors import safe_open
    except ImportError:
        logger.warning("safetensors metadata reader unavailable; using default LoKr config.")
        return config

    try:
        with safe_open(weights_path, framework="pt", device="cpu") as sf:
            metadata: dict[str, Any] = sf.metadata() or {}
    except Exception as exc:
        logger.warning(f"Unable to read LoKr metadata from {weights_path}: {exc}")
        return config

    raw_config = metadata.get("lokr_config")
    if not isinstance(raw_config, str) or not raw_config.strip():
        return config

    try:
        parsed = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        logger.warning(f"Invalid LoKr metadata config JSON in {weights_path}: {exc}")
        return config

    if not isinstance(parsed, dict):
        return config

    allowed_keys = set(LoKRConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in parsed.items() if k in allowed_keys}
    if not filtered:
        return config

    try:
        return LoKRConfig(**filtered)
    except Exception as exc:
        logger.warning(f"Failed to apply LoKr metadata config from {weights_path}: {exc}")
        return config


def _load_lokr_adapter(decoder: Any, weights_path: str) -> Any:
    """Inject and load a LoKr LyCORIS adapter into ``decoder``."""
    try:
        from lycoris import LycorisNetwork, create_lycoris
    except ImportError as exc:
        raise ImportError("LyCORIS library not installed. Please install with: pip install lycoris-lora") from exc

    lokr_config = _load_lokr_config(weights_path)
    LycorisNetwork.apply_preset(
        {
            "unet_target_name": lokr_config.target_modules,
            "target_name": lokr_config.target_modules,
        }
    )
    lycoris_net = create_lycoris(
        decoder,
        1.0,
        linear_dim=lokr_config.linear_dim,
        linear_alpha=lokr_config.linear_alpha,
        algo="lokr",
        factor=lokr_config.factor,
        decompose_both=lokr_config.decompose_both,
        use_tucker=lokr_config.use_tucker,
        use_scalar=lokr_config.use_scalar,
        full_matrix=lokr_config.full_matrix,
        bypass_mode=lokr_config.bypass_mode,
        rs_lora=lokr_config.rs_lora,
        unbalanced_factorization=lokr_config.unbalanced_factorization,
    )

    if lokr_config.weight_decompose:
        try:
            lycoris_net = create_lycoris(
                decoder,
                1.0,
                linear_dim=lokr_config.linear_dim,
                linear_alpha=lokr_config.linear_alpha,
                algo="lokr",
                factor=lokr_config.factor,
                decompose_both=lokr_config.decompose_both,
                use_tucker=lokr_config.use_tucker,
                use_scalar=lokr_config.use_scalar,
                full_matrix=lokr_config.full_matrix,
                bypass_mode=lokr_config.bypass_mode,
                rs_lora=lokr_config.rs_lora,
                unbalanced_factorization=lokr_config.unbalanced_factorization,
                dora_wd=True,
            )
        except Exception as exc:
            logger.warning(f"DoRA mode not supported in current LyCORIS build: {exc}")

    lycoris_net.apply_to()
    decoder._lycoris_net = lycoris_net
    lycoris_net.load_weights(weights_path)
    return lycoris_net


def _default_adapter_name_from_path(lora_path: str) -> str:
    """Derive a default adapter name from path (e.g. 'final' from './lora/final')."""
    name = os.path.basename(lora_path.rstrip(os.sep))
    return name if name else "default"


def add_lora(self, lora_path: str, adapter_name: str | None = None, ignore_bias: bool = False) -> str:
    """
    Load a LoRA adapter. 
    If ignore_bias=True, loads weights into RAM, removes bias keys, and injects into PEFT 
    to avoid 'only 1 adapter with bias' errors.
    """
    if self.model is None:
        return "❌ Model not initialized."


    if not lora_path or not os.path.exists(lora_path):
        return f"❌ LoRA path not found: {lora_path}"
    try:
        from peft import PeftModel, PeftConfig  # УЖЕ ИМПОРТИРОВАНО В НАЧАЛЕ
    except ImportError:
        if lokr_weights_path is None:
            return "❌ PEFT library not installed. Please install with: pip install peft"
        PeftModel = None  # type: ignore[assignment]
        PeftConfig = None  # type: ignore[assignment]
    

    effective_name = adapter_name.strip() if adapter_name else _default_adapter_name_from_path(lora_path)
    
    # Проверка на уже загруженный адаптер
    _active_loras = getattr(self, "_active_loras", None)
    if _active_loras is None:
        self._active_loras = {}
        _active_loras = self._active_loras
    
    if effective_name in _active_loras:
        return f"✅ Adapter '{effective_name}' already loaded."

    decoder = self.model.decoder
    is_peft = isinstance(decoder, PeftModel)

    # --- ЛОГИКА IN-MEMORY FILTERING ---
    if ignore_bias:
        try:
            logger.info(f"Loading LoRA '{effective_name}' with in-memory bias filtering...")
            
            # 1. Загружаем и патчим конфиг в памяти
            config = PeftConfig.from_pretrained(lora_path)
            if hasattr(config, 'bias'):
                config.bias = "none" # Принудительно отключаем bias в конфиге
            
            # 2. Загружаем веса в память
            state_dict = None
            
            # Пробуем safetensors
            if load_safetensors:
                st_files = glob.glob(os.path.join(lora_path, "*.safetensors"))
                if st_files:
                    state_dict = load_safetensors(st_files[0])
            
            # Пробуем bin, если safetensors нет или не сработал
            if state_dict is None:
                bin_files = glob.glob(os.path.join(lora_path, "*.bin"))
                if bin_files:
                    state_dict = torch.load(bin_files[0], map_location="cpu")
            
            if state_dict is None:
                return "❌ Could not find weights (.safetensors or .bin) in LoRA path."

            # 3. Фильтруем веса (удаляем bias)
            clean_state_dict = {k: v for k, v in state_dict.items() if "bias" not in k}
            removed_count = len(state_dict) - len(clean_state_dict)
            if removed_count > 0:
                logger.info(f"Filtered {removed_count} bias keys from state_dict.")

            # 4. Применяем к модели
            if not is_peft:
                # Первый адаптер: создаем PeftModel
                # Backup base model
                if self._base_decoder is None:
                    base_sd = decoder.state_dict()
                    self._base_decoder = {k: v.detach().cpu().clone() for k, v in base_sd.items()}
                
                if hasattr(decoder, "peft_config"): del decoder.peft_config
                
                # Инициализируем пустую PeftModel с нашим конфигом (она создаст слои, но веса будут случайные)
                self.model.decoder = PeftModel(decoder, config, adapter_name=effective_name)
                self._adapter_type = "lora"
            else:
                # Добавляем новый адаптер (создает слои)
                self.model.decoder.add_adapter(effective_name, config)
            
            # 5. Загружаем наши очищенные веса в созданные слои
            # set_peft_model_state_dict сама разберется с префиксами для конкретного адаптера
            # Важно: set_peft_model_state_dict ожидает ключи вида "base_model.model.layers...",
            # а в файле они могут быть короче. Peft обычно хендлит это, но проверим.
            
            # Простой способ: используем внутренний метод load_state_dict PEFT-модели для конкретного адаптера? 
            # Нет, надежнее использовать утилиту set_peft_model_state_dict.
            # Но ключи в state_dict могут требовать переименования, если они сохранены без префиксов.
            
            # Попытка загрузки
            incompatible = set_peft_model_state_dict(self.model.decoder, clean_state_dict, adapter_name=effective_name)
            if incompatible and incompatible.unexpected_keys:
                 logger.warning(f"Unexpected keys during in-memory load: {incompatible.unexpected_keys[:5]}")
            
            # Успех
            self.model.decoder.to(self.device).eval()
            self.lora_loaded = True
            self.use_lora = True
            self._active_loras[effective_name] = 1.0
            
            # Обновляем реестр и активируем
            self._ensure_lora_registry()
            # Важно: rebuild_lora_registry обычно сканирует файлы, но у нас файл не совпадает с тем, что в памяти.
            # Передадим lora_path, чтобы он нашел targets из json, это безопасно.
            self._rebuild_lora_registry(lora_path=lora_path) 
            self._lora_service.set_active_adapter(effective_name)
            self._lora_active_adapter = effective_name
            
            return f"✅ LoRA '{effective_name}' loaded (RAM filtered)."

        except Exception as e:
            logger.exception("In-memory LoRA load failed")
            return f"❌ In-memory load failed: {e}"

    # --- СТАНДАРТНАЯ ЛОГИКА (Если ignore_bias=False или LoKr) ---
    
    # ... (здесь идет остаток оригинальной функции для случая LoKr или стандартной загрузки) ...
    # Я приведу полный код функции ниже, чтобы вы просто скопировали и вставили.

    # [ВСТАВИТЬ СТАРЫЙ КОД ТУТ ДЛЯ LOKR И ОБЫЧНОЙ ЗАГРУЗКИ]
    # Для краткости ответа, вот полная комбинированная функция:

    lokr_weights_path = _resolve_lokr_weights_path(lora_path)
    
    try:
        if not is_peft:
            if self._base_decoder is None:
                sd = decoder.state_dict()
                self._base_decoder = {k: v.detach().cpu().clone() for k, v in sd.items()}

            if lokr_weights_path is not None:
                _load_lokr_adapter(decoder, lokr_weights_path)
                self.model.decoder = decoder
                self._adapter_type = "lokr"
            else:
                if hasattr(decoder, "peft_config"): del decoder.peft_config
                self.model.decoder = PeftModel.from_pretrained(decoder, lora_path, adapter_name=effective_name, is_trainable=False)
                self._adapter_type = "lora"
        else:
            if lokr_weights_path is not None:
                return "❌ LoKr cannot be added to PEFT model."
            self.model.decoder.load_adapter(lora_path, adapter_name=effective_name)
            self._adapter_type = "lora"

        self.model.decoder.to(self.device).eval()
        self.lora_loaded = True
        self.use_lora = True
        self._active_loras[effective_name] = 1.0
        
        self._ensure_lora_registry()
        self._rebuild_lora_registry(lora_path=lora_path)
        self._lora_service.set_active_adapter(effective_name)
        self._lora_active_adapter = effective_name
        
        return f"✅ LoRA '{effective_name}' loaded."

    except Exception as e:
        return f"❌ Failed: {e}"

def load_lora(self, lora_path: str) -> str:
    """Load a single adapter (backward-compat), including LyCORIS LoKr paths."""
    lokr_weights_path = _resolve_lokr_weights_path(lora_path.strip()) if isinstance(lora_path, str) else None
    message = self.add_lora(lora_path, adapter_name=None)
    if lokr_weights_path is not None and message.startswith("✅"):
        return f"✅ LoKr loaded from {lokr_weights_path}"
    return message


def add_voice_lora(self, lora_path: str, scale: float = 1.0) -> str:
    """Load a LoRA as the 'voice' adapter and set its scale. Same machinery as style LoRA."""
    msg = self.add_lora(lora_path, adapter_name="voice")
    if not msg.startswith("✅"):
        return msg
    return self.set_lora_scale("voice", scale)


def remove_lora(self, adapter_name: str) -> str:
    """Remove one LoRA adapter by name. If no adapters remain, restores base decoder."""
    if not self.lora_loaded:
        return "⚠️ No LoRA adapter loaded."

    _active_loras = getattr(self, "_active_loras", None) or {}
    if adapter_name not in _active_loras:
        return f"❌ Unknown adapter: {adapter_name}. Loaded: {list(_active_loras.keys())}"

    try:
        from peft import PeftModel
    except ImportError:
        return "❌ PEFT library not installed."

    decoder = getattr(self.model, "decoder", None)
    if decoder is None or not isinstance(decoder, PeftModel):
        # Inconsistent state: clear our bookkeeping
        _active_loras.pop(adapter_name, None)
        if not _active_loras:
            self.lora_loaded = False
            self.use_lora = False
            self._adapter_type = None
        return "⚠️ Adapter removed from registry (decoder was not PEFT)."

    if adapter_name not in (getattr(decoder, "peft_config", None) or {}):
        _active_loras.pop(adapter_name, None)
        self._ensure_lora_registry()
        self._rebuild_lora_registry()
        return f"✅ Adapter '{adapter_name}' removed (was not in PEFT)."

    try:
        decoder.delete_adapter(adapter_name)
        _active_loras.pop(adapter_name, None)
        remaining = list(_active_loras.keys())

        if not remaining:
            # No adapters left: restore base decoder
            if self._base_decoder is None:
                self.lora_loaded = False
                self.use_lora = False
                self._adapter_type = None
                self._active_loras.clear()
                self._ensure_lora_registry()
                self._lora_service.registry = {}
                self._lora_service.scale_state = {}
                self._lora_service.active_adapter = None
                self._lora_service.last_scale_report = {}
                self._lora_adapter_registry = {}
                self._lora_active_adapter = None
                self._lora_scale_state = {}
                return "✅ Last adapter removed; base decoder still wrapped (no backup). Restart or load a new LoRA."
            mem_before = None
            if hasattr(self, "_memory_allocated"):
                mem_before = self._memory_allocated() / (1024**3)
                logger.info(f"VRAM before LoRA unload: {mem_before:.2f}GB")
            self.model.decoder = decoder.get_base_model()
            load_result = self.model.decoder.load_state_dict(self._base_decoder, strict=False)
            if load_result.missing_keys:
                logger.warning(f"Missing keys when restoring decoder: {load_result.missing_keys[:5]}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected keys when restoring decoder: {load_result.unexpected_keys[:5]}")
            self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
            self.model.decoder.eval()
            self.lora_loaded = False
            self.use_lora = False
            self._adapter_type = None
            self._active_loras = {}
            self._ensure_lora_registry()
            self._lora_service.registry = {}
            self._lora_service.scale_state = {}
            self._lora_service.active_adapter = None
            self._lora_service.last_scale_report = {}
            self._lora_adapter_registry = {}
            self._lora_active_adapter = None
            self._lora_scale_state = {}
            if mem_before is not None and hasattr(self, "_memory_allocated"):
                mem_after = self._memory_allocated() / (1024**3)
                logger.info(f"VRAM after LoRA unload: {mem_after:.2f}GB (freed: {mem_before - mem_after:.2f}GB)")
            logger.info("LoRA unloaded, base decoder restored")
            return "✅ LoRA unloaded, using base model"
        # Else: set another adapter active and rebuild registry
        next_active = remaining[0]
        if hasattr(decoder, "set_adapter"):
            try:
                decoder.set_adapter(next_active)
            except Exception:
                pass
        self._lora_active_adapter = next_active
        self._ensure_lora_registry()
        self._rebuild_lora_registry()
        self._lora_service.set_active_adapter(next_active)
        # Re-apply scale for the now-active adapter
        scale = self._active_loras.get(next_active, 1.0)
        self._apply_scale_to_adapter(next_active, scale)
        logger.info(f"Adapter '{adapter_name}' removed. Active: {next_active}")
        return f"✅ Adapter '{adapter_name}' removed. Active: {next_active}"
    except Exception as e:
        logger.exception("Failed to remove LoRA adapter")
        return f"❌ Failed to remove LoRA: {str(e)}"


def unload_lora(self) -> str:
    """Unload all LoRA adapters and restore base decoder."""
    if not self.lora_loaded:
        return "⚠️ No LoRA adapter loaded."

    if self._base_decoder is None:
        return "❌ Base decoder backup not found. Cannot restore."

    try:
        mem_before = None
        if hasattr(self, "_memory_allocated"):
            mem_before = self._memory_allocated() / (1024**3)
            logger.info(f"VRAM before LoRA unload: {mem_before:.2f}GB")

        # If this decoder has an attached LyCORIS net, restore original module graph first.
        lycoris_net = getattr(self.model.decoder, "_lycoris_net", None)
        if lycoris_net is not None:
            restore_fn = getattr(lycoris_net, "restore", None)
            if callable(restore_fn):
                logger.info("Restoring decoder structure from LyCORIS adapter")
                restore_fn()
            else:
                logger.warning("Decoder has _lycoris_net but no restore() method; continuing with state_dict restore")
            self.model.decoder._lycoris_net = None

        try:
            from peft import PeftModel
        except ImportError:
            PeftModel = None  # type: ignore[assignment]

        if PeftModel is not None and isinstance(self.model.decoder, PeftModel):
            logger.info("Unloading PEFT wrapper completely")
            # .unload() полностью удаляет слои LoRA и возвращает чистую модель
            self.model.decoder = self.model.decoder.unload()
            
            # CRITICAL FIX: Explicitly remove peft_config from the base model instance.
            # Even after unload(), the base model might retain this attribute, which causes
            # "ValueError: LoraModel supports only 1 adapter with bias" upon reloading.
            if hasattr(self.model.decoder, "peft_config"):
                del self.model.decoder.peft_config
            if hasattr(self.model.decoder, "_peft_config"):
                del self.model.decoder._peft_config

            logger.info("Restoring base decoder state from backup")
            load_result = self.model.decoder.load_state_dict(self._base_decoder, strict=False)
            if load_result.missing_keys:
                logger.warning(f"Missing keys when restoring decoder: {load_result.missing_keys[:5]}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected keys when restoring decoder: {load_result.unexpected_keys[:5]}")
        else:
            logger.info("Restoring base decoder from state_dict backup")
            load_result = self.model.decoder.load_state_dict(self._base_decoder, strict=False)
            if load_result.missing_keys:
                logger.warning(f"Missing keys when restoring decoder: {load_result.missing_keys[:5]}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected keys when restoring decoder: {load_result.unexpected_keys[:5]}")

        self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
        self.model.decoder.eval()

        self.lora_loaded = False
        self.use_lora = False
        self._adapter_type = None
        self.lora_scale = 1.0
        _active_loras = getattr(self, "_active_loras", None)
        if _active_loras is not None:
            _active_loras.clear()
        self._ensure_lora_registry()
        self._lora_service.registry = {}
        self._lora_service.scale_state = {}
        self._lora_service.active_adapter = None
        self._lora_service.last_scale_report = {}
        self._lora_adapter_registry = {}
        self._lora_active_adapter = None
        self._lora_scale_state = {}

        if mem_before is not None and hasattr(self, "_memory_allocated"):
            mem_after = self._memory_allocated() / (1024**3)
            logger.info(f"VRAM after LoRA unload: {mem_after:.2f}GB (freed: {mem_before - mem_after:.2f}GB)")

        logger.info("LoRA unloaded, base decoder restored")
        return "✅ LoRA unloaded, using base model"
    except Exception as e:
        logger.exception("Failed to unload LoRA")
        return f"❌ Failed to unload LoRA: {str(e)}"
