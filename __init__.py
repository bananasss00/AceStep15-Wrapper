from . import nodes, acestep_trainer_node, smart_lora_merger

NODE_CLASS_MAPPINGS = {
    **nodes.NODE_CLASS_MAPPINGS,
    **acestep_trainer_node.NODE_CLASS_MAPPINGS,
    **smart_lora_merger.NODE_CLASS_MAPPINGS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **nodes.NODE_DISPLAY_NAME_MAPPINGS,
    **acestep_trainer_node.NODE_DISPLAY_NAME_MAPPINGS,
    **smart_lora_merger.NODE_DISPLAY_NAME_MAPPINGS
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]