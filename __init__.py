"""
ComfyUI Bawk Nodes - Node Registration
File: __init__.py
"""

from .nodes import (
    DiffusionModelLoader,
    FluxImageSaver,
    FluxWildcardEncode,
    BawkSampler,
    BawkBatchProcessor,
    BawkControlNet,
    BawkImageLoader
)
from .nodes.flux_dynamic_lora import FluxDynamicLoraStack, FluxLoraStackApply
from .nodes.zimage_loader import ZImageModelLoader

NODE_CLASS_MAPPINGS = {
    "DiffusionModelLoader": DiffusionModelLoader,
    "FluxImageSaver": FluxImageSaver,
    "FluxWildcardEncode": FluxWildcardEncode,
    "BawkSampler": BawkSampler,
    "BawkBatchProcessor": BawkBatchProcessor,
    "BawkControlNet": BawkControlNet,
    "BawkImageLoader": BawkImageLoader,
    "FluxDynamicLoraStack": FluxDynamicLoraStack,
    "FluxLoraStackApply": FluxLoraStackApply,
    "ZImageModelLoader": ZImageModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionModelLoader": "🚀 Bawk Model Loader",
    "FluxImageSaver": "💾 Bawk Image Saver",
    "FluxWildcardEncode": "🎲 Bawk Wildcard Encoder",
    "BawkSampler": "🐓 Bawk Sampler",
    "BawkBatchProcessor": "📁 Bawk Batch Processor",
    "BawkControlNet": "🎛️ Bawk ControlNet",
    "BawkImageLoader": "📸 Bawk Image Loader",
    "FluxDynamicLoraStack": "🔗 Bawk LoRA Stack",
    "FluxLoraStackApply": "⚡ Bawk LoRA Apply",
    "ZImageModelLoader": "⚡ Z-Image Loader",
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

__version__ = "2.3.0"
__author__ = "Bawk Nodes"
__description__ = "FLUX & Z-Image optimized ComfyUI nodes for enhanced workflows"

# Colored startup messages
print(f"\033[1m\033[92m🐓 ComfyUI Bawk Nodes v{__version__} loaded successfully!\033[0m")
print(f"\033[1m\033[93m   ⭐ FLUX.1 & Z-Image Optimized - Quality & Performance!\033[0m")
print(f"\033[1m\033[96m   Current nodes:\033[0m")
print(f"\033[94m   • 🚀  Bawk Model Loader - FLUX-optimized model loading\033[0m")
print(f"\033[95m   • 🎲  Bawk Wildcard Encoder - Text encoding with 6 LoRA slots\033[0m")
print(f"\033[92m   • 💾  Bawk Image Saver - Organized saving with metadata & Discord webhooks\033[0m")
print(f"\033[91m   • 🐓  Bawk Sampler - All-in-one sampler for FLUX/Z-Image\033[0m")
print(f"\033[93m   • 📁  Bawk Batch Processor - Process multiple prompts from CSV/JSON\033[0m")
print(f"\033[96m   • 🎛️  Bawk ControlNet - FLUX-optimized ControlNet preprocessing\033[0m")
print(f"\033[97m   • 📸  Bawk Image Loader - Enhanced image loading with preprocessing\033[0m")
print(f"\033[33m   • 🔗  Bawk LoRA Stack - Dynamic LoRA management (7+ LoRAs)\033[0m")
print(f"\033[33m   • ⚡  Bawk LoRA Apply - Apply LoRA stacks to model/CLIP\033[0m")
print(f"\033[36m   • ⚡  Z-Image Loader - Z-Image-Turbo model (8 steps, no CFG)\033[0m")