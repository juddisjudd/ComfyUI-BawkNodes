"""
ComfyUI Bawk Nodes - Node Registration (Without FluxPromptSaver)
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

NODE_CLASS_MAPPINGS = {
    "DiffusionModelLoader": DiffusionModelLoader,
    "FluxImageSaver": FluxImageSaver,
    "FluxWildcardEncode": FluxWildcardEncode,
    "BawkSampler": BawkSampler,
    "BawkBatchProcessor": BawkBatchProcessor,
    "BawkControlNet": BawkControlNet,
    "BawkImageLoader": BawkImageLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionModelLoader": "🚀 Bawk Model Loader",
    "FluxImageSaver": "💾 Bawk Image Saver",
    "FluxWildcardEncode": "🎲 Bawk Wildcard Encoder",
    "BawkSampler": "🐓 Bawk Sampler",
    "BawkBatchProcessor": "📁 Bawk Batch Processor",
    "BawkControlNet": "🎛️ Bawk ControlNet",
    "BawkImageLoader": "📸 Bawk Image Loader",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

__version__ = "2.0.5"
__author__ = "Bawk Nodes"
__description__ = "A complete collection of FLUX-optimized ComfyUI nodes for enhanced workflows"

# Colored startup messages
print(f"\033[1m\033[92m🐓 ComfyUI Bawk Nodes v{__version__} loaded successfully!\033[0m")
print(f"\033[1m\033[93m   🎉 Major Update - Complete FLUX Workflow Suite!\033[0m")
print(f"\033[1m\033[96m   Current nodes:\033[0m")
print(f"\033[94m   • 🚀  Bawk Model Loader - FLUX-optimized model loading with advanced caching\033[0m")
print(f"\033[95m   • 🎲  Bawk Wildcard Encoder - Text encoding, wildcards & 6 LoRA slots\033[0m")
print(f"\033[92m   • 💾  Bawk Image Saver - Organized saving with metadata, prompts & Discord webhooks\033[0m")
print(f"\033[91m   • 🐓  Bawk Sampler - All-in-one text2img and img2img sampler with VAE decoding\033[0m")
print(f"\033[93m   • 📁  Bawk Batch Processor - Process multiple prompts from CSV/JSON files\033[0m")
print(f"\033[96m   • 🎛️  Bawk ControlNet - FLUX-optimized ControlNet preprocessing\033[0m")
print(f"\033[97m   • 📸  Bawk Image Loader - Enhanced image loading with preprocessing\033[0m")
print(f"\033[3m\033[96m   • 📦  Modular architecture for easy maintenance and debugging\033[0m")