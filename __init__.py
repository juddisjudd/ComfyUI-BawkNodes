"""
ComfyUI Bawk Nodes - Node Registration (Without FluxPromptSaver)
File: __init__.py
"""

from .nodes import (
    DiffusionModelLoader, 
    FluxImageSaver, 
    FluxWildcardEncode,
    BawkSampler
)

NODE_CLASS_MAPPINGS = {
    "DiffusionModelLoader": DiffusionModelLoader,
    "FluxImageSaver": FluxImageSaver,
    "FluxWildcardEncode": FluxWildcardEncode,
    "BawkSampler": BawkSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionModelLoader": "🚀 Diffusion Model Loader (Advanced)",
    "FluxImageSaver": "💾 FLUX Image Saver",
    "FluxWildcardEncode": "🎲 FLUX Wildcard Encoder",
    "BawkSampler": "🐓 Bawk Sampler (All-in-One)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

__version__ = "2.0.2"
__author__ = "Bawk Nodes"
__description__ = "A complete collection of FLUX-optimized ComfyUI nodes for enhanced workflows"

# Colored startup messages
print(f"\033[1m\033[92m🐓 ComfyUI Bawk Nodes v{__version__} loaded successfully!\033[0m")
print(f"\033[1m\033[93m   🎉 Major Update - Complete FLUX Workflow Suite!\033[0m")
print(f"\033[1m\033[96m   Current nodes:\033[0m")
print(f"\033[94m   • 🚀  Diffusion Model Loader (Advanced) - FLUX-optimized model loading\033[0m")
print(f"\033[95m   • 🎲  FLUX Wildcard Encoder - Text encoding with wildcard support and 6 LoRA slots\033[0m")
print(f"\033[92m   • 💾  FLUX Image Saver - Organized image saving with metadata and prompt files\033[0m")
print(f"\033[91m   • 🐓  Bawk Sampler (All-in-One) - Combined latent optimizer and sampler\033[0m")
print(f"\033[3m\033[96m   •  📦  Modular architecture for easy maintenance and debugging\033[0m")