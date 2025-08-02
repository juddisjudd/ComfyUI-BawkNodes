"""
ComfyUI Bawk Nodes - Node Registration
File: __init__.py
"""

from .nodes import (
    DiffusionModelLoader, 
    FluxImageSaver, 
    FluxPromptSaver, 
    FluxWildcardEncode,
    BawkSampler
)

NODE_CLASS_MAPPINGS = {
    "DiffusionModelLoader": DiffusionModelLoader,
    "FluxImageSaver": FluxImageSaver,
    "FluxPromptSaver": FluxPromptSaver,
    "FluxWildcardEncode": FluxWildcardEncode,
    "BawkSampler": BawkSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionModelLoader": "🚀 Diffusion Model Loader (Advanced)",
    "FluxImageSaver": "💾 FLUX Image Saver",
    "FluxPromptSaver": "📝 FLUX Prompt Saver", 
    "FluxWildcardEncode": "🎲 FLUX Wildcard Encoder",
    "BawkSampler": "🐓 Bawk Sampler (All-in-One)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

__version__ = "2.0.1"
__author__ = "Bawk Nodes"
__description__ = "A complete collection of FLUX-optimized ComfyUI nodes for enhanced workflows"

print(f"🐓 ComfyUI Bawk Nodes v{__version__} loaded successfully!")
print("   🎉 Major Update - Complete FLUX Workflow Suite!")
print("   Current nodes:")
print("   • 🚀 Diffusion Model Loader (Advanced) - FLUX-optimized model loading")
print("   • 🎲 FLUX Wildcard Encoder - Text encoding with wildcard support and 6 LoRA slots")
print("   • 💾 FLUX Image Saver - Organized image saving with metadata")
print("   • 📝 FLUX Prompt Saver - Save prompts and generation parameters")
print("   • 🐓 Bawk Sampler (All-in-One) - Combined latent optimizer and sampler")
print("   📦 Modular architecture for easy maintenance and debugging")
print("   Visit: https://github.com/juddisjudd/ComfyUI-BawkNodes")