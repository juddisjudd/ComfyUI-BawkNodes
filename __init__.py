"""
ComfyUI Bawk Nodes - Node Registration
File: __init__.py
"""

from .nodes import DiffusionModelLoader

# Node class mappings for ComfyUI discovery
NODE_CLASS_MAPPINGS = {
    "DiffusionModelLoader": DiffusionModelLoader,
}

# Display names in the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionModelLoader": "🚀 Diffusion Model Loader (Advanced)",
}

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Package metadata
__version__ = "1.0.0"
__author__ = "Bawk Nodes"
__description__ = "A collection of useful ComfyUI nodes for enhanced workflows"

# Optional: Print loading message
print(f"🐓 ComfyUI Bawk Nodes v{__version__} loaded successfully!")
print("   Current nodes:")
print("   • 🚀 Diffusion Model Loader (Advanced) - FLUX-optimized model loading")
print("   • More nodes coming soon...")
print("   Visit: https://github.com/juddisjudd/ComfyUI-BawkNodes")