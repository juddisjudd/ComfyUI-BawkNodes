"""
ComfyUI Bawk Nodes - compact FLUX.1 / FLUX.2 workflow nodes.
"""

from .nodes import (
    BawkBatchProcessor,
    BawkImageLoader,
    BawkLoraStack,
    BawkLoraSwitch,
    BawkSampler,
    DiffusionModelLoader,
    FluxImageSaver,
    FluxWildcardEncode,
)

NODE_CLASS_MAPPINGS = {
    "DiffusionModelLoader": DiffusionModelLoader,
    "BawkLoraStack": BawkLoraStack,
    "BawkLoraSwitch": BawkLoraSwitch,
    "FluxWildcardEncode": FluxWildcardEncode,
    "BawkSampler": BawkSampler,
    "FluxImageSaver": FluxImageSaver,
    "BawkImageLoader": BawkImageLoader,
    "BawkBatchProcessor": BawkBatchProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionModelLoader": "🚀 Bawk Model Loader",
    "BawkLoraStack": "🔗 Bawk LoRA Stack",
    "BawkLoraSwitch": "🔀 Bawk LoRA Switch",
    "FluxWildcardEncode": "🎲 Bawk Wildcard Encoder",
    "BawkSampler": "🐓 Bawk Sampler",
    "FluxImageSaver": "💾 Bawk Image Saver",
    "BawkImageLoader": "📸 Bawk Image Loader",
    "BawkBatchProcessor": "📁 Bawk Batch Processor",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
