"""
FluxDynamicLoraStack - Dynamic LoRA loading with add/remove controls
File: nodes/flux_dynamic_lora.py
"""

import folder_paths
from typing import Tuple, Any


class FluxDynamicLoraStack:
    """
    Dynamic LoRA stack that allows adding/removing LoRA slots on demand.
    Outputs a list of (lora_name, strength) tuples that can be applied to a model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")

        return {
            "required": {
                "lora_count": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of LoRA slots (change this to add/remove slots dynamically)"
                }),
            },
            "optional": {
                # First LoRA (always present)
                "lora_1": (["None"] + lora_list, {"default": "None"}),
                "strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "build_lora_stack"
    CATEGORY = "BawkNodes/loaders"
    DESCRIPTION = "🔗 Dynamic LoRA stack builder - adjust lora_count to add/remove up to 20 LoRA slots on demand"

    def build_lora_stack(self, lora_count=1, **kwargs):
        """
        Build a list of (lora_name, strength) tuples from the dynamic inputs
        """
        lora_stack = []

        for i in range(1, lora_count + 1):
            lora_name = kwargs.get(f"lora_{i}", "None")
            strength = kwargs.get(f"strength_{i}", 1.0)

            if lora_name and lora_name != "None":
                lora_stack.append((lora_name, strength))
                print(f"[FluxDynamicLoraStack] Added LoRA {i}: {lora_name} @ {strength}")

        print(f"[FluxDynamicLoraStack] Built stack with {len(lora_stack)} LoRAs")
        return (lora_stack,)


class FluxLoraStackApply:
    """
    Applies a LoRA stack to a model and CLIP
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_stack": ("LORA_STACK",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    FUNCTION = "apply_lora_stack"
    CATEGORY = "BawkNodes/loaders"
    DESCRIPTION = "⚡ Apply LoRA stack to model and CLIP - processes all LoRAs from FluxDynamicLoraStack"

    def apply_lora_stack(self, model, clip, lora_stack):
        """
        Apply all LoRAs in the stack to the model and CLIP
        """
        if not lora_stack or len(lora_stack) == 0:
            print("[FluxLoraStackApply] No LoRAs in stack, returning unchanged")
            return (model, clip)

        # Import LoRA loading utilities
        import comfy.utils
        import comfy.sd

        for lora_name, strength in lora_stack:
            lora_path = folder_paths.get_full_path("loras", lora_name)

            if lora_path is None:
                print(f"[FluxLoraStackApply] WARNING: LoRA not found: {lora_name}")
                continue

            print(f"[FluxLoraStackApply] Loading LoRA: {lora_name} @ strength {strength}")

            # Load LoRA
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

            # Apply to model and clip
            model, clip = comfy.sd.load_lora_for_models(
                model, clip, lora, strength, strength
            )

        print(f"[FluxLoraStackApply] Applied {len(lora_stack)} LoRAs")
        return (model, clip)


# Export nodes
__all__ = ['FluxDynamicLoraStack', 'FluxLoraStackApply']
