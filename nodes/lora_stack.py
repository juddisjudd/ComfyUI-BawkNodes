"""
BawkLoraStack - build LoRA sets that can be chained, toggled and applied in one go.
"""

import inspect
import logging

import comfy.sd
import comfy.utils
import folder_paths

LORA_SLOTS = 6
_LORA_METADATA_KW = "lora_metadata" in inspect.signature(comfy.sd.load_lora_for_models).parameters


def lora_slot_inputs(slots=LORA_SLOTS):
    loras = ["None"] + folder_paths.get_filename_list("loras")
    inputs = {}
    for i in range(1, slots + 1):
        inputs[f"lora_{i}"] = (loras, {"default": "None"})
        inputs[f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
    return inputs


def collect_lora_slots(kwargs, slots=LORA_SLOTS):
    stack = []
    for i in range(1, slots + 1):
        name = kwargs.get(f"lora_{i}", "None")
        strength = float(kwargs.get(f"strength_{i}", 1.0))
        if name and name != "None" and strength != 0.0:
            stack.append((name, strength))
    return stack


def apply_lora_stack(model, clip, lora_stack):
    for name, strength in lora_stack:
        path = folder_paths.get_full_path_or_raise("loras", name)
        lora, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        extra = {"lora_metadata": metadata} if _LORA_METADATA_KW else {}
        model, clip = comfy.sd.load_lora_for_models(model, clip, lora, strength, strength, **extra)
        logging.info(f"[BawkNodes] LoRA {name} @ {strength}")
    return model, clip


class BawkLoraStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Master switch for this set. Off passes the incoming stack through unchanged",
                }),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", {"tooltip": "Chain another Bawk LoRA Stack to combine sets"}),
                **lora_slot_inputs(),
            },
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "build"
    CATEGORY = "BawkNodes/loaders"
    DESCRIPTION = (
        "A set of up to 6 LoRAs. Chain several stacks and flip 'enabled' to switch between sets. "
        "Plug into Bawk Wildcard Encoder's lora_stack input."
    )

    def build(self, enabled=True, lora_stack=None, **kwargs):
        stack = list(lora_stack) if lora_stack else []
        if enabled:
            stack += collect_lora_slots(kwargs)
        return (stack,)
