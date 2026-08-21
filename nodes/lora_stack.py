"""
BawkLoraStack / BawkLoraSwitch - LoRA sets with trigger words, chainable and switchable.

LORA_STACK is a dict: {"loras": [(name, strength), ...], "triggers": [str, ...]}
"""

import inspect
import logging

import comfy.sd
import comfy.utils
import folder_paths

LORA_SLOTS = 6
SWITCH_SLOTS = ["A", "B", "C", "D"]
_LORA_METADATA_KW = "lora_metadata" in inspect.signature(comfy.sd.load_lora_for_models).parameters


def empty_stack():
    return {"loras": [], "triggers": []}


def normalize_stack(stack):
    if not stack:
        return empty_stack()
    if isinstance(stack, dict):
        return {"loras": list(stack.get("loras", [])), "triggers": list(stack.get("triggers", []))}
    return {"loras": list(stack), "triggers": []}


def merge_stacks(*stacks):
    out = empty_stack()
    for stack in stacks:
        stack = normalize_stack(stack)
        out["loras"] += stack["loras"]
        out["triggers"] += [t for t in stack["triggers"] if t and t not in out["triggers"]]
    return out


def lora_slot_inputs(slots=LORA_SLOTS, with_triggers=False):
    loras = ["None"] + folder_paths.get_filename_list("loras")
    inputs = {}
    for i in range(1, slots + 1):
        inputs[f"lora_{i}"] = (loras, {"default": "None"})
        inputs[f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
        if with_triggers:
            inputs[f"trigger_{i}"] = ("STRING", {
                "default": "", "dynamicPrompts": False,
                "tooltip": "Trigger words for this LoRA. Added to the prompt only while this slot is active",
            })
    return inputs


def collect_lora_slots(kwargs, slots=LORA_SLOTS):
    stack = empty_stack()
    for i in range(1, slots + 1):
        name = kwargs.get(f"lora_{i}", "None")
        strength = float(kwargs.get(f"strength_{i}", 1.0))
        if name and name != "None" and strength != 0.0:
            stack["loras"].append((name, strength))
            trigger = (kwargs.get(f"trigger_{i}") or "").strip().strip(",").strip()
            if trigger and trigger not in stack["triggers"]:
                stack["triggers"].append(trigger)
    return stack


def stack_info(stack):
    return ", ".join(f"{name}@{strength:g}" for name, strength in normalize_stack(stack)["loras"])


def apply_lora_stack(model, clip, stack):
    for name, strength in normalize_stack(stack)["loras"]:
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
                **lora_slot_inputs(with_triggers=True),
            },
        }

    RETURN_TYPES = ("LORA_STACK", "STRING", "STRING")
    RETURN_NAMES = ("lora_stack", "lora_triggers", "lora_info")
    FUNCTION = "build"
    CATEGORY = "BawkNodes/loras"
    DESCRIPTION = (
        "A set of up to 6 LoRAs with per-LoRA trigger words. Chain stacks to combine sets, "
        "or feed several into Bawk LoRA Switch. Plug into Bawk Wildcard Encoder's lora_stack input."
    )

    def build(self, enabled=True, lora_stack=None, **kwargs):
        stack = merge_stacks(lora_stack, collect_lora_slots(kwargs) if enabled else None)
        return (stack, ", ".join(stack["triggers"]), stack_info(stack))


class BawkLoraSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "active": (SWITCH_SLOTS, {"default": "A", "tooltip": "Which LoRA set to use"}),
            },
            "optional": {f"set_{s.lower()}": ("LORA_STACK", {"tooltip": f"LoRA set {s}"}) for s in SWITCH_SLOTS},
        }

    RETURN_TYPES = ("LORA_STACK", "STRING", "STRING")
    RETURN_NAMES = ("lora_stack", "lora_triggers", "lora_info")
    FUNCTION = "switch"
    CATEGORY = "BawkNodes/loras"
    DESCRIPTION = "Pick one of up to four LoRA sets. Only the chosen set is loaded."

    def switch(self, active="A", **kwargs):
        stack = normalize_stack(kwargs.get(f"set_{active.lower()}"))
        return (stack, ", ".join(stack["triggers"]), stack_info(stack))
