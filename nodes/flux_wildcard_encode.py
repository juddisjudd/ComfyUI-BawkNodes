"""
FluxWildcardEncode - LoRA application + trigger words + {a|b|c} wildcards + FLUX text encoding.
"""

import random
import re

from .lora_stack import apply_lora_stack, collect_lora_slots, lora_slot_inputs, merge_stacks, stack_info

WILDCARD_RE = re.compile(r"\{([^{}]*\|[^{}]*)\}")
TRIGGER_MODES = ["prepend", "append", "off"]


def process_wildcards(prompt, seed):
    rng = random.Random(seed)

    def pick(match):
        options = [o.strip() for o in match.group(1).split("|")]
        return rng.choice(options)

    for _ in range(10):  # inner groups resolve first, so nested wildcards work
        resolved = WILDCARD_RE.sub(pick, prompt)
        if resolved == prompt:
            break
        prompt = resolved
    return prompt


def add_triggers(prompt, triggers, mode):
    triggers = [t for t in triggers if t and t.lower() not in prompt.lower()]
    if mode == "off" or not triggers:
        return prompt
    joined = ", ".join(triggers)
    if not prompt.strip():
        return joined
    return f"{joined}, {prompt}" if mode == "prepend" else f"{prompt.rstrip().rstrip(',')}, {joined}"


class FluxWildcardEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False,
                    "tooltip": "Prompt. Use {red|green|blue} to pick one option at random",
                }),
                "wildcard_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Seed for wildcard picks. Same seed = same picks",
                }),
                "trigger_words": (TRIGGER_MODES, {
                    "default": "prepend",
                    "tooltip": "Where LoRA trigger words go. Words already in the prompt are not repeated",
                }),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", {
                    "tooltip": "LoRA set(s) from Bawk LoRA Stack / Switch. Applied before the slots below",
                }),
                **lora_slot_inputs(with_triggers=True),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "CONDITIONING", "PROMPT_OUT", "LORA_INFO")
    FUNCTION = "encode"
    CATEGORY = "BawkNodes/conditioning"
    DESCRIPTION = (
        "Apply LoRAs, add their trigger words, resolve {a|b|c} wildcards and encode the prompt for FLUX. "
        "FLUX has no negative prompt; guidance is set in Bawk Sampler."
    )

    def encode(self, model, clip, prompt, wildcard_seed=0, trigger_words="prepend", lora_stack=None, **kwargs):
        stack = merge_stacks(lora_stack, collect_lora_slots(kwargs))
        model, clip = apply_lora_stack(model, clip, stack)

        final_prompt = process_wildcards(add_triggers(prompt, stack["triggers"], trigger_words), wildcard_seed)
        tokens = clip.tokenize(final_prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return (model, clip, conditioning, final_prompt, stack_info(stack))
