"""
FluxWildcardEncode - LoRA application + {a|b|c} wildcards + FLUX text encoding.
"""

import random
import re

from .lora_stack import apply_lora_stack, collect_lora_slots, lora_slot_inputs

WILDCARD_RE = re.compile(r"\{([^{}]*\|[^{}]*)\}")


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
            },
            "optional": {
                "lora_stack": ("LORA_STACK", {
                    "tooltip": "LoRA set(s) from Bawk LoRA Stack. Applied before the slots below",
                }),
                **lora_slot_inputs(),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "CONDITIONING", "PROMPT_OUT")
    FUNCTION = "encode"
    CATEGORY = "BawkNodes/conditioning"
    DESCRIPTION = (
        "Apply LoRAs, resolve {a|b|c} wildcards and encode the prompt for FLUX. "
        "FLUX has no negative prompt; guidance is set in Bawk Sampler."
    )

    def encode(self, model, clip, prompt, wildcard_seed=0, lora_stack=None, **kwargs):
        stack = list(lora_stack or []) + collect_lora_slots(kwargs)
        model, clip = apply_lora_stack(model, clip, stack)

        final_prompt = process_wildcards(prompt, wildcard_seed)
        tokens = clip.tokenize(final_prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return (model, clip, conditioning, final_prompt)
