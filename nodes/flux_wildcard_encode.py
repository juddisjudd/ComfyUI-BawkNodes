"""
FluxWildcardEncode Node with LoRA Support (using rgthree-style dynamic UI)
File: nodes/flux_wildcard_encode.py
"""

import re
import random
import torch
import folder_paths
from typing import Tuple, Any


# Fallback implementation for flexible inputs
class FlexibleOptionalInputType(dict):
    def __init__(self, type_func, data=None):
        super().__init__()
        self.data = data or {}
        self.type = type_func
        for key, value in self.data.items():
            self[key] = value
    
    def __contains__(self, key):
        return True
    
    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        return self.type


def any_type(s=None):
    return ("*",)


class FluxWildcardEncode:
    """
    Enhanced FLUX wildcard text encoder with integrated LoRA loading.
    Uses rgthree-style dynamic UI for clean LoRA management.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available LoRA files
        lora_list = folder_paths.get_filename_list("loras")
        lora_options = ["None"] + lora_list
        
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "wildcard_seed": ("INT", {
                    "default": -1, "min": -1, "max": 0xffffffffffffffff,
                    "tooltip": "Seed for wildcard selection. Use -1 to disable wildcards"
                }),
            },
            "optional": {
                # LoRA 1
                "lora_1_on": ("BOOLEAN", {"default": False}),
                "lora_1_name": (lora_options, {"default": "None"}),
                "lora_1_strength": ("FLOAT", {"default": 1.00, "min": -10.0, "max": 10.0, "step": 0.01}),
                
                # LoRA 2
                "lora_2_on": ("BOOLEAN", {"default": False}),
                "lora_2_name": (lora_options, {"default": "None"}),
                "lora_2_strength": ("FLOAT", {"default": 1.00, "min": -10.0, "max": 10.0, "step": 0.01}),
                
                # LoRA 3
                "lora_3_on": ("BOOLEAN", {"default": False}),
                "lora_3_name": (lora_options, {"default": "None"}),
                "lora_3_strength": ("FLOAT", {"default": 1.00, "min": -10.0, "max": 10.0, "step": 0.01}),
                
                # LoRA 4
                "lora_4_on": ("BOOLEAN", {"default": False}),
                "lora_4_name": (lora_options, {"default": "None"}),
                "lora_4_strength": ("FLOAT", {"default": 1.00, "min": -10.0, "max": 10.0, "step": 0.01}),
                
                # LoRA 5
                "lora_5_on": ("BOOLEAN", {"default": False}),
                "lora_5_name": (lora_options, {"default": "None"}),
                "lora_5_strength": ("FLOAT", {"default": 1.00, "min": -10.0, "max": 10.0, "step": 0.01}),
                
                # LoRA 6
                "lora_6_on": ("BOOLEAN", {"default": False}),
                "lora_6_name": (lora_options, {"default": "None"}),
                "lora_6_strength": ("FLOAT", {"default": 1.00, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "CONDITIONING", "PROMPT_OUT")
    FUNCTION = "encode_with_loras"
    CATEGORY = "BawkNodes/conditioning"
    DESCRIPTION = "🎲 FLUX Wildcard Encoder with Dynamic LoRA Support"
    
    def encode_with_loras(self, model, clip, prompt, wildcard_seed=-1, **kwargs):
        """
        Encode text prompt with wildcard processing and dynamic LoRA loading
        """
        try:
            print(f"[FluxWildcardEncode] Starting encode with dynamic LoRAs")
            print(f"[FluxWildcardEncode] Received kwargs: {list(kwargs.keys())}")
            
            # Step 1: Process wildcards if seed is provided
            processed_prompt = prompt
            if wildcard_seed != -1:
                processed_prompt = self._process_wildcards(prompt, wildcard_seed)
                if processed_prompt != prompt:
                    print(f"[FluxWildcardEncode] Processed wildcards in prompt")
            
            # Step 2: Apply LoRAs from fixed slots
            working_model = model
            working_clip = clip
            ui_lora_count = 0
            
            # Process 6 fixed LoRA slots
            for i in range(1, 7):  # LoRA 1-6
                on_key = f"lora_{i}_on"
                name_key = f"lora_{i}_name"
                strength_key = f"lora_{i}_strength"
                
                # Check if this LoRA slot is enabled and has a valid selection
                lora_enabled = kwargs.get(on_key, False)
                lora_name = kwargs.get(name_key, "None")
                
                if lora_enabled and lora_name and lora_name != "None":
                    strength = kwargs.get(strength_key, 1.00)
                    
                    # Skip if strength is zero
                    if strength == 0:
                        continue
                    
                    try:
                        from nodes import LoraLoader
                        working_model, working_clip = LoraLoader().load_lora(
                            working_model, working_clip, lora_name, strength, strength
                        )
                        ui_lora_count += 1
                        print(f"[FluxWildcardEncode] Applied LoRA {i}: {lora_name} (strength: {strength})")
                    except Exception as e:
                        print(f"[FluxWildcardEncode] Failed to load LoRA {i} ({lora_name}): {e}")
            
            if ui_lora_count > 0:
                print(f"[FluxWildcardEncode] Applied {ui_lora_count} LoRAs")
            else:
                print(f"[FluxWildcardEncode] No LoRAs applied")
            
            # Step 3: Encode the processed prompt
            if not processed_prompt.strip():
                print("[FluxWildcardEncode] Warning: Empty prompt")
                empty_conditioning = [[torch.zeros((1, 77, 768)), {"pooled_output": torch.zeros((1, 768))}]]
                return (working_model, working_clip, empty_conditioning, processed_prompt)
            
            # Encode using the LoRA-modified CLIP
            tokens = working_clip.tokenize(processed_prompt)
            conditioning, pooled = working_clip.encode_from_tokens(tokens, return_pooled=True)
            conditioning = [[conditioning, {"pooled_output": pooled}]]
            
            print(f"[FluxWildcardEncode] Success: {ui_lora_count} LoRAs, conditioning shape: {conditioning[0][0].shape}")
            return (working_model, working_clip, conditioning, processed_prompt)
            
        except Exception as e:
            error_msg = f"Failed to encode with LoRAs: {str(e)}"
            print(f"[FluxWildcardEncode] Error: {error_msg}")
            import traceback
            traceback.print_exc()
            empty_conditioning = [[torch.zeros((1, 77, 768)), {"pooled_output": torch.zeros((1, 768))}]]
            return (model, clip, empty_conditioning, prompt)
    
    def _process_wildcards(self, prompt, seed):
        """
        Process wildcard syntax in prompts
        Wildcard format: {option1|option2|option3}
        """
        # Set seed for consistent wildcard selection
        random.seed(seed)
        
        processed = prompt
        
        # Find and process wildcards in {option1|option2|option3} format
        wildcard_pattern = r'\{([^}]+)\}'
        
        def replace_wildcard(match):
            wildcard_content = match.group(1)
            if '|' in wildcard_content:
                options = [option.strip() for option in wildcard_content.split('|')]
                if options:  # Make sure we have options
                    return random.choice(options)
            return match.group(0)  # Return original if no valid options
        
        # Replace all wildcards
        processed = re.sub(wildcard_pattern, replace_wildcard, processed)
        
        return processed
    
    def _get_lora_by_filename(self, file_path):
        """Find LoRA file with fuzzy matching (rgthree style)"""
        if not file_path:
            return None
            
        lora_paths = folder_paths.get_filename_list('loras')
        
        # Direct match
        if file_path in lora_paths:
            return file_path
        
        # Match without extension
        lora_paths_no_ext = [os.path.splitext(x)[0] for x in lora_paths]
        if file_path in lora_paths_no_ext:
            return lora_paths[lora_paths_no_ext.index(file_path)]
        
        # Force input without extension
        file_path_no_ext = os.path.splitext(file_path)[0]
        if file_path_no_ext in lora_paths_no_ext:
            return lora_paths[lora_paths_no_ext.index(file_path_no_ext)]
        
        # Basename matching
        lora_basenames = [os.path.basename(x) for x in lora_paths]
        if file_path in lora_basenames:
            return lora_paths[lora_basenames.index(file_path)]
        
        # Basename without extension
        file_basename = os.path.basename(file_path)
        if file_basename in lora_basenames:
            return lora_paths[lora_basenames.index(file_basename)]
        
        # Basename no extension matching
        lora_basenames_no_ext = [os.path.splitext(os.path.basename(x))[0] for x in lora_paths]
        file_basename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        if file_basename_no_ext in lora_basenames_no_ext:
            return lora_paths[lora_basenames_no_ext.index(file_basename_no_ext)]
        
        # Fuzzy partial match
        for lora_path in lora_paths:
            if file_path in lora_path:
                print(f"[FluxWildcardEncode] Fuzzy-matched '{file_path}' to '{lora_path}'")
                return lora_path
        
        print(f"[FluxWildcardEncode] LoRA '{file_path}' not found")
        return None