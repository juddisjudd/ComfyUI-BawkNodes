"""
FluxWildcardEncode Node with LoRA Support (using rgthree-style dynamic UI)
File: nodes/flux_wildcard_encode.py
"""

import os
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
                "lora_1_on": ("BOOLEAN", {"default": False, "tooltip": "Enable LoRA 1 - Use for primary style/character modifications"}),
                "lora_1_name": (lora_options, {"default": "None", "tooltip": "Select LoRA file for slot 1 - Choose your main style or character LoRA"}),
                "lora_1_strength": ("FLOAT", {"default": 1.00, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA 1 strength - Recommended: 0.6-1.2 for styles, 0.8-1.0 for characters"}),

                # LoRA 2
                "lora_2_on": ("BOOLEAN", {"default": False, "tooltip": "Enable LoRA 2 - Use for secondary effects or style blending"}),
                "lora_2_name": (lora_options, {"default": "None", "tooltip": "Select LoRA file for slot 2 - Good for lighting or pose adjustments"}),
                "lora_2_strength": ("FLOAT", {"default": 0.80, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA 2 strength - Recommended: 0.5-0.8 for subtle effects, 0.8-1.2 for strong effects"}),

                # LoRA 3
                "lora_3_on": ("BOOLEAN", {"default": False, "tooltip": "Enable LoRA 3 - Use for clothing, objects, or environmental effects"}),
                "lora_3_name": (lora_options, {"default": "None", "tooltip": "Select LoRA file for slot 3 - Perfect for clothing or object LoRAs"}),
                "lora_3_strength": ("FLOAT", {"default": 0.70, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA 3 strength - Recommended: 0.4-0.7 for clothing, 0.6-1.0 for objects"}),

                # LoRA 4
                "lora_4_on": ("BOOLEAN", {"default": False, "tooltip": "Enable LoRA 4 - Use for fine details or texture adjustments"}),
                "lora_4_name": (lora_options, {"default": "None", "tooltip": "Select LoRA file for slot 4 - Good for detail enhancement or texture LoRAs"}),
                "lora_4_strength": ("FLOAT", {"default": 0.60, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA 4 strength - Recommended: 0.3-0.6 for subtle details, 0.5-0.8 for noticeable changes"}),

                # LoRA 5
                "lora_5_on": ("BOOLEAN", {"default": False, "tooltip": "Enable LoRA 5 - Use for color grading or mood adjustments"}),
                "lora_5_name": (lora_options, {"default": "None", "tooltip": "Select LoRA file for slot 5 - Ideal for color/mood LoRAs or experimental combinations"}),
                "lora_5_strength": ("FLOAT", {"default": 0.50, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA 5 strength - Recommended: 0.2-0.5 for color adjustments, 0.4-0.7 for experimental effects"}),

                # LoRA 6
                "lora_6_on": ("BOOLEAN", {"default": False, "tooltip": "Enable LoRA 6 - Use for final touches or very subtle adjustments"}),
                "lora_6_name": (lora_options, {"default": "None", "tooltip": "Select LoRA file for slot 6 - Use for final polish or very specific adjustments"}),
                "lora_6_strength": ("FLOAT", {"default": 0.40, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA 6 strength - Recommended: 0.1-0.4 for subtle effects, negative values to reduce certain features"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "CONDITIONING", "PROMPT_OUT")
    FUNCTION = "encode_with_loras"
    CATEGORY = "BawkNodes/conditioning"
    DESCRIPTION = "FLUX text encoder with wildcard support and 6 LoRA slots"
    
    def encode_with_loras(self, model, clip, prompt, wildcard_seed=-1, **kwargs):
        """
        Encode text prompt with wildcard processing and dynamic LoRA loading
        """
        try:
            print(f"[FluxWildcardEncode] Starting encode with dynamic LoRAs")
            print(f"[FluxWildcardEncode] Received kwargs: {list(kwargs.keys())}")

            # Clean up kwargs - handle ComfyUI's string conversion issues
            cleaned_kwargs = self._clean_kwargs(kwargs)

            # Step 1: Process wildcards if seed is provided
            processed_prompt = prompt
            if wildcard_seed != -1:
                processed_prompt = self._process_wildcards(prompt, wildcard_seed)
                if processed_prompt != prompt:
                    print(f"[FluxWildcardEncode] Processed wildcards in prompt")
            
            # Step 2: Apply LoRAs from fixed slots with smart validation
            working_model = model
            working_clip = clip
            ui_lora_count = 0
            lora_warnings = []

            # Process 6 fixed LoRA slots
            for i in range(1, 7):  # LoRA 1-6
                on_key = f"lora_{i}_on"
                name_key = f"lora_{i}_name"
                strength_key = f"lora_{i}_strength"

                # Get cleaned values for this LoRA slot
                lora_enabled = cleaned_kwargs.get(on_key, False)
                lora_name = cleaned_kwargs.get(name_key, "None")
                lora_strength = cleaned_kwargs.get(strength_key, 1.0)

                # Validate LoRA configuration and provide feedback
                if lora_enabled and lora_name != "None":
                    self._validate_lora_config(i, lora_name, lora_strength, lora_warnings)

                if lora_enabled and lora_name and lora_name != "None":
                    strength = lora_strength
                    
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

    def _validate_lora_config(self, slot_num, lora_name, strength, warnings_list):
        """Validate LoRA configuration and provide helpful feedback"""

        # Check strength values and provide recommendations
        if strength > 2.0:
            warnings_list.append(f"LoRA {slot_num}: Very high strength ({strength:.2f}) may cause artifacts or overfitting")
            print(f"[FluxWildcardEncode] ⚠️  LoRA {slot_num} strength ({strength:.2f}) is very high - consider reducing to 0.6-1.2")
        elif strength > 1.5:
            print(f"[FluxWildcardEncode] ℹ️  LoRA {slot_num} strength ({strength:.2f}) is high - monitor for quality issues")

        if strength < 0.1 and strength > 0:
            print(f"[FluxWildcardEncode] ℹ️  LoRA {slot_num} strength ({strength:.2f}) is very low - effect may be minimal")
        elif strength < 0:
            print(f"[FluxWildcardEncode] ℹ️  LoRA {slot_num} using negative strength ({strength:.2f}) to reduce features")

        # Provide slot-specific recommendations based on typical usage patterns
        slot_recommendations = {
            1: "Primary LoRA - Best for main character or style (recommended: 0.8-1.2)",
            2: "Secondary LoRA - Good for lighting or pose adjustments (recommended: 0.6-1.0)",
            3: "Detail LoRA - Perfect for clothing or objects (recommended: 0.5-0.8)",
            4: "Texture LoRA - Use for material or surface details (recommended: 0.4-0.7)",
            5: "Mood LoRA - Ideal for color grading or atmosphere (recommended: 0.3-0.6)",
            6: "Polish LoRA - Final touches or subtle adjustments (recommended: 0.2-0.5)"
        }

        if slot_num <= 6:
            print(f"[FluxWildcardEncode] ✅ LoRA {slot_num} loaded: {lora_name} @ {strength:.2f} ({slot_recommendations[slot_num]})")

        # Check for common LoRA naming patterns to suggest optimal strengths
        lora_lower = lora_name.lower()
        if "style" in lora_lower and strength < 0.6:
            print(f"[FluxWildcardEncode] 💡 Style LoRAs typically work best at 0.8-1.2 strength")
        elif "character" in lora_lower and strength < 0.7:
            print(f"[FluxWildcardEncode] 💡 Character LoRAs typically work best at 0.8-1.1 strength")
        elif "pose" in lora_lower and strength > 1.0:
            print(f"[FluxWildcardEncode] 💡 Pose LoRAs often work better at 0.6-0.9 strength")
        elif "lighting" in lora_lower and strength > 0.8:
            print(f"[FluxWildcardEncode] 💡 Lighting LoRAs typically work best at 0.4-0.7 strength")


    def _clean_kwargs(self, kwargs):
        """Clean kwargs to handle ComfyUI's string conversion issues"""
        cleaned = {}

        for key, value in kwargs.items():
            if key.endswith('_on'):
                # Boolean parameters
                if isinstance(value, str):
                    cleaned[key] = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    cleaned[key] = bool(value)
            elif key.endswith('_strength'):
                # Float parameters
                if isinstance(value, str):
                    if value.lower() in ('none', '', 'null'):
                        cleaned[key] = 1.0  # Default strength
                    else:
                        try:
                            cleaned[key] = float(value)
                        except ValueError:
                            cleaned[key] = 1.0  # Default on error
                else:
                    cleaned[key] = float(value) if value is not None else 1.0
            elif key.endswith('_name'):
                # String parameters (LoRA names)
                if isinstance(value, str):
                    if value.lower() in ('false', 'none', '', 'null'):
                        cleaned[key] = "None"
                    else:
                        cleaned[key] = value
                else:
                    cleaned[key] = str(value) if value is not None else "None"
            else:
                # Other parameters - pass through
                cleaned[key] = value

        return cleaned