"""
FluxPromptSaver Node
File: nodes/flux_prompt_saver.py
"""

import os
import json
import re
import folder_paths
from datetime import datetime
from typing import Tuple


class FluxPromptSaver:
    """
    Standalone prompt saver for FLUX workflows
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_string": ("STRING", {"forceInput": True}),
                "prompt": ("STRING", {"multiline": True}),
                "filename_prefix": ("STRING", {"default": "flux_prompt"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "generation_params": ("STRING", {"multiline": True, "default": ""}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_prompt"
    CATEGORY = "BawkNodes/text"
    OUTPUT_NODE = True
    DESCRIPTION = "Save FLUX prompts and generation parameters"
    
    def save_prompt(self, model_string, prompt, filename_prefix="flux_prompt",
                   negative_prompt="", generation_params="", extra_pnginfo=None):
        """Save prompt and parameters to organized text files"""
        try:
            # Get current date for folder structure
            now = datetime.now()
            date_str = now.strftime("%d-%m-%Y")
            datetime_str = now.strftime("%d-%m-%Y_%H-%M-%S")
            
            # Clean model string for folder name
            clean_model = self._clean_filename(model_string)
            
            # Create folder structure matching image saver
            folder_name = f"[{clean_model}]-{date_str}"
            output_dir = os.path.join(folder_paths.get_output_directory(), folder_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename
            filename = f"{filename_prefix}_{datetime_str}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Prepare data
            prompt_data = {
                "timestamp": now.isoformat(),
                "model": model_string,
                "positive_prompt": prompt,
                "negative_prompt": negative_prompt,
                "generation_parameters": generation_params,
                "workflow_info": extra_pnginfo
            }
            
            # Save as JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(prompt_data, f, indent=2, ensure_ascii=False)
            
            print(f"[FluxPromptSaver] Saved: {filepath}")
            return (f"Prompt saved to {filename}",)
            
        except Exception as e:
            error_msg = f"Failed to save prompt: {str(e)}"
            print(f"[FluxPromptSaver] Error: {error_msg}")
            return (f"Error: {error_msg}",)
    
    def _clean_filename(self, filename):
        """Clean filename for use in folder/file names"""
        clean = re.sub(r'\.[^.]*$', '', filename)
        clean = re.sub(r'[<>:"/\\|?*]', '_', clean)
        clean = re.sub(r'_{2,}', '_', clean)
        return clean.strip('_')