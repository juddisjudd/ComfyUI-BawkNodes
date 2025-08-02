"""
DiffusionModelLoader Node
File: nodes/diffusion_model_loader.py
"""

import os
import torch
import comfy.sd
import comfy.utils
import folder_paths
from typing import Tuple, Any

# Try importing validation from parent modules
try:
    from ..modules.validation import ValidationMixin
except ImportError:
    # Fallback if modules structure is different
    class ValidationMixin:
        pass


class DiffusionModelLoader(ValidationMixin):
    """
    Advanced Diffusion Model Loader with support for:
    - Multiple model formats (FLUX, SDXL, SD1.5)
    - Flexible weight data types (fp8, fp16, bf16, fp32)
    - Separate VAE and CLIP loading
    - Multiple directory support
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "vae_name": (folder_paths.get_filename_list("vae") + ["baked VAE"],),
                "clip_name1": (folder_paths.get_filename_list("text_encoders") + ["none"],),
                "clip_name2": (folder_paths.get_filename_list("text_encoders") + ["none"],),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
            }
        }
    
    RETURN_TYPES = ("MODEL", "VAE", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "VAE", "CLIP", "MODEL_STRING")
    FUNCTION = "load_diffusion_model"
    CATEGORY = "BawkNodes/loaders"
    DESCRIPTION = "Advanced diffusion model loader for FLUX and other modern architectures"
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Simple validation matching the working loader pattern"""
        errors = []
        
        # Validate model exists in diffusion_models directory
        model_name = kwargs.get("model_name")
        if model_name:
            try:
                folder_paths.get_full_path_or_raise("diffusion_models", model_name)
            except:
                errors.append(f"Model '{model_name}' not found in diffusion_models directory")
        
        # Validate VAE if specified
        vae_name = kwargs.get("vae_name", "baked VAE")
        if vae_name and vae_name != "baked VAE":
            try:
                folder_paths.get_full_path_or_raise("vae", vae_name)
            except:
                errors.append(f"VAE '{vae_name}' not found in vae directory")
        
        # Validate CLIP models if specified
        clip1 = kwargs.get("clip_name1", "none")
        if clip1 and clip1 != "none":
            try:
                folder_paths.get_full_path_or_raise("text_encoders", clip1)
            except:
                errors.append(f"CLIP model '{clip1}' not found in text_encoders directory")
        
        clip2 = kwargs.get("clip_name2", "none")
        if clip2 and clip2 != "none":
            try:
                folder_paths.get_full_path_or_raise("text_encoders", clip2)
            except:
                errors.append(f"CLIP model '{clip2}' not found in text_encoders directory")
        
        return "; ".join(errors) if errors else True
    
    def load_diffusion_model(
        self, 
        model_name: str, 
        weight_dtype: str = "default",
        vae_name: str = "baked VAE",
        clip_name1: str = "none", 
        clip_name2: str = "none"
    ) -> Tuple[Any, Any, Any, str]:
        """
        Main loading function for diffusion models
        """
        try:
            # Handle "none" and "baked VAE" values
            vae_name = vae_name if vae_name and vae_name != "baked VAE" else None
            clip_name1 = clip_name1 if clip_name1 and clip_name1 != "none" else None
            clip_name2 = clip_name2 if clip_name2 and clip_name2 != "none" else None
            weight_dtype = weight_dtype if weight_dtype else "default"
            
            # Set up model options based on weight_dtype
            model_options = {}
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2
            
            # Load diffusion model
            print(f"[DiffusionModelLoader] Loading diffusion model: {model_name}")
            model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
            model = comfy.sd.load_diffusion_model(model_path, model_options=model_options)
            
            if model is None:
                raise RuntimeError("Failed to load diffusion model - output was None")
            
            # Load VAE
            vae = None
            if vae_name:
                print(f"[DiffusionModelLoader] Loading VAE: {vae_name}")
                vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
                vae_sd = comfy.utils.load_torch_file(vae_path)
                vae = comfy.sd.VAE(sd=vae_sd)
            
            # Load CLIP models
            clip_paths = []
            if clip_name1:
                clip_paths.append(folder_paths.get_full_path_or_raise("text_encoders", clip_name1))
            if clip_name2:
                clip_paths.append(folder_paths.get_full_path_or_raise("text_encoders", clip_name2))
            
            clip = None
            if clip_paths:
                print(f"[DiffusionModelLoader] Loading CLIP models")
                clip = comfy.sd.load_clip(
                    ckpt_paths=clip_paths, 
                    embedding_directory=folder_paths.get_folder_paths("embeddings"), 
                    clip_type=comfy.sd.CLIPType.FLUX
                )
            
            # Generate simple model name string
            model_string = os.path.splitext(model_name)[0]
            
            print(f"[DiffusionModelLoader] Successfully loaded diffusion model: {model_string}")
            return (model, vae, clip, model_string)
            
        except Exception as e:
            error_msg = f"Failed to load diffusion model '{model_name}': {str(e)}"
            print(f"[DiffusionModelLoader] Error: {error_msg}")
            raise RuntimeError(error_msg)