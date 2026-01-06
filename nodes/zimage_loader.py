"""
ZImageModelLoader - Load Z-Image models using native ComfyUI support
File: nodes/zimage_loader.py

Z-Image models are natively supported in ComfyUI via comfy.supported_models.ZImage.
This loader provides a simplified interface focused on Z-Image-Turbo settings.
"""

import os
import torch
import comfy.sd
import comfy.utils
import folder_paths
from typing import Tuple, Any


class ZImageModelLoader:
    """
    Load Z-Image-Turbo model using native ComfyUI support.
    Z-Image uses a single-stream DiT architecture and works with standard samplers.
    
    Outputs standard MODEL, VAE, CLIP types that work with BawkSampler.
    Recommended settings: steps=8, guidance=0.0 (distilled model, no CFG needed)
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get available diffusion models, prioritize Z-Image models
        model_list = folder_paths.get_filename_list("diffusion_models")
        zimage_models = [m for m in model_list if "zimage" in m.lower() or "z-image" in m.lower()]
        all_models = zimage_models + [m for m in model_list if m not in zimage_models]

        return {
            "required": {
                "model_name": (all_models if all_models else ["None - No models found"], {
                    "tooltip": "Z-Image model file from models/diffusion_models/"
                }),
                "vae_name": (folder_paths.get_filename_list("vae") + ["baked VAE"], {
                    "tooltip": "VAE model. Z-Image typically uses FLUX VAE (ae.safetensors)"
                }),
                "clip_name": (folder_paths.get_filename_list("text_encoders") + ["none"], {
                    "tooltip": "Text encoder for Z-Image (Gemma-based)"
                }),
                "weight_dtype": (["default", "bf16", "fp16", "fp8_e4m3fn"], {
                    "default": "bf16",
                    "tooltip": "Model precision. bf16 recommended for Z-Image (best quality/speed)"
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "VAE", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "VAE", "CLIP", "MODEL_STRING")
    FUNCTION = "load_zimage_model"
    CATEGORY = "BawkNodes/loaders"
    DESCRIPTION = "⚡ Load Z-Image-Turbo model - 6B params, 8 steps, state-of-the-art quality. Use with BawkSampler (steps=8, guidance=0.0)"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate model files exist"""
        errors = []
        
        model_name = kwargs.get("model_name", "")
        if not model_name or model_name == "None - No models found":
            errors.append("No Z-Image model selected. Place model in ComfyUI/models/diffusion_models/")
        elif model_name:
            try:
                folder_paths.get_full_path_or_raise("diffusion_models", model_name)
            except:
                errors.append(f"Model '{model_name}' not found in diffusion_models directory")
        
        vae_name = kwargs.get("vae_name", "baked VAE")
        if vae_name and vae_name != "baked VAE":
            try:
                folder_paths.get_full_path_or_raise("vae", vae_name)
            except:
                errors.append(f"VAE '{vae_name}' not found in vae directory")
        
        clip_name = kwargs.get("clip_name", "none")
        if clip_name and clip_name != "none":
            try:
                folder_paths.get_full_path_or_raise("text_encoders", clip_name)
            except:
                errors.append(f"Text encoder '{clip_name}' not found in text_encoders directory")
        
        return "; ".join(errors) if errors else True

    def load_zimage_model(
        self,
        model_name: str,
        vae_name: str = "baked VAE",
        clip_name: str = "none",
        weight_dtype: str = "bf16"
    ) -> Tuple[Any, Any, Any, str]:
        """
        Load Z-Image model using native ComfyUI support
        """
        try:
            # Handle "none" and "baked VAE" values
            load_vae = vae_name and vae_name != "baked VAE"
            load_clip = clip_name and clip_name != "none"
            
            # Set up model options based on weight_dtype
            model_options = {}
            if weight_dtype == "bf16":
                model_options["dtype"] = torch.bfloat16
            elif weight_dtype == "fp16":
                model_options["dtype"] = torch.float16
            elif weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            
            # Load diffusion model (ComfyUI auto-detects Z-Image)
            print(f"[ZImageModelLoader] Loading Z-Image model: {model_name}")
            model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
            model = comfy.sd.load_diffusion_model(model_path, model_options=model_options)
            
            if model is None:
                raise RuntimeError("Failed to load Z-Image model - output was None")
            
            # Load VAE
            vae = None
            if load_vae:
                print(f"[ZImageModelLoader] Loading VAE: {vae_name}")
                vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
                vae_sd = comfy.utils.load_torch_file(vae_path)
                vae = comfy.sd.VAE(sd=vae_sd)
            
            # Load CLIP/text encoder
            clip = None
            if load_clip:
                print(f"[ZImageModelLoader] Loading text encoder: {clip_name}")
                clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
                clip = comfy.sd.load_clip(
                    ckpt_paths=[clip_path],
                    embedding_directory=folder_paths.get_folder_paths("embeddings")
                )
            
            # Generate model name string
            model_string = os.path.splitext(model_name)[0]
            
            print(f"[ZImageModelLoader] ✅ Successfully loaded Z-Image model: {model_string}")
            print(f"[ZImageModelLoader] 💡 Recommended: steps=8, guidance=0.0 (Turbo model)")
            return (model, vae, clip, model_string)
            
        except Exception as e:
            error_msg = f"Failed to load Z-Image model '{model_name}': {str(e)}"
            print(f"[ZImageModelLoader] ❌ {error_msg}")
            raise RuntimeError(error_msg)


# Export
__all__ = ['ZImageModelLoader']
