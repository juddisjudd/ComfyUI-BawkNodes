"""
ComfyUI Bawk Nodes - Diffusion Model Loader
File: nodes.py
"""

import os
import torch
import comfy.sd
import comfy.model_management as mm
import comfy.utils
import folder_paths
from typing import Optional, Tuple, Dict, Any, Union
from .modules.validation import ValidationMixin


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
        # Match the exact format of the working DoomFluxLoader
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
        Main loading function for diffusion models (based on working DoomFluxLoader)
        """
        try:
            # Handle "none" and "baked VAE" values like the working loader
            vae_name = vae_name if vae_name and vae_name != "baked VAE" else None
            clip_name1 = clip_name1 if clip_name1 and clip_name1 != "none" else None
            clip_name2 = clip_name2 if clip_name2 and clip_name2 != "none" else None
            weight_dtype = weight_dtype if weight_dtype else "default"
            
            # Set up model options based on weight_dtype (like DoomFluxLoader)
            model_options = {}
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2
            
            # Load diffusion model using the exact same method as DoomFluxLoader
            print(f"[DiffusionModelLoader] Loading diffusion model: {model_name}")
            model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
            model = comfy.sd.load_diffusion_model(model_path, model_options=model_options)
            
            if model is None:
                raise RuntimeError("Failed to load diffusion model - output was None")
            
            # Load VAE (exactly like DoomFluxLoader)
            vae = None
            if vae_name:
                print(f"[DiffusionModelLoader] Loading VAE: {vae_name}")
                vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
                vae_sd = comfy.utils.load_torch_file(vae_path)
                vae = comfy.sd.VAE(sd=vae_sd)
            
            # Load CLIP models (exactly like DoomFluxLoader)
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
    
    def _create_cache_key(self, model_name: str, weight_dtype: str, vae_name: str, 
                         clip_name1: str, clip_name2: str) -> str:
        """Create unique cache key for this configuration"""
        return f"{model_name}|{weight_dtype}|{vae_name}|{clip_name1}|{clip_name2}"
    
    def _validate_model_path(self, model_name: str) -> str:
        """Validate model exists and is accessible, checking multiple directories"""
        model_path = None
        
        # Try checkpoints directory first
        model_path = folder_paths.get_full_path("checkpoints", model_name)
        
        # If not found, try diffusion_models directory
        if model_path is None:
            model_path = folder_paths.get_full_path("diffusion_models", model_name)
        
        if model_path is None:
            raise FileNotFoundError(f"Model not found in checkpoints or diffusion_models directories: {model_name}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        
        # Basic file size validation
        file_size = os.path.getsize(model_path)
        if file_size < 1024 * 1024:  # Less than 1MB
            raise ValueError(f"Model file appears corrupted (too small): {model_path}")
        
        return model_path
    
    def _select_dtype(self, weight_dtype: str, device: torch.device) -> torch.dtype:
        """Select appropriate data type based on user preference and hardware"""
        if weight_dtype == "default":
            return self._auto_select_dtype(device)
        
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        
        # Handle FP8 types if supported
        if weight_dtype.startswith("fp8_"):
            if self._supports_fp8(device):
                if weight_dtype == "fp8_e4m3fn":
                    return torch.float8_e4m3fn
                elif weight_dtype == "fp8_e5m2":
                    return torch.float8_e5m2
            else:
                print(f"FP8 not supported on device {device}, falling back to fp16")
                return torch.float16
        
        selected_dtype = dtype_map.get(weight_dtype, torch.float32)
        
        # Validate dtype compatibility
        if not self.validate_dtype_compatibility(weight_dtype, device):
            print(f"Dtype {weight_dtype} not compatible with {device}, using auto selection")
            return self._auto_select_dtype(device)
        
        return selected_dtype
    
    def _auto_select_dtype(self, device: torch.device) -> torch.dtype:
        """Automatically select best dtype for device"""
        if self._supports_fp8(device):
            free_memory = mm.get_free_memory(device)
            if free_memory < 6 * 1024**3:  # Less than 6GB VRAM
                return torch.float8_e4m3fn
        
        if mm.should_use_fp16(device):
            return torch.float16
        
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        
        return torch.float32
    
    def _supports_fp8(self, device: torch.device) -> bool:
        """Check if device supports FP8 computation"""
        if device.type != "cuda":
            return False
        
        try:
            compute_capability = torch.cuda.get_device_capability(device)
            return compute_capability >= (8, 9)  # Ada Lovelace or newer
        except:
            return False
    
    def _load_checkpoint_components(
        self, model_path: str, dtype: torch.dtype, device: torch.device
    ) -> Tuple[Any, Any, Any]:
        """Load main checkpoint components using ComfyUI's standard API"""
        try:
            # Use ComfyUI's memory management
            mm.soft_empty_cache()  # Clear cache before loading
            
            # Use the same API as CheckpointLoaderSimple
            out = comfy.sd.load_checkpoint_guess_config(
                model_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
            
            if out is None or len(out) < 3:
                raise RuntimeError("Failed to load checkpoint components")
            
            return out[:3]  # model, clip, vae
            
        except Exception as e:
            if "Could not detect model type" in str(e):
                raise RuntimeError(
                    f"Unsupported model format. This may be a FLUX model that requires "
                    f"separate component loading. Try using individual loaders for "
                    f"UNet/CLIP/VAE components."
                )
            raise RuntimeError(f"Error loading checkpoint: {str(e)}")
    
    def _load_separate_vae(self, vae_name: str):
        """Load separate VAE component"""
        vae_path = folder_paths.get_full_path("vae", vae_name)
        if vae_path is None:
            raise FileNotFoundError(f"VAE not found: {vae_name}")
        
        try:
            # Use ComfyUI's standard VAE loading
            vae_sd = comfy.utils.load_torch_file(vae_path, safe_load=True)
            vae = comfy.sd.VAE(sd=vae_sd)
            return vae
        except Exception as e:
            raise RuntimeError(f"Failed to load VAE '{vae_name}': {str(e)}")
    
    def _load_custom_clip(self, clip_name1: str, clip_name2: str):
        """Load custom CLIP models (for FLUX dual text encoders)"""
        clip_paths = []
        
        if clip_name1 != "none":
            # Try clip directory first, then text_encoders
            clip_path1 = folder_paths.get_full_path("clip", clip_name1)
            if clip_path1 is None:
                clip_path1 = folder_paths.get_full_path("text_encoders", clip_name1)
            
            if clip_path1 is None:
                raise FileNotFoundError(f"CLIP model not found in clip or text_encoders directories: {clip_name1}")
            clip_paths.append(clip_path1)
        
        if clip_name2 != "none":
            # Try clip directory first, then text_encoders
            clip_path2 = folder_paths.get_full_path("clip", clip_name2)
            if clip_path2 is None:
                clip_path2 = folder_paths.get_full_path("text_encoders", clip_name2)
            
            if clip_path2 is None:
                raise FileNotFoundError(f"CLIP model not found in clip or text_encoders directories: {clip_name2}")
            clip_paths.append(clip_path2)
        
        if not clip_paths:
            return None
        
        try:
            # Use ComfyUI's standard CLIP loading
            clip = comfy.sd.load_clip(
                ckpt_paths=clip_paths,
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
            return clip
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP models: {str(e)}")
    
    def _generate_model_info(
        self, model_name: str, dtype: torch.dtype, device: torch.device,
        vae_name: str, clip_name1: str, clip_name2: str
    ) -> str:
        """Generate model name string (just the base model name without extension)"""
        # Extract base name without extension
        base_name = os.path.splitext(model_name)[0]
        return base_name
    
    def get_model_string_preview(self, model_name: str, weight_dtype: str = "default", 
                                vae_name: str = "baked VAE", clip_name1: str = "none", 
                                clip_name2: str = "none") -> str:
        """
        Preview what the model string output will look like.
        Returns just the base model name without extension.
        """
        try:
            return os.path.splitext(model_name)[0]
        except Exception as e:
            return f"Error: {str(e)}"