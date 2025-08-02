"""
Validation utilities for ComfyUI Diffusion Model Loader
File: modules/validation.py
"""

import os
import torch
from typing import List, Dict, Any, Union, Optional
import folder_paths


class ValidationMixin:
    """
    Mixin class providing comprehensive validation capabilities
    for model loading operations
    """
    
    @staticmethod
    def validate_dtype_compatibility(dtype_str: str, device: torch.device) -> bool:
        """
        Validate data type compatibility with target device
        
        Args:
            dtype_str: String representation of data type
            device: Target PyTorch device
            
        Returns:
            True if compatible, False otherwise
        """
        # Valid dtype mappings
        valid_dtypes = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
            "fp8_e5m2": getattr(torch, "float8_e5m2", None),
            "default": None  # Auto-selection
        }
        
        if dtype_str not in valid_dtypes:
            return False
        
        target_dtype = valid_dtypes[dtype_str]
        if target_dtype is None:  # Auto-selection or unsupported type
            return True
        
        # Check device-specific compatibility
        if device.type == "cpu":
            # CPU supports most types except FP8
            return target_dtype not in [
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e5m2", None)
            ]
        
        elif device.type == "cuda":
            # Check CUDA capabilities
            try:
                # FP8 requires modern GPU architectures
                if target_dtype in [getattr(torch, "float8_e4m3fn", None), 
                                   getattr(torch, "float8_e5m2", None)]:
                    compute_capability = torch.cuda.get_device_capability(device)
                    return compute_capability >= (8, 9)  # Ada Lovelace or newer
                
                # bfloat16 support check
                if target_dtype == torch.bfloat16:
                    return torch.cuda.is_bf16_supported()
                
                # fp16 is generally supported on modern CUDA devices
                if target_dtype == torch.float16:
                    return True
                
                # fp32 is always supported
                if target_dtype == torch.float32:
                    return True
                    
            except Exception:
                # If we can't determine capability, assume fp16/fp32 work
                return target_dtype in [torch.float16, torch.float32]
        
        # For other device types, be conservative
        return target_dtype in [torch.float16, torch.float32]
    
    @staticmethod
    def validate_file_exists(file_path: str, file_type: str = "model") -> tuple[bool, str]:
        """
        Validate that a file exists and is accessible
        
        Args:
            file_path: Path to file
            file_type: Type of file for error messages
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path:
            return False, f"{file_type} path is empty"
        
        if not os.path.exists(file_path):
            return False, f"{file_type} file does not exist: {file_path}"
        
        if not os.path.isfile(file_path):
            return False, f"{file_type} path is not a file: {file_path}"
        
        # Check file size (basic corruption check)
        try:
            file_size = os.path.getsize(file_path)
            if file_size < 1024:  # Less than 1KB
                return False, f"{file_type} file appears to be empty or corrupted"
        except OSError as e:
            return False, f"Cannot access {file_type} file: {str(e)}"
        
        # Check read permissions
        if not os.access(file_path, os.R_OK):
            return False, f"No read permission for {file_type} file: {file_path}"
        
        return True, ""
    
    @staticmethod
    def validate_model_format(file_path: str) -> tuple[bool, str]:
        """
        Validate model file format
        
        Args:
            file_path: Path to model file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path:
            return False, "No file path provided"
        
        valid_extensions = {
            '.safetensors': 'SafeTensors',
            '.ckpt': 'Checkpoint',
            '.pth': 'PyTorch',
            '.pt': 'PyTorch',
            '.bin': 'Binary'
        }
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension not in valid_extensions:
            valid_exts = ', '.join(valid_extensions.keys())
            return False, f"Unsupported file format '{file_extension}'. Supported: {valid_exts}"
        
        return True, f"Valid {valid_extensions[file_extension]} format"
    
    @staticmethod
    def validate_device_availability(device_str: str) -> tuple[bool, str]:
        """
        Validate device availability
        
        Args:
            device_str: Device string (e.g., 'cuda:0', 'cpu', 'auto')
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if device_str == "auto":
            return True, "Auto device selection"
        
        if device_str == "cpu":
            return True, "CPU device available"
        
        if device_str.startswith("cuda"):
            if not torch.cuda.is_available():
                return False, "CUDA not available on this system"
            
            # Extract device ID if specified
            if ":" in device_str:
                try:
                    device_id = int(device_str.split(":")[1])
                    if device_id >= torch.cuda.device_count():
                        return False, f"CUDA device {device_id} not available (only {torch.cuda.device_count()} devices found)"
                    
                    # Check if device is accessible
                    try:
                        torch.cuda.get_device_properties(device_id)
                        return True, f"CUDA device {device_id} available"
                    except Exception as e:
                        return False, f"Cannot access CUDA device {device_id}: {str(e)}"
                        
                except ValueError:
                    return False, f"Invalid CUDA device ID in '{device_str}'"
            else:
                return True, "CUDA device available"
        
        return False, f"Unknown device type: {device_str}"
    
    @staticmethod
    def validate_memory_requirements(model_path: str, device_str: str) -> tuple[bool, str]:
        """
        Validate memory requirements for model loading
        
        Args:
            model_path: Path to model file
            device_str: Target device string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Estimate model memory requirements
            file_size = os.path.getsize(model_path)
            estimated_memory = file_size * 2  # Conservative estimate
            
            if device_str == "cpu" or device_str == "auto":
                # For CPU/auto, we can't easily check system RAM, so assume it's OK
                return True, "Memory requirements assumed OK for CPU/auto"
            
            if device_str.startswith("cuda"):
                if not torch.cuda.is_available():
                    return False, "CUDA not available"
                
                device_id = 0
                if ":" in device_str:
                    try:
                        device_id = int(device_str.split(":")[1])
                    except ValueError:
                        return False, f"Invalid device string: {device_str}"
                
                if device_id >= torch.cuda.device_count():
                    return False, f"Device {device_id} not available"
                
                # Check VRAM
                try:
                    device_props = torch.cuda.get_device_properties(device_id)
                    total_memory = device_props.total_memory
                    
                    # Leave some headroom (20%)
                    available_memory = total_memory * 0.8
                    
                    if estimated_memory > available_memory:
                        required_gb = estimated_memory / (1024**3)
                        available_gb = available_memory / (1024**3)
                        return False, (f"Insufficient VRAM. Required: ~{required_gb:.1f}GB, "
                                     f"Available: ~{available_gb:.1f}GB")
                    
                    return True, "Sufficient VRAM available"
                    
                except Exception as e:
                    return False, f"Error checking VRAM: {str(e)}"
            
            return True, "Memory requirements check passed"
            
        except Exception as e:
            return False, f"Error validating memory requirements: {str(e)}"


class ModelValidator:
    """
    Comprehensive model validation class with detailed checking
    """
    
    def __init__(self):
        self.validation_cache = {}
    
    def validate_model_inputs(self, **kwargs) -> Dict[str, Any]:
        """
        Comprehensive validation of all model inputs
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        # Validate model name
        model_name = kwargs.get("model_name")
        if model_name:
            self._validate_model_file(model_name, results)
        else:
            results["errors"].append("Model name is required")
            results["is_valid"] = False
        
        # Validate weight dtype
        weight_dtype = kwargs.get("weight_dtype", "default")
        device = kwargs.get("device", "auto")
        self._validate_dtype_and_device(weight_dtype, device, results)
        
        # Validate VAE
        vae_name = kwargs.get("vae_name", "baked VAE")
        if vae_name != "baked VAE":
            self._validate_vae_file(vae_name, results)
        
        # Validate CLIP models
        clip_name1 = kwargs.get("clip_name1", "none")
        clip_name2 = kwargs.get("clip_name2", "none")
        self._validate_clip_models(clip_name1, clip_name2, results)
        
        return results
    
    def _validate_model_file(self, model_name: str, results: Dict):
        """Validate main model file"""
        model_path = folder_paths.get_full_path("checkpoints", model_name)
        
        if model_path is None:
            results["errors"].append(f"Checkpoint '{model_name}' not found in checkpoints directory")
            results["is_valid"] = False
            return
        
        # Check file existence and format
        file_valid, file_msg = ValidationMixin.validate_file_exists(model_path, "Model")
        if not file_valid:
            results["errors"].append(file_msg)
            results["is_valid"] = False
            return
        
        format_valid, format_msg = ValidationMixin.validate_model_format(model_path)
        if not format_valid:
            results["errors"].append(format_msg)
            results["is_valid"] = False
        else:
            results["info"].append(format_msg)
        
        # Check file size and provide info
        try:
            file_size_gb = os.path.getsize(model_path) / (1024**3)
            results["info"].append(f"Model size: {file_size_gb:.1f}GB")
            
            if file_size_gb > 20:
                results["warnings"].append("Very large model - may require significant VRAM")
            elif file_size_gb < 1:
                results["warnings"].append("Small model file - ensure this is not a corrupted download")
        except Exception as e:
            results["warnings"].append(f"Could not determine model size: {str(e)}")
    
    def _validate_dtype_and_device(self, weight_dtype: str, device: str, results: Dict):
        """Validate data type and device compatibility"""
        # Validate device
        device_valid, device_msg = ValidationMixin.validate_device_availability(device)
        if not device_valid:
            results["errors"].append(device_msg)
            results["is_valid"] = False
        else:
            results["info"].append(device_msg)
        
        # Validate dtype compatibility
        if device_valid and weight_dtype != "default":
            try:
                torch_device = torch.device(device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
                dtype_valid = ValidationMixin.validate_dtype_compatibility(weight_dtype, torch_device)
                
                if not dtype_valid:
                    results["warnings"].append(f"Data type '{weight_dtype}' may not be supported on {device}, will fall back to auto-selection")
                else:
                    results["info"].append(f"Data type '{weight_dtype}' compatible with {device}")
            except Exception as e:
                results["warnings"].append(f"Could not validate dtype compatibility: {str(e)}")
    
    def _validate_vae_file(self, vae_name: str, results: Dict):
        """Validate VAE file"""
        vae_path = folder_paths.get_full_path("vae", vae_name)
        
        if vae_path is None:
            results["errors"].append(f"VAE '{vae_name}' not found in vae directory")
            results["is_valid"] = False
            return
        
        file_valid, file_msg = ValidationMixin.validate_file_exists(vae_path, "VAE")
        if not file_valid:
            results["errors"].append(file_msg)
            results["is_valid"] = False
        else:
            results["info"].append(f"VAE file validated: {vae_name}")
    
    def _validate_clip_models(self, clip_name1: str, clip_name2: str, results: Dict):
        """Validate CLIP model files"""
        clip_models = []
        
        if clip_name1 != "none":
            clip_models.append((clip_name1, "CLIP1"))
        
        if clip_name2 != "none":
            clip_models.append((clip_name2, "CLIP2"))
        
        for clip_name, clip_label in clip_models:
            clip_path = folder_paths.get_full_path("clip", clip_name)
            
            if clip_path is None:
                results["errors"].append(f"{clip_label} '{clip_name}' not found in clip directory")
                results["is_valid"] = False
                continue
            
            file_valid, file_msg = ValidationMixin.validate_file_exists(clip_path, clip_label)
            if not file_valid:
                results["errors"].append(file_msg)
                results["is_valid"] = False
            else:
                results["info"].append(f"{clip_label} file validated: {clip_name}")
        
        # Check for dual CLIP configuration consistency
        if clip_name1 != "none" and clip_name2 == "none":
            results["warnings"].append("Only one CLIP model specified - dual text encoder models typically require both")
        elif clip_name1 == "none" and clip_name2 != "none":
            results["warnings"].append("Only second CLIP model specified - consider specifying the first as well")
    
    def get_validation_summary(self, validation_results: Dict[str, Any]) -> str:
        """Generate human-readable validation summary"""
        if validation_results["is_valid"]:
            summary = "✓ All validations passed"
        else:
            summary = "✗ Validation failed"
        
        if validation_results["errors"]:
            summary += f"\nErrors ({len(validation_results['errors'])}):"
            for error in validation_results["errors"]:
                summary += f"\n  - {error}"
        
        if validation_results["warnings"]:
            summary += f"\nWarnings ({len(validation_results['warnings'])}):"
            for warning in validation_results["warnings"]:
                summary += f"\n  - {warning}"
        
        if validation_results["info"]:
            summary += f"\nInfo ({len(validation_results['info'])}):"
            for info in validation_results["info"]:
                summary += f"\n  - {info}"
        
        return summary