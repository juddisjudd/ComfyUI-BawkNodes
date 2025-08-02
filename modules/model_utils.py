"""
Model Utilities for ComfyUI Diffusion Model Loader
File: modules/model_utils.py
"""

import os
import json
import torch
from typing import Dict, Any, Optional, Union
from pathlib import Path


class ModelUtils:
    """
    Centralized utilities for model handling, metadata extraction,
    and memory estimation
    """
    
    def __init__(self):
        self._model_cache = {}
        self._memory_estimates = {}
    
    def estimate_model_memory(self, file_path: str) -> int:
        """
        Estimate memory usage for model file in bytes
        
        Args:
            file_path: Path to model file
            
        Returns:
            Estimated memory usage in bytes
        """
        if file_path in self._memory_estimates:
            return self._memory_estimates[file_path]
        
        try:
            file_size = os.path.getsize(file_path)
            
            # Different expansion factors based on file type
            if file_path.endswith('.safetensors'):
                # SafeTensors are more memory efficient
                expansion_factor = 1.6
            elif file_path.endswith('.ckpt') or file_path.endswith('.pth'):
                # Standard checkpoints expand more
                expansion_factor = 2.0
            else:
                # Default conservative estimate
                expansion_factor = 1.8
            
            # Additional overhead for FLUX models (they're typically larger)
            if self._is_likely_flux_model(file_path):
                expansion_factor *= 1.2
            
            estimated_memory = int(file_size * expansion_factor)
            self._memory_estimates[file_path] = estimated_memory
            
            return estimated_memory
            
        except (OSError, IOError) as e:
            print(f"Warning: Could not estimate memory for {file_path}: {e}")
            return 4 * 1024**3  # Default 4GB estimate
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive model metadata and information
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dictionary containing model information
        """
        cache_key = f"info_{model_path}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        info = {
            "file_size": 0,
            "file_name": os.path.basename(model_path),
            "format": self._detect_format(model_path),
            "estimated_memory_mb": 0,
            "metadata": {},
            "model_type": "unknown",
            "architecture": "unknown"
        }
        
        try:
            # Basic file information
            info["file_size"] = os.path.getsize(model_path)
            info["estimated_memory_mb"] = self.estimate_model_memory(model_path) // (1024**2)
            
            # Extract metadata based on format
            if info["format"] == "safetensors":
                info["metadata"] = self._extract_safetensors_metadata(model_path)
            elif info["format"] == "checkpoint":
                info["metadata"] = self._extract_checkpoint_metadata(model_path)
            
            # Infer model type and architecture
            info["model_type"] = self._infer_model_type(model_path, info["metadata"])
            info["architecture"] = self._infer_architecture(model_path, info["metadata"])
            
            # Cache the result
            self._model_cache[cache_key] = info
            
        except Exception as e:
            print(f"Warning: Could not extract full info for {model_path}: {e}")
        
        return info
    
    def _detect_format(self, file_path: str) -> str:
        """Detect model file format"""
        extension = Path(file_path).suffix.lower()
        
        if extension == ".safetensors":
            return "safetensors"
        elif extension in [".ckpt", ".pth", ".pt"]:
            return "checkpoint"
        else:
            return "unknown"
    
    def _extract_safetensors_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from SafeTensors file"""
        try:
            from safetensors import safe_open
            
            with safe_open(file_path, framework="pt") as f:
                metadata = f.metadata() or {}
                
                # Convert to regular dict and handle JSON strings
                parsed_metadata = {}
                for key, value in metadata.items():
                    try:
                        # Try to parse JSON values
                        parsed_metadata[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        # Keep as string if not JSON
                        parsed_metadata[key] = value
                
                return parsed_metadata
                
        except ImportError:
            print("Warning: safetensors library not available for metadata extraction")
            return {}
        except Exception as e:
            print(f"Warning: Could not extract SafeTensors metadata: {e}")
            return {}
    
    def _extract_checkpoint_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from checkpoint file"""
        try:
            # Load only the metadata without the full model
            checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
            
            metadata = {}
            
            # Common metadata keys
            if isinstance(checkpoint, dict):
                for key in ["metadata", "meta", "info", "config"]:
                    if key in checkpoint:
                        metadata[key] = checkpoint[key]
                
                # Check for state dict structure info
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                    metadata["num_parameters"] = sum(p.numel() for p in state_dict.values() if torch.is_tensor(p))
                    metadata["state_dict_keys"] = list(state_dict.keys())[:10]  # First 10 keys for inspection
            
            return metadata
            
        except Exception as e:
            print(f"Warning: Could not extract checkpoint metadata: {e}")
            return {}
    
    def _infer_model_type(self, file_path: str, metadata: Dict[str, Any]) -> str:
        """Infer model type from file name and metadata"""
        file_name = os.path.basename(file_path).lower()
        
        # Check file name patterns
        if "flux" in file_name:
            return "flux"
        elif "sdxl" in file_name:
            return "sdxl"
        elif "sd3" in file_name:
            return "sd3"
        elif any(term in file_name for term in ["sd_v1", "v1-5", "1.5"]):
            return "sd1.5"
        elif "sd_v2" in file_name or "v2" in file_name:
            return "sd2.x"
        
        # Check metadata for clues
        if metadata:
            metadata_str = str(metadata).lower()
            if "flux" in metadata_str:
                return "flux"
            elif "sdxl" in metadata_str:
                return "sdxl"
        
        return "unknown"
    
    def _infer_architecture(self, file_path: str, metadata: Dict[str, Any]) -> str:
        """Infer model architecture from metadata and file characteristics"""
        file_size_gb = os.path.getsize(file_path) / (1024**3)
        
        # Size-based inference
        if file_size_gb > 20:
            return "transformer_large"  # Likely FLUX or similar
        elif file_size_gb > 6:
            return "unet_xl"  # Likely SDXL
        elif file_size_gb > 3:
            return "unet_large"  # Likely SD2.x
        elif file_size_gb > 1:
            return "unet_base"  # Likely SD1.5
        
        return "unknown"
    
    def _is_likely_flux_model(self, file_path: str) -> bool:
        """Check if model is likely a FLUX model"""
        file_name = os.path.basename(file_path).lower()
        return "flux" in file_name or os.path.getsize(file_path) > 15 * 1024**3  # > 15GB
    
    def get_compatible_devices(self, model_path: str) -> list:
        """Get list of devices compatible with this model"""
        compatible_devices = ["cpu"]
        
        if torch.cuda.is_available():
            estimated_memory = self.estimate_model_memory(model_path)
            
            for i in range(torch.cuda.device_count()):
                device_memory = torch.cuda.get_device_properties(i).total_memory
                if estimated_memory < device_memory * 0.8:  # Leave 20% headroom
                    compatible_devices.append(f"cuda:{i}")
        
        return compatible_devices
    
    def optimize_loading_strategy(self, model_path: str, target_device: str) -> Dict[str, Any]:
        """
        Recommend optimal loading strategy based on model and hardware
        
        Returns:
            Dictionary with optimization recommendations
        """
        model_info = self.get_model_info(model_path)
        estimated_memory = model_info["estimated_memory_mb"] * 1024**2  # Convert to bytes
        
        strategy = {
            "recommended_dtype": "fp16",
            "use_cpu_offload": False,
            "chunk_loading": False,
            "memory_format": "channels_last"
        }
        
        if target_device.startswith("cuda"):
            try:
                device_id = int(target_device.split(":")[1]) if ":" in target_device else 0
                available_memory = torch.cuda.get_device_properties(device_id).total_memory
                
                # Adjust strategy based on memory pressure
                memory_ratio = estimated_memory / available_memory
                
                if memory_ratio > 0.9:
                    strategy["recommended_dtype"] = "fp8_e4m3fn"
                    strategy["use_cpu_offload"] = True
                    strategy["chunk_loading"] = True
                elif memory_ratio > 0.7:
                    strategy["recommended_dtype"] = "fp16"
                    strategy["use_cpu_offload"] = True
                elif memory_ratio > 0.5:
                    strategy["recommended_dtype"] = "fp16"
                else:
                    strategy["recommended_dtype"] = "fp32"
                    
            except Exception:
                pass  # Fallback to default strategy
        
        return strategy
    
    @staticmethod
    def debug_directory_contents():
        """Debug function to print directory contents"""
        directories = [
            ("checkpoints", "checkpoints"),
            ("diffusion_models", "diffusion_models"), 
            ("vae", "vae"),
            ("clip", "clip"),
            ("text_encoders", "text_encoders")
        ]
        
        print("\n=== ComfyUI Directory Contents Debug ===")
        for dir_name, folder_key in directories:
            try:
                files = folder_paths.get_filename_list(folder_key) or []
                print(f"{dir_name}: {len(files)} files")
                if files:
                    for i, file in enumerate(files[:5]):  # Show first 5 files
                        print(f"  - {file}")
                    if len(files) > 5:
                        print(f"  ... and {len(files) - 5} more")
                else:
                    print(f"  (empty or not found)")
                    # Try to get the actual path to see if directory exists
                    try:
                        folder_path = folder_paths.get_folder_paths(folder_key)
                        if folder_path:
                            print(f"  Path: {folder_path[0] if isinstance(folder_path, list) else folder_path}")
                        else:
                            print(f"  Path: Not configured")
                    except:
                        print(f"  Path: Could not determine")
            except Exception as e:
                print(f"{dir_name}: Error accessing - {str(e)}")
        print("=========================================\n")
    
    def clear_cache(self):
        """Clear internal caches"""
        self._model_cache.clear()
        self._memory_estimates.clear()