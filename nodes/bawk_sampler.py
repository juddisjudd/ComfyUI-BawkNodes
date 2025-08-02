"""
BawkSampler - Combined Latent Generator and Advanced Sampler for FLUX
File: nodes/bawk_sampler.py
"""

import torch
import math
import time
import logging
import comfy.samplers
import comfy.model_management as mm
from comfy.utils import ProgressBar
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, BasicScheduler, BasicGuider, SamplerCustomAdvanced
from comfy_extras.nodes_latent import LatentBatch
from comfy_extras.nodes_model_advanced import ModelSamplingFlux, ModelSamplingAuraFlow
from typing import Tuple, Dict, Any, List


def round_to_nearest_multiple(value, multiple):
    """Rounds a value to the nearest multiple of 'multiple'."""
    if multiple <= 0:
        return value
    return int(round(value / multiple) * multiple)


def parse_string_to_list(input_string):
    """Parse comma/newline separated string to list of numbers"""
    try:
        if not input_string:
            return []
        items = input_string.replace('\n', ',').split(',')
        result = []
        for item in items:
            item = item.strip()
            if not item:
                continue
            try:
                num = float(item)
                if num.is_integer():
                    num = int(num)
                result.append(num)
            except ValueError:
                continue
        return result
    except:
        return []


def conditioning_set_values(conditioning, values):
    """Set conditioning values like guidance scale"""
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k, v in values.items():
            if k == "guidance":
                n[1]['guidance_scale'] = v
        c.append(tuple(n))
    return c


class BawkSampler:
    """
    Combined latent generator and advanced sampler for FLUX models.
    Generates optimized latent dimensions, then performs advanced sampling.
    Designed to work with FluxWildcardEncode for prompt handling.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Combined resolution presets with aspect ratios built-in
        # Format: "Name - WxH (aspect ratio)"
        resolution_presets = [
            # 16:9 aspect ratio options
            "HD 16:9 - 1024x576",
            "FHD 16:9 - 1920x1080", 
            "2K 16:9 - 2048x1152",
            
            # 1:1 square options  
            "Small Square - 768x768",
            "Medium Square - 1024x1024",
            "Large Square - 1536x1536",
            "XL Square - 2048x2048",
            
            # 3:2 photo aspect ratio
            "Photo 3:2 - 1152x768",
            "Photo 3:2 HD - 1728x1152",
            
            # 4:3 classic aspect ratio
            "Classic 4:3 - 1024x768",
            "Classic 4:3 HD - 1536x1152",
            
            # 9:16 portrait/mobile
            "Portrait 9:16 - 576x1024",
            "Mobile 9:16 - 1080x1920",
            
            # 2:3 portrait photo
            "Portrait Photo - 768x1152",
            
            # 21:9 ultra-wide
            "Ultra-wide - 1792x768",
            
            # Custom option
            "Custom Resolution",
        ]
        
        return {
            "required": {
                # Core inputs
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),  # Added VAE input for decoding
                
                # Latent generation
                "resolution": (resolution_presets, {
                    "default": "FHD 16:9 - 1920x1080",
                    "tooltip": "Select resolution preset or custom option"
                }),
                "batch_size": ("INT", {
                    "default": 4, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Number of images to generate"
                }),
                
                # Sampling parameters
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for generation"
                }),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler",
                    "tooltip": "Sampling method for generation"
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "beta",
                    "tooltip": "Noise scheduling method"
                }),
                "steps": ("INT", {
                    "default": 30, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Number of sampling steps"
                }),
                "guidance": ("FLOAT", {
                    "default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Guidance scale for FLUX"
                }),
                "max_shift": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Max shift parameter for FLUX"
                }),
                "base_shift": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Base shift parameter for FLUX"
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise strength"
                }),
                
                # Custom resolution toggle
                "use_custom_resolution": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable custom width/height instead of presets"
                }),
            },
            "optional": {
                # Custom resolution (only used when use_custom_resolution=True)
                "custom_width": ("INT", {
                    "default": 1920, "min": 64, "max": 4096, "step": 64,
                    "tooltip": "Custom width (must be multiple of 64). Only used when custom resolution is enabled."
                }),
                "custom_height": ("INT", {
                    "default": 1080, "min": 64, "max": 4096, "step": 64,
                    "tooltip": "Custom height (must be multiple of 64). Only used when custom resolution is enabled."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("images", "latent")
    FUNCTION = "generate_sample_and_decode"
    CATEGORY = "BawkNodes/sampling"
    DESCRIPTION = "Combined latent generator, sampler, and VAE decoder optimized for FLUX models"
    
    def generate_sample_and_decode(
        self,
        model, conditioning, vae,
        resolution="FHD 16:9 - 1920x1080", batch_size=4,
        seed=0, sampler="euler", scheduler="beta", steps=30,
        guidance=3.5, max_shift=0.5, base_shift=0.3, denoise=1.0,
        use_custom_resolution=False, custom_width=1920, custom_height=1080
    ):
        """
        Generate optimized latent, perform FLUX sampling, and decode to images
        """
        try:
            # Step 1: Generate optimized latent
            latent = self._generate_optimized_latent(
                resolution, batch_size, use_custom_resolution, custom_width, custom_height
            )
            
            # Step 2: Perform sampling
            sampled_latent = self._perform_flux_sampling(
                model, conditioning, latent, seed, sampler, scheduler,
                steps, guidance, max_shift, base_shift, denoise
            )
            
            # Step 3: Decode latent to images using VAE
            decoded_images = self._decode_latent_to_images(vae, sampled_latent)
            
            print(f"[BawkSampler] Successfully generated, sampled, and decoded {batch_size} images")
            
            return (decoded_images, sampled_latent)
            
        except Exception as e:
            error_msg = f"BawkSampler failed: {str(e)}"
            print(f"[BawkSampler] Error: {error_msg}")
            raise RuntimeError(error_msg)
    
    def _generate_optimized_latent(
        self, resolution, batch_size, use_custom_resolution, custom_width, custom_height
    ) -> Dict:
        """Generate optimized empty latent for FLUX"""
        
        # Determine target dimensions
        if use_custom_resolution:
            # Use custom dimensions (ensure they're multiples of 64)
            target_width = round_to_nearest_multiple(custom_width, 64)
            target_height = round_to_nearest_multiple(custom_height, 64)
            print(f"[BawkSampler] Using custom resolution: {target_width}x{target_height}")
        else:
            # Parse dimensions from preset string
            target_width, target_height = self._parse_resolution_preset(resolution)
            print(f"[BawkSampler] Using preset resolution: {resolution}")
        
        # FLUX-specific parameters
        vae_scale_factor = 8
        latent_channels = 16  # FLUX uses 16 channels
        
        # Calculate latent dimensions
        latent_width = target_width // vae_scale_factor
        latent_height = target_height // vae_scale_factor
        
        # Generate latent tensor
        try:
            latent_tensor = torch.zeros([batch_size, latent_channels, latent_height, latent_width])
            latent = {"samples": latent_tensor}
        except Exception as e:
            raise RuntimeError(f"Error creating FLUX latent tensor [{batch_size}, {latent_channels}, {latent_height}, {latent_width}]: {e}")
        
        print(f"[BawkSampler] Generated FLUX latent: {latent_width}x{latent_height} "
              f"(pixel: {target_width}x{target_height}, batch: {batch_size})")
        
        return latent
    
    def _parse_resolution_preset(self, resolution_preset):
        """Parse width and height from resolution preset string"""
        try:
            # Extract dimensions from string like "FHD 16:9 - 1920x1080"
            # Split by " - " and take the part after it
            parts = resolution_preset.split(" - ")
            if len(parts) < 2:
                raise ValueError(f"Invalid resolution format: {resolution_preset}")
            
            dimensions = parts[1]  # e.g., "1920x1080"
            width_str, height_str = dimensions.split("x")
            
            target_width = int(width_str)
            target_height = int(height_str)
            
            # Ensure dimensions are multiples of 64 for FLUX compatibility
            target_width = round_to_nearest_multiple(target_width, 64)
            target_height = round_to_nearest_multiple(target_height, 64)
            
            return target_width, target_height
            
        except Exception as e:
            print(f"[BawkSampler] Warning: Could not parse resolution '{resolution_preset}': {e}")
            # Fallback to default FHD resolution
            return 1920, 1080
    
    def _perform_flux_sampling(
        self, model, conditioning, latent_image, seed, sampler, scheduler,
        steps, guidance, max_shift, base_shift, denoise
    ):
        """Perform FLUX-optimized sampling"""
        
        # Single parameter sampling (no multi-parameter sweeps for cleaner UX)
        noise_seed = seed
        
        # Initialize FLUX sampling components
        basicscheduler = BasicScheduler()
        basicguider = BasicGuider()
        samplercustomadvanced = SamplerCustomAdvanced()
        modelsamplingflux = ModelSamplingFlux()
        randnoise = Noise_RandomNoise(noise_seed)
        
        # Get dimensions for model sampling
        width = latent_image["samples"].shape[3] * 8
        height = latent_image["samples"].shape[2] * 8
        
        # Apply FLUX model sampling
        work_model = modelsamplingflux.patch(model, max_shift, base_shift, width, height)[0]
        
        # Set guidance in conditioning
        cond = conditioning_set_values(conditioning, {"guidance": guidance})
        guider = basicguider.get_guider(work_model, cond)[0]
        
        # Create sampler object
        samplerobj = comfy.samplers.sampler_object(sampler)
        
        # Generate sigmas
        sigmas = basicscheduler.get_sigmas(work_model, scheduler, steps, denoise)[0]
        
        logging.info(f"FLUX Sampling: seed={noise_seed}, {sampler}_{scheduler}, "
                   f"steps={steps}, guidance={guidance}, max_shift={max_shift}, base_shift={base_shift}")
        
        # Perform FLUX sampling
        latent = samplercustomadvanced.sample(
            randnoise, guider, samplerobj, sigmas, latent_image
        )[1]
        
        print(f"[BawkSampler] Completed FLUX sampling with {sampler}_{scheduler}")
        return latent
    
    def _decode_latent_to_images(self, vae, latent):
        """Decode latent to images using the provided VAE"""
        try:
            print(f"[BawkSampler] Decoding latent to images...")
            
            # Decode latent using VAE
            decoded_images = vae.decode(latent["samples"])
            
            print(f"[BawkSampler] Successfully decoded latent to images, shape: {decoded_images.shape}")
            return decoded_images
            
        except Exception as e:
            error_msg = f"Failed to decode latent: {str(e)}"
            print(f"[BawkSampler] Error: {error_msg}")
            raise RuntimeError(error_msg)