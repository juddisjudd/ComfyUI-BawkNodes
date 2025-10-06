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
        # FLUX-optimized resolution presets based on official limits and community testing
        # Official FLUX range: 0.1MP (512x512) to 2.0MP (1408x1408 square / 1920x1080 wide)
        # Beyond 2.0MP: Unstable, may require upscaling post-processing
        resolution_presets = [
            # OPTIMAL SWEET SPOTS (Best quality/speed balance - Official FLUX range)
            "FLUX Optimal - 1024x1024 - 1.0MP (Recommended)",
            "Fast Portrait - 768x1024 - 0.8MP (3:4)",
            "Fast Landscape - 1024x768 - 0.8MP (4:3)",

            # QUICK TESTING (Minimum viable resolution)
            "Quick Test - 512x512 - 0.25MP (Minimum - Draft mode)",

            # OFFICIAL MAXIMUM RESOLUTION (2.0MP limit)
            "FLUX Max Square - 1408x1408 - 2.0MP (Official maximum)",
            "FLUX Max Wide - 1920x1080 - 2.1MP (Near official limit - May be unstable)",

            # SOCIAL MEDIA FORMATS (Within official range)
            "Instagram Square - 1024x1024 - 1.0MP (1:1)",
            "Instagram Portrait - 1024x1280 - 1.3MP (4:5)",
            "TikTok Vertical - 1088x1920 - 2.1MP (9:16 - Near limit)",

            # STABLE ALTERNATIVES (Well within official range)
            "Square Medium - 832x832 - 0.7MP (1:1)",
            "Portrait Classic - 768x1024 - 0.8MP (3:4)",
            "Landscape Classic - 1024x768 - 0.8MP (4:3)",
            "Wide Screen - 1152x768 - 0.9MP (3:2)",
            "Print Quality - 1344x896 - 1.2MP (3:2)",

            # EXPERIMENTAL (Beyond official 2.0MP - May be unstable)
            "⚠️ Experimental Square - 1536x1536 - 2.4MP (May need upscaler)",
            "⚠️ Experimental Wide - 1792x1024 - 1.8MP (May need upscaler)",

            # EXTREME (4.0MP+ - Unstable, requires post-processing)
            "⚠️ Extreme 4MP - 2048x2048 - 4.2MP (Unstable - Use upscaler)",

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
                    "default": "FLUX Optimal - 1024x1024 - 1.0MP (Recommended)",
                    "tooltip": "Resolution presets optimized for FLUX - 1024x1024 is the sweet spot for quality/speed"
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
                    "default": 28, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Sampling steps - Sweet spot: 25-30 (community tested for quality/speed balance)"
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
                    "tooltip": "Denoise strength - Use 1.0 for text-to-image, 0.6-0.9 for image-to-image"
                }),

                # Custom resolution toggle
                "use_custom_resolution": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable custom width/height instead of presets"
                }),
            },
            "optional": {
                # Img2Img support
                "input_image": ("IMAGE", {
                    "tooltip": "Input image for img2img generation. Leave empty for text-to-image mode."
                }),

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
    DESCRIPTION = "All-in-one FLUX sampler with text-to-image and image-to-image support, including VAE decoding"
    
    def generate_sample_and_decode(
        self,
        model, conditioning, vae,
        resolution="FHD 16:9 - 1920x1080 - 2.1MP", batch_size=4,
        seed=0, sampler="euler", scheduler="beta", steps=30,
        guidance=3.5, max_shift=0.5, base_shift=0.3, denoise=1.0,
        use_custom_resolution=False, input_image=None, custom_width=1920, custom_height=1080
    ):
        """
        Generate optimized latent, perform FLUX sampling, and decode to images
        """
        try:
            # Step 0: Smart validation with user feedback
            is_img2img = input_image is not None
            self._validate_parameters_with_feedback(
                model, batch_size, steps, guidance, max_shift, base_shift,
                resolution, use_custom_resolution, custom_width, custom_height, is_img2img, denoise
            )

            # Step 1: Generate or encode latent (img2img vs txt2img)
            if is_img2img:
                print(f"[BawkSampler] Using img2img mode with denoise strength: {denoise}")
                latent = self._encode_image_to_latent(vae, input_image, batch_size)
            else:
                print(f"[BawkSampler] Using text-to-image mode")
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
            error_msg, suggestion = self._get_error_with_solution(str(e))
            print(f"[BawkSampler] ❌ Error: {error_msg}")
            if suggestion:
                print(f"[BawkSampler] 💡 Suggestion: {suggestion}")
            raise RuntimeError(f"{error_msg}\nSuggestion: {suggestion}" if suggestion else error_msg)
    
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
            # Extract dimensions from strings like:
            # "FLUX Optimal - 1024x1024 - 1.0MP (Recommended)"
            # "Fast Portrait - 768x1024 - 0.8MP (3:4)"
            # "Pro Cinematic - 1920x1080 - 2.1MP (16:9 - Requires 12GB+ VRAM)"

            # Split by " - " and take the dimensions part (middle section)
            parts = resolution_preset.split(" - ")
            if len(parts) < 2:
                raise ValueError(f"Invalid resolution format: {resolution_preset}")

            # Handle new format "Name - WxH - MP info (notes)"
            dimensions_part = parts[1]  # e.g., "1024x1024", "1920x1080"

            # Extract just the WxH part (ignore any MP info or notes)
            if "x" not in dimensions_part:
                raise ValueError(f"No dimensions found in: {dimensions_part}")

            width_str, height_str = dimensions_part.split("x")

            target_width = int(width_str.strip())
            target_height = int(height_str.strip())

            # Ensure dimensions are multiples of 64 for FLUX compatibility
            target_width = round_to_nearest_multiple(target_width, 64)
            target_height = round_to_nearest_multiple(target_height, 64)

            print(f"[BawkSampler] Parsed resolution: {target_width}x{target_height} from preset '{resolution_preset}'")
            return target_width, target_height

        except Exception as e:
            print(f"[BawkSampler] Warning: Could not parse resolution '{resolution_preset}': {e}")
            # Fallback to FLUX optimal resolution (within official 2.0MP limit)
            print(f"[BawkSampler] Using fallback: 1024x1024 (FLUX optimal)")
            return 1024, 1024
    
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

    def _validate_parameters_with_feedback(
        self, model, batch_size, steps, guidance, max_shift, base_shift,
        resolution, use_custom_resolution, custom_width, custom_height, is_img2img=False, denoise=1.0
    ):
        """Smart validation with user-friendly feedback and recommendations"""

        # Validate batch size for performance
        if batch_size > 16:
            print(f"[BawkSampler] ⚠️  WARNING: Large batch size ({batch_size}) may cause memory issues. Consider reducing to 8-16 for better stability.")
        elif batch_size > 8:
            print(f"[BawkSampler] ℹ️  INFO: Batch size {batch_size} is large. Monitor memory usage during generation.")

        # Validate steps for quality/performance balance (based on community research)
        if steps < 20:
            print(f"[BawkSampler] ⚠️  WARNING: Low step count ({steps}) may result in poor quality. Community consensus: 25-30 steps optimal.")
        elif steps > 50:
            print(f"[BawkSampler] ℹ️  INFO: High step count ({steps}) will increase generation time. Sweet spot: 25-30 steps for quality/speed.")
        elif 25 <= steps <= 30:
            print(f"[BawkSampler] ✅ OPTIMAL: {steps} steps is in the community-tested sweet spot (25-30) for best quality/time balance.")
        elif 20 <= steps <= 35:
            print(f"[BawkSampler] ✅ GOOD: {steps} steps will produce high-quality results with reasonable generation time.")

        # Validate guidance scale for FLUX
        if guidance < 1.0:
            print(f"[BawkSampler] ⚠️  WARNING: Very low guidance ({guidance}) may ignore your prompt. Recommended: 3.0-7.0 for FLUX.")
        elif guidance > 10.0:
            print(f"[BawkSampler] ⚠️  WARNING: Very high guidance ({guidance}) may cause artifacts. Recommended: 3.0-7.0 for FLUX.")

        # Validate shift parameters for FLUX
        if max_shift > 1.5:
            print(f"[BawkSampler] ⚠️  WARNING: High max_shift ({max_shift}) may cause instability. Recommended: 0.3-1.0 for FLUX.")
        if base_shift > max_shift:
            print(f"[BawkSampler] ⚠️  WARNING: base_shift ({base_shift}) should be ≤ max_shift ({max_shift}). Consider adjusting values.")

        # Validate resolution against official FLUX limits and provide recommendations
        if use_custom_resolution:
            total_pixels = custom_width * custom_height
            megapixels = total_pixels / 1_000_000

            # Official FLUX stability zones
            if megapixels > 4.0:
                print(f"[BawkSampler] 🚨 EXTREME: Resolution ({custom_width}x{custom_height}, {megapixels:.1f}MP) far exceeds FLUX capabilities.")
                print(f"[BawkSampler] 💡 RECOMMENDATION: Use 1408x1408 (2.0MP official max) + Flux-dev-Upscaler for better results.")
            elif megapixels > 2.0:
                print(f"[BawkSampler] ⚠️  BEYOND OFFICIAL LIMIT: Resolution ({custom_width}x{custom_height}, {megapixels:.1f}MP) exceeds FLUX's 2.0MP limit.")
                print(f"[BawkSampler] ⚠️  WARNING: Expect instability, blurriness, or artifacts. Consider post-processing upscaler.")
                print(f"[BawkSampler] 💡 STABLE ALTERNATIVE: Use 1408x1408 (2.0MP) or 1024x1024 (1.0MP) for reliable results.")
            elif megapixels >= 1.8:
                print(f"[BawkSampler] ⚠️  NEAR LIMIT: Resolution ({custom_width}x{custom_height}, {megapixels:.1f}MP) approaches FLUX's 2.0MP stability limit.")
                print(f"[BawkSampler] ℹ️  INFO: Monitor for quality issues. 1024x1024 (1.0MP) is more reliable.")
            elif megapixels < 0.15:
                print(f"[BawkSampler] ℹ️  MINIMUM: Resolution ({custom_width}x{custom_height}, {megapixels:.1f}MP) near FLUX minimum (0.1MP).")
                print(f"[BawkSampler] ℹ️  INFO: Good for rapid prototyping but limited detail.")
            elif 0.8 <= megapixels <= 1.2:
                print(f"[BawkSampler] ✅ OPTIMAL ZONE: Resolution ({custom_width}x{custom_height}, {megapixels:.1f}MP) in FLUX sweet spot.")

            # Check if dimensions are 64-pixel aligned (critical for FLUX)
            if custom_width % 64 != 0 or custom_height % 64 != 0:
                print(f"[BawkSampler] ⚠️  WARNING: Dimensions not 64-pixel aligned. FLUX requires multiples of 64 for stability.")
                aligned_w = round_to_nearest_multiple(custom_width, 64)
                aligned_h = round_to_nearest_multiple(custom_height, 64)
                print(f"[BawkSampler] 💡 CORRECTION: Use {aligned_w}x{aligned_h} for proper FLUX compatibility.")
        else:
            # Provide specific advice for preset choices based on official FLUX limits
            if "⚠️ Extreme" in resolution:
                print(f"[BawkSampler] 🚨 EXTREME PRESET: This resolution exceeds FLUX capabilities and will likely produce poor results.")
                print(f"[BawkSampler] 💡 BETTER APPROACH: Generate at 1408x1408 and use Flux-dev-Upscaler for higher resolution.")
            elif "⚠️ Experimental" in resolution:
                print(f"[BawkSampler] ⚠️  EXPERIMENTAL: This resolution exceeds FLUX's 2.0MP official limit. Expect potential issues.")
                print(f"[BawkSampler] 💡 STABLE ALTERNATIVE: Consider 'FLUX Max Square' (1408x1408) for reliable results.")
            elif "Official maximum" in resolution:
                print(f"[BawkSampler] ✅ OFFICIAL MAX: Using FLUX's maximum stable resolution (2.0MP). Excellent quality expected.")
            elif "Near official limit" in resolution:
                print(f"[BawkSampler] ⚠️  NEAR LIMIT: Close to FLUX's stability boundary. Monitor for quality issues.")
            elif "Recommended" in resolution:
                print(f"[BawkSampler] ✅ OPTIMAL: 1024x1024 is the proven sweet spot for FLUX (15-30s on RTX 4090).")
            elif "Draft mode" in resolution:
                print(f"[BawkSampler] ✅ MINIMUM: Perfect for rapid prototyping (~5s generation, limited detail).")
            elif "0.8MP" in resolution and "Fast" in resolution:
                print(f"[BawkSampler] ✅ FAST & STABLE: Good balance for quick generation (1-2min on mid-range GPUs).")

        # Img2Img specific validation
        if is_img2img:
            if denoise == 1.0:
                print(f"[BawkSampler] ℹ️  IMG2IMG: Denoise at 1.0 will completely replace input image. Consider 0.6-0.9 for image modification.")
            elif denoise < 0.3:
                print(f"[BawkSampler] ℹ️  IMG2IMG: Very low denoise ({denoise}) may result in minimal changes to input image.")
            elif denoise > 0.9:
                print(f"[BawkSampler] ℹ️  IMG2IMG: High denoise ({denoise}) will heavily modify the input image.")
            else:
                print(f"[BawkSampler] ✅ IMG2IMG: Good denoise strength ({denoise}) for image modification.")

        # Memory estimation and recommendations
        self._estimate_memory_usage(batch_size, resolution, use_custom_resolution, custom_width, custom_height)

        mode_str = "img2img" if is_img2img else "text-to-image"
        print(f"[BawkSampler] ✅ Validation complete. Proceeding with {mode_str} generation...")

    def _estimate_memory_usage(self, batch_size, resolution, use_custom_resolution, custom_width, custom_height):
        """Estimate and report memory usage"""
        try:
            # Get dimensions
            if use_custom_resolution:
                width, height = custom_width, custom_height
            else:
                width, height = self._parse_resolution_preset(resolution)

            # FLUX latent calculations (16 channels, 8x downscale)
            latent_width = width // 8
            latent_height = height // 8

            # Estimate memory usage (rough calculation)
            # Latent: batch_size * 16 * latent_height * latent_width * 4 bytes (fp32)
            latent_memory_mb = (batch_size * 16 * latent_height * latent_width * 4) / (1024 * 1024)

            # Final image: batch_size * 3 * height * width * 4 bytes (fp32)
            image_memory_mb = (batch_size * 3 * height * width * 4) / (1024 * 1024)

            total_estimated_mb = latent_memory_mb + image_memory_mb

            if total_estimated_mb > 2048:  # > 2GB
                print(f"[BawkSampler] ⚠️  MEMORY: Estimated usage ~{total_estimated_mb:.0f}MB. Consider reducing batch size or resolution.")
            elif total_estimated_mb > 1024:  # > 1GB
                print(f"[BawkSampler] ℹ️  MEMORY: Estimated usage ~{total_estimated_mb:.0f}MB. Monitor VRAM during generation.")
            else:
                print(f"[BawkSampler] ✅ MEMORY: Estimated usage ~{total_estimated_mb:.0f}MB. Should run smoothly.")

        except Exception as e:
            print(f"[BawkSampler] Could not estimate memory usage: {str(e)}")

    def _get_error_with_solution(self, error_msg: str):
        """Generate helpful error messages with suggested solutions"""
        error_lower = error_msg.lower()

        # Memory-related errors
        if "out of memory" in error_lower or "cuda out of memory" in error_lower:
            return (
                "GPU memory exhausted during generation",
                "Try reducing batch size, using a smaller resolution preset, or closing other GPU applications"
            )

        # Model loading errors
        if "model" in error_lower and ("load" in error_lower or "file" in error_lower):
            return (
                "Failed to load model",
                "Ensure the model file exists and is a valid FLUX checkpoint. Check file permissions and disk space"
            )

        # Dimension/tensor errors
        if "size mismatch" in error_lower or "dimension" in error_lower:
            return (
                "Tensor dimension mismatch",
                "This usually indicates model incompatibility. Ensure you're using a FLUX model, not SD1.5/SDXL"
            )

        # CUDA/device errors
        if "cuda" in error_lower and "device" in error_lower:
            return (
                "GPU device error",
                "Check CUDA installation, update GPU drivers, or switch to CPU mode if GPU issues persist"
            )

        # Resolution parsing errors
        if "resolution" in error_lower or "parse" in error_lower:
            return (
                "Resolution parsing failed",
                "Use a preset resolution or ensure custom dimensions are multiples of 64"
            )

        # VAE decoding errors
        if "vae" in error_lower or "decode" in error_lower:
            return (
                "VAE decoding failed",
                "Ensure VAE is compatible with FLUX. Try using the built-in FLUX VAE or check VAE file integrity"
            )

        # Sampling errors
        if "sampl" in error_lower or "noise" in error_lower:
            return (
                "Sampling process failed",
                "Try different sampler/scheduler combinations, reduce steps, or adjust guidance scale (3.0-7.0 recommended)"
            )

        # File permission errors
        if "permission" in error_lower or "access" in error_lower:
            return (
                "File access permission denied",
                "Check file permissions, ensure ComfyUI has write access to output directory, or run as administrator"
            )

        # Network/download errors
        if "download" in error_lower or "network" in error_lower or "connection" in error_lower:
            return (
                "Network or download error",
                "Check internet connection, verify URLs, or try downloading models manually"
            )

        # Generic fallback
        return (error_msg, "Check the console for detailed error information and ensure all inputs are valid")

    def _encode_image_to_latent(self, vae, input_image, batch_size):
        """Encode input image to latent for img2img processing"""
        try:
            print(f"[BawkSampler] Encoding input image to latent space...")

            # Handle batch size adjustment
            if len(input_image.shape) == 4:  # Batch dimension exists
                image_batch_size = input_image.shape[0]
                if image_batch_size == 1 and batch_size > 1:
                    # Repeat single image for batch
                    input_image = input_image.repeat(batch_size, 1, 1, 1)
                    print(f"[BawkSampler] Expanded single input image to batch size {batch_size}")
                elif image_batch_size != batch_size:
                    # Use first image and repeat if needed
                    input_image = input_image[0:1].repeat(batch_size, 1, 1, 1)
                    print(f"[BawkSampler] Using first image from batch, expanded to batch size {batch_size}")

            # Encode image to latent using VAE
            latent_samples = vae.encode(input_image)

            # Create latent dictionary
            latent = {"samples": latent_samples}

            image_h, image_w = input_image.shape[-2:]
            latent_h, latent_w = latent_samples.shape[-2:]
            print(f"[BawkSampler] Encoded image {image_w}x{image_h} to latent {latent_w}x{latent_h}")

            return latent

        except Exception as e:
            error_msg = f"Failed to encode input image to latent: {str(e)}"
            print(f"[BawkSampler] Error: {error_msg}")
            raise RuntimeError(error_msg)