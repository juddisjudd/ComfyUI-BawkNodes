"""
BawkImageLoader - Enhanced Image Loading with Preprocessing
File: nodes/bawk_image_loader.py
"""

import os
import numpy as np
import torch
from PIL import Image, ImageOps, ExifTags
from typing import Tuple, Any

# ComfyUI imports with fallback
try:
    import folder_paths
    from comfy.utils import common_upsampling_factor
except ImportError:
    folder_paths = None
    common_upsampling_factor = None


class BawkImageLoader:
    """
    Enhanced image loader with file browser and preprocessing options.
    Compatible with ComfyUI's Load Image node but with additional features.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get available images from ComfyUI input directory
        input_dir = folder_paths.get_input_directory() if folder_paths else "input"
        files = []
        if os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif'))]

        return {
            "required": {
                "image": (sorted(files), {
                    "image_upload": True,
                    "tooltip": "Select image file or upload new one"
                }),
            },
            "optional": {
                "auto_orient": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically rotate image based on EXIF orientation data"
                }),
                "target_size": (["Original", "512", "768", "1024", "1536", "2048"], {
                    "default": "Original",
                    "tooltip": "Resize image to target size (maintains aspect ratio)"
                }),
                "resize_method": (["Lanczos", "Bilinear", "Bicubic", "Nearest"], {
                    "default": "Lanczos",
                    "tooltip": "Resampling method for resizing"
                }),
                "pad_to_square": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Pad image to square aspect ratio with black borders"
                }),
                "normalize_colors": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize color values to 0-1 range for better FLUX compatibility"
                }),
                "create_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Generate mask output (white = opaque, black = transparent)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "filename", "width", "height")
    FUNCTION = "load_image"
    CATEGORY = "BawkNodes/image"
    DESCRIPTION = "Enhanced image loader with file browser, preprocessing, and mask support"

    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        """Check if image file has changed"""
        if folder_paths:
            image_path = folder_paths.get_annotated_filepath(image)
            if os.path.exists(image_path):
                return os.path.getmtime(image_path)
        return float("inf")

    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        """Validate that the image file exists"""
        if not folder_paths:
            return True

        if not image:
            return "No image selected"

        image_path = folder_paths.get_annotated_filepath(image)
        if not os.path.exists(image_path):
            return f"Image file does not exist: {image}"

        return True

    def load_image(
        self,
        image,
        auto_orient=True,
        target_size="Original",
        resize_method="Lanczos",
        pad_to_square=False,
        normalize_colors=True,
        create_mask=False
    ):
        """
        Load and preprocess image with various options
        """
        try:
            print(f"[BawkImageLoader] Loading image: {image}")

            # Get full path using ComfyUI's folder_paths
            if folder_paths:
                image_path = folder_paths.get_annotated_filepath(image)
            else:
                image_path = image

            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")

            # Load image with PIL
            pil_image = Image.open(image_path)
            filename = os.path.basename(image_path)

            # Store original dimensions
            original_width, original_height = pil_image.size
            print(f"[BawkImageLoader] Original size: {original_width}x{original_height}")

            # Handle EXIF orientation
            if auto_orient:
                pil_image = self._auto_orient_image(pil_image)
                if pil_image.size != (original_width, original_height):
                    print(f"[BawkImageLoader] Auto-rotated image to: {pil_image.size[0]}x{pil_image.size[1]}")

            # Convert to RGB if needed
            if pil_image.mode not in ('RGB', 'RGBA'):
                print(f"[BawkImageLoader] Converting from {pil_image.mode} to RGB")
                pil_image = pil_image.convert('RGB')

            # Resize if requested
            if target_size != "Original":
                pil_image = self._resize_image(pil_image, int(target_size), resize_method)
                print(f"[BawkImageLoader] Resized to: {pil_image.size[0]}x{pil_image.size[1]}")

            # Pad to square if requested
            if pad_to_square:
                pil_image = self._pad_to_square(pil_image)
                print(f"[BawkImageLoader] Padded to square: {pil_image.size[0]}x{pil_image.size[1]}")

            # Convert to tensor
            image_tensor = self._pil_to_tensor(pil_image, normalize_colors)

            # Create mask if requested
            mask_tensor = self._create_mask_tensor(pil_image) if create_mask else torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)

            final_width, final_height = pil_image.size

            print(f"[BawkImageLoader] ✅ Successfully loaded {filename} ({final_width}x{final_height})")

            return (image_tensor, mask_tensor, filename, final_width, final_height)

        except Exception as e:
            error_msg = f"Failed to load image: {str(e)}"
            print(f"[BawkImageLoader] ❌ {error_msg}")

            # Return a small black image as fallback
            fallback_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            fallback_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (fallback_image, fallback_mask, f"Error: {error_msg}", 64, 64)

    def _auto_orient_image(self, image: Image.Image) -> Image.Image:
        """Auto-rotate image based on EXIF orientation"""
        try:
            # Get EXIF data
            exif = image._getexif()
            if exif is not None:
                # Find orientation tag
                for tag, value in exif.items():
                    if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'Orientation':
                        # Apply rotation based on orientation value
                        if value == 3:
                            image = image.rotate(180, expand=True)
                        elif value == 6:
                            image = image.rotate(270, expand=True)
                        elif value == 8:
                            image = image.rotate(90, expand=True)
                        break
        except (AttributeError, KeyError, TypeError):
            # No EXIF data or orientation tag, use ImageOps fallback
            try:
                image = ImageOps.exif_transpose(image)
            except Exception:
                pass  # Keep original orientation

        return image

    def _resize_image(self, image: Image.Image, target_size: int, method: str) -> Image.Image:
        """Resize image maintaining aspect ratio"""
        # Map method names to PIL constants
        method_map = {
            "Lanczos": Image.Resampling.LANCZOS,
            "Bilinear": Image.Resampling.BILINEAR,
            "Bicubic": Image.Resampling.BICUBIC,
            "Nearest": Image.Resampling.NEAREST
        }

        resample_method = method_map.get(method, Image.Resampling.LANCZOS)

        # Calculate new size maintaining aspect ratio
        width, height = image.size
        aspect_ratio = width / height

        if width > height:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)

        # Ensure dimensions are even numbers (better for some models)
        new_width = (new_width // 2) * 2
        new_height = (new_height // 2) * 2

        return image.resize((new_width, new_height), resample_method)

    def _pad_to_square(self, image: Image.Image) -> Image.Image:
        """Pad image to square with black borders"""
        width, height = image.size
        max_size = max(width, height)

        # Create new square image with black background
        square_image = Image.new('RGB', (max_size, max_size), (0, 0, 0))

        # Calculate position to center the original image
        x_offset = (max_size - width) // 2
        y_offset = (max_size - height) // 2

        # Paste original image onto square background
        square_image.paste(image, (x_offset, y_offset))

        return square_image

    def _pil_to_tensor(self, image: Image.Image, normalize: bool = True) -> torch.Tensor:
        """Convert PIL image to tensor format expected by ComfyUI"""
        # Convert to numpy array
        image_np = np.array(image)

        # Normalize to 0-1 range if requested
        if normalize:
            image_np = image_np.astype(np.float32) / 255.0
        else:
            image_np = image_np.astype(np.float32)

        # Convert to tensor and add batch dimension [batch, height, width, channels]
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)

        return image_tensor

    def _create_mask_tensor(self, image: Image.Image) -> torch.Tensor:
        """Create mask tensor from image alpha channel or full white mask"""
        if image.mode == 'RGBA':
            # Use alpha channel as mask
            alpha = image.split()[-1]  # Get alpha channel
            mask_np = np.array(alpha).astype(np.float32) / 255.0
        else:
            # Create full white mask (opaque)
            mask_np = np.ones((image.size[1], image.size[0]), dtype=np.float32)

        # Convert to tensor [batch, height, width]
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        return mask_tensor

    def _get_supported_formats(self):
        """Get list of supported image formats"""
        return ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif']