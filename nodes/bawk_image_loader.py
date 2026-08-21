"""
BawkImageLoader - Load Image with optional resize / pad-to-square and size outputs.
"""

import hashlib
import os

import numpy as np
import torch
import folder_paths
import node_helpers
from PIL import Image, ImageOps

RESAMPLE = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST,
}


class BawkImageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
            "optional": {
                "target_size": (["Original", "512", "768", "1024", "1536", "2048"], {
                    "default": "Original",
                    "tooltip": "Resize so the longest side matches this (keeps aspect ratio)",
                }),
                "resize_method": (list(RESAMPLE), {"default": "lanczos"}),
                "pad_to_square": ("BOOLEAN", {"default": False, "tooltip": "Pad with black to a square canvas"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "filename", "width", "height")
    FUNCTION = "load_image"
    CATEGORY = "BawkNodes/image"
    DESCRIPTION = "Load Image with EXIF auto-rotation, optional resize / pad-to-square, alpha mask and size outputs."

    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        path = folder_paths.get_annotated_filepath(image)
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            digest.update(f.read())
        return digest.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True

    def load_image(self, image, target_size="Original", resize_method="lanczos", pad_to_square=False):
        path = folder_paths.get_annotated_filepath(image)
        pil_image = node_helpers.pillow(Image.open, path)
        pil_image = node_helpers.pillow(ImageOps.exif_transpose, pil_image)
        if pil_image.mode == "I":
            pil_image = pil_image.point(lambda i: i * (1 / 255))

        alpha = pil_image.getchannel("A") if "A" in pil_image.getbands() else None
        rgb = pil_image.convert("RGB")

        if target_size != "Original":
            size = int(target_size)
            w, h = rgb.size
            scale = size / max(w, h)
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            rgb = rgb.resize(new_size, RESAMPLE[resize_method])
            if alpha is not None:
                alpha = alpha.resize(new_size, Image.Resampling.NEAREST)

        if pad_to_square:
            side = max(rgb.size)
            rgb = ImageOps.pad(rgb, (side, side), color=(0, 0, 0))
            if alpha is not None:
                alpha = ImageOps.pad(alpha, (side, side), color=255)

        image_tensor = torch.from_numpy(np.array(rgb).astype(np.float32) / 255.0).unsqueeze(0)
        if alpha is not None:
            mask = 1.0 - torch.from_numpy(np.array(alpha).astype(np.float32) / 255.0)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32)

        width, height = rgb.size
        return (image_tensor, mask.unsqueeze(0), os.path.basename(path), width, height)
