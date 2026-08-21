"""
FluxImageSaver - organized saving with embedded workflow metadata and optional Discord webhook.
"""

import json
import logging
import os
import re
from datetime import datetime

import numpy as np
import requests
import folder_paths
from comfy.cli_args import args
from PIL import Image
from PIL.PngImagePlugin import PngInfo

CONTENT_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp"}


def _clean_name(value):
    value = re.sub(r"\.[^.]*$", "", value)
    value = re.sub(r'[<>:"/\\|?*]', "_", value)
    return re.sub(r"_{2,}", "_", value).strip("_")


def _unique_path(directory, stem, ext):
    path = os.path.join(directory, f"{stem}.{ext}")
    counter = 1
    while os.path.exists(path):
        path = os.path.join(directory, f"{stem}_{counter:03d}.{ext}")
        counter += 1
    return path


class FluxImageSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prefix": ("STRING", {"default": "flux_image", "dynamicPrompts": False}),
                "format": (["png", "jpg", "webp"], {"default": "png"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "tooltip": "jpg / webp quality"}),
                "save_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Embed the workflow in PNG files (drag back into ComfyUI). jpg / webp get a .json sidecar",
                }),
                "save_prompt": ("BOOLEAN", {"default": True, "tooltip": "Also write the final prompt to a .txt file"}),
                "send_to_discord": ("BOOLEAN", {"default": False, "tooltip": "Post the images to a Discord webhook"}),
                "discord_webhook_url": ("STRING", {"default": "", "dynamicPrompts": False}),
            },
            "optional": {
                "model_string": ("STRING", {
                    "forceInput": True, "tooltip": "MODEL_STRING from Bawk Model Loader (used for the folder name)",
                }),
                "processed_prompt": ("STRING", {"forceInput": True, "tooltip": "PROMPT_OUT from Bawk Wildcard Encoder"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    CATEGORY = "BawkNodes/image"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Save images into [model]-date folders with embedded workflow metadata, prompt file and optional Discord upload."
    )

    def save_images(self, images, prefix="flux_image", format="png", quality=95, save_metadata=True, save_prompt=True,
                    send_to_discord=False, discord_webhook_url="", model_string="", processed_prompt="",
                    prompt=None, extra_pnginfo=None):
        now = datetime.now()
        date_str = now.strftime("%d-%m-%Y")
        stamp = now.strftime("%d-%m-%Y_%H-%M-%S")

        folder_name = f"[{_clean_name(model_string)}]-{date_str}" if model_string else date_str
        output_dir = os.path.join(folder_paths.get_output_directory(), folder_name)
        os.makedirs(output_dir, exist_ok=True)
        prefix = _clean_name(prefix) or "flux_image"

        if save_prompt and processed_prompt:
            with open(_unique_path(output_dir, f"{prefix}_{stamp}_prompt", "txt"), "w", encoding="utf-8") as f:
                f.write(processed_prompt)

        pnginfo = None
        sidecar = None
        if save_metadata and not args.disable_metadata:
            if format == "png":
                pnginfo = PngInfo()
                if prompt is not None:
                    pnginfo.add_text("prompt", json.dumps(prompt))
                for key, value in (extra_pnginfo or {}).items():
                    pnginfo.add_text(key, json.dumps(value))
                if processed_prompt:
                    pnginfo.add_text("bawk_prompt", processed_prompt)
                if model_string:
                    pnginfo.add_text("bawk_model", model_string)
            else:
                sidecar = {
                    "timestamp": now.isoformat(),
                    "model": model_string,
                    "processed_prompt": processed_prompt,
                    "prompt": prompt,
                    "extra_pnginfo": extra_pnginfo,
                }

        saved_paths = []
        results = []
        for index, image in enumerate(images):
            pil_image = Image.fromarray(np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8))
            suffix = f"_{index + 1:03d}" if len(images) > 1 else ""
            path = _unique_path(output_dir, f"{prefix}_{stamp}{suffix}", format)

            if format == "png":
                pil_image.save(path, pnginfo=pnginfo, compress_level=4)
            elif format == "jpg":
                pil_image.save(path, quality=quality, optimize=True)
            else:
                pil_image.save(path, quality=quality, method=6)

            if sidecar is not None:
                with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
                    json.dump(sidecar, f, indent=2, ensure_ascii=False)

            saved_paths.append(path)
            results.append({"filename": os.path.basename(path), "subfolder": folder_name, "type": "output"})

        if send_to_discord and discord_webhook_url.strip():
            try:
                self._send_to_discord(saved_paths, discord_webhook_url.strip(), processed_prompt, model_string)
            except Exception as e:
                logging.warning(f"[FluxImageSaver] Discord webhook failed: {e}")

        return {"ui": {"images": results}}

    def _send_to_discord(self, paths, webhook_url, prompt, model):
        description = f"**Model:** {model or 'unknown'}\n**Prompt:** {prompt[:1000]}{'...' if len(prompt) > 1000 else ''}"
        embed = {
            "title": "New generation",
            "description": description,
            "color": 0x5865F2,
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": "ComfyUI BawkNodes"},
        }
        for batch_start in range(0, len(paths), 10):  # Discord allows 10 files per message
            batch = paths[batch_start:batch_start + 10]
            payload = {"embeds": [embed]} if batch_start == 0 else {"content": f"Batch {batch_start // 10 + 1}"}
            files = {}
            for i, path in enumerate(batch):
                with open(path, "rb") as f:
                    content_type = CONTENT_TYPES.get(os.path.splitext(path)[1], "image/png")
                    files[f"file{i}"] = (os.path.basename(path), f.read(), content_type)
            response = requests.post(webhook_url, data={"payload_json": json.dumps(payload)}, files=files, timeout=30)
            if not response.ok:
                logging.warning(f"[FluxImageSaver] Discord returned {response.status_code}: {response.text}")
