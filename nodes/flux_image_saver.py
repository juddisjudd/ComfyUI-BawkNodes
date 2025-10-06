"""
FluxImageSaver Node
File: nodes/flux_image_saver.py
"""

import os
import json
import re
import numpy as np
import folder_paths
import requests
import io
import base64
from datetime import datetime
from typing import Dict, Any, List


class FluxImageSaver:
    """
    FLUX-optimized image saver with organized folder structure and metadata support
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_string": ("STRING", {"forceInput": True}),
                "processed_prompt": ("STRING", {"forceInput": True}),  # Connect from wildcard encoder
                "prefix": ("STRING", {"default": "flux_image"}),
                "format": (["png", "jpg", "webp"], {"default": "png"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "save_metadata": ("BOOLEAN", {"default": True}),
                "save_prompt": ("BOOLEAN", {"default": True, "tooltip": "Save the processed prompt as a separate text file"}),
                "send_to_discord": ("BOOLEAN", {"default": False, "tooltip": "Send images to Discord via webhook"}),
                "discord_webhook_url": ("STRING", {"default": "", "tooltip": "Discord webhook URL (only used when send_to_discord is True)"}),
            },
            "hidden": {
                "prompt_hidden": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_images"
    CATEGORY = "BawkNodes/image"
    OUTPUT_NODE = True
    DESCRIPTION = "Save FLUX-generated images with organized folder structure and optional Discord webhook integration"
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
    
    def save_images(self, images, model_string, processed_prompt, prefix="flux_image",
                   format="png", quality=95, save_metadata=True, save_prompt=True,
                   send_to_discord=False, discord_webhook_url="",
                   prompt_hidden=None, extra_pnginfo=None):
        """
        Save images with FLUX-optimized organization and metadata
        """
        try:
            from PIL import Image
            
            print(f"[FluxImageSaver] Processing {len(images)} images")
            
            # Get current date for folder and filename
            now = datetime.now()
            date_str = now.strftime("%d-%m-%Y")
            datetime_str = now.strftime("%d-%m-%Y_%H-%M-%S")
            
            # Clean model string for folder name
            clean_model = self._clean_filename(model_string)
            
            # Create folder structure: [MODEL]-DD-MM-YYYY
            folder_name = f"[{clean_model}]-{date_str}"
            output_dir = os.path.join(folder_paths.get_output_directory(), folder_name)
            os.makedirs(output_dir, exist_ok=True)
            
            saved_paths = []
            results = []  # For UI image preview
            
            # Save the processed prompt as a text file if enabled
            if save_prompt and processed_prompt:
                self._save_prompt_file(output_dir, processed_prompt, datetime_str, prefix)
            
            for i, image_tensor in enumerate(images):
                print(f"[FluxImageSaver] Processing image {i+1}/{len(images)}, shape: {image_tensor.shape}")
                
                # Convert tensor to PIL Image
                image_np = (255.0 * image_tensor.cpu().numpy()).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
                
                # Generate filename: prefix_DD-MM-YYYY_HH-MM-SS_001.ext
                batch_suffix = f"_{i+1:03d}" if len(images) > 1 else ""
                filename = f"{prefix}_{datetime_str}{batch_suffix}.{format}"
                filepath = os.path.join(output_dir, filename)
                
                # Check if file exists and add counter if necessary
                counter = 1
                original_filepath = filepath
                while os.path.exists(filepath):
                    name_part = f"{prefix}_{datetime_str}{batch_suffix}_{counter:03d}"
                    filename = f"{name_part}.{format}"
                    filepath = os.path.join(output_dir, filename)
                    counter += 1
                
                # Save image with format-specific options
                save_kwargs = {}
                if format.lower() == "jpg" or format.lower() == "jpeg":
                    save_kwargs = {"quality": quality, "optimize": True}
                elif format.lower() == "webp":
                    save_kwargs = {"quality": quality, "method": 6}
                elif format.lower() == "png":
                    save_kwargs = {"optimize": True}
                    # Add metadata to PNG
                    if save_metadata and extra_pnginfo:
                        from PIL.PngImagePlugin import PngInfo
                        metadata = PngInfo()
                        for key, value in extra_pnginfo.items():
                            metadata.add_text(key, str(value))
                        save_kwargs["pnginfo"] = metadata
                
                pil_image.save(filepath, format.upper(), **save_kwargs)
                saved_paths.append(filepath)
                
                # Add to results for UI preview (relative to output directory)
                results.append({
                    "filename": filename,
                    "subfolder": folder_name,
                    "type": self.type
                })
                
                print(f"[FluxImageSaver] Saved: {filepath}")
                
                # Save metadata/prompt file if requested
                if save_metadata and prompt_hidden:
                    self._save_metadata(filepath, prompt_hidden, extra_pnginfo, 
                                      model_string, format, quality, processed_prompt)
            
            print(f"[FluxImageSaver] Returning {len(results)} images for preview")

            # Send to Discord if enabled
            if send_to_discord and discord_webhook_url.strip():
                try:
                    self._send_to_discord(saved_paths, discord_webhook_url, processed_prompt, model_string)
                except Exception as e:
                    print(f"[FluxImageSaver] Discord webhook failed: {str(e)}")

            return {"ui": {"images": results}}
            
        except Exception as e:
            error_msg = f"Failed to save images: {str(e)}"
            print(f"[FluxImageSaver] Error: {error_msg}")
            import traceback
            traceback.print_exc()
            return {"ui": {"images": []}}
    
    def _clean_filename(self, filename):
        """Clean filename for use in folder/file names"""
        # Remove file extensions and clean up
        clean = re.sub(r'\.[^.]*$', '', filename)  # Remove extension
        clean = re.sub(r'[<>:"/\\|?*]', '_', clean)  # Replace invalid chars
        clean = re.sub(r'_{2,}', '_', clean)  # Collapse multiple underscores
        return clean.strip('_')
    
    def _save_prompt_file(self, output_dir, processed_prompt, datetime_str, prefix):
        """Save the processed prompt as a text file"""
        try:
            # Create prompt filename
            prompt_filename = f"{prefix}_{datetime_str}_prompt.txt"
            prompt_filepath = os.path.join(output_dir, prompt_filename)
            
            # Check if file exists and add counter if necessary
            counter = 1
            while os.path.exists(prompt_filepath):
                prompt_filename = f"{prefix}_{datetime_str}_prompt_{counter:03d}.txt"
                prompt_filepath = os.path.join(output_dir, prompt_filename)
                counter += 1
            
            # Save the prompt
            with open(prompt_filepath, 'w', encoding='utf-8') as f:
                f.write(processed_prompt)
            
            print(f"[FluxImageSaver] Saved prompt: {prompt_filepath}")
            
        except Exception as e:
            print(f"[FluxImageSaver] Failed to save prompt file: {str(e)}")
    
    def _save_metadata(self, image_path, prompt_hidden, extra_pnginfo, 
                      model_string, format, quality, processed_prompt=None):
        """Save metadata as JSON file"""
        try:
            # Create metadata filename
            base_path = os.path.splitext(image_path)[0]
            metadata_path = f"{base_path}_metadata.json"
            
            # Collect metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "model": model_string,
                "processed_prompt": processed_prompt,  # Add the processed prompt
                "format": format,
                "quality": quality if format != "png" else None,
                "workflow_prompt": prompt_hidden,
                "extra_info": extra_pnginfo
            }
            
            # Save as JSON
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"[FluxImageSaver] Metadata saved: {metadata_path}")
            
        except Exception as e:
            print(f"[FluxImageSaver] Failed to save metadata: {str(e)}")

    def _send_to_discord(self, image_paths: List[str], webhook_url: str, prompt: str, model: str):
        """Send images to Discord via webhook with batch support"""
        if not webhook_url or not image_paths:
            return

        print(f"[FluxImageSaver] Sending {len(image_paths)} images to Discord")

        # Prepare the embed with generation info
        embed = {
            "title": "🎨 New AI Generation",
            "description": f"**Model:** {model}\n**Prompt:** {prompt[:1000]}{'...' if len(prompt) > 1000 else ''}",
            "color": 0x5865F2,  # Discord blurple
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": "Generated with ComfyUI BawkNodes"}
        }

        # Discord has a limit of 10 files per message, so we'll batch them
        max_files_per_message = 10
        for i in range(0, len(image_paths), max_files_per_message):
            batch_paths = image_paths[i:i + max_files_per_message]
            self._send_discord_batch(webhook_url, batch_paths, embed if i == 0 else None, i // max_files_per_message + 1)

    def _send_discord_batch(self, webhook_url: str, image_paths: List[str], embed: dict = None, batch_num: int = 1):
        """Send a batch of images to Discord"""
        try:
            files = {}
            payload = {}

            # Add embed only to first batch
            if embed:
                payload["embeds"] = [embed]

            # Add batch info if multiple batches
            if batch_num > 1:
                payload["content"] = f"📎 Batch {batch_num} of images"

            # Prepare files for upload
            for idx, image_path in enumerate(image_paths):
                if not os.path.exists(image_path):
                    print(f"[FluxImageSaver] Warning: Image file not found: {image_path}")
                    continue

                try:
                    with open(image_path, 'rb') as f:
                        file_data = f.read()

                    filename = os.path.basename(image_path)
                    files[f'file{idx}'] = (filename, file_data, self._get_content_type(image_path))

                except Exception as e:
                    print(f"[FluxImageSaver] Failed to read image {image_path}: {str(e)}")
                    continue

            if not files:
                print(f"[FluxImageSaver] No valid images to send in batch {batch_num}")
                return

            # Send to Discord
            response = requests.post(
                webhook_url,
                data={"payload_json": json.dumps(payload)} if payload else None,
                files=files,
                timeout=30
            )

            if response.status_code == 200:
                print(f"[FluxImageSaver] Successfully sent batch {batch_num} with {len(files)} images to Discord")
            else:
                print(f"[FluxImageSaver] Discord webhook failed for batch {batch_num}: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"[FluxImageSaver] Network error sending to Discord: {str(e)}")
        except Exception as e:
            print(f"[FluxImageSaver] Unexpected error sending to Discord: {str(e)}")

    def _get_content_type(self, image_path: str) -> str:
        """Get the appropriate content type for the image"""
        ext = os.path.splitext(image_path)[1].lower()
        content_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
            '.gif': 'image/gif'
        }
        return content_types.get(ext, 'image/png')