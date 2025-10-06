"""
BawkBatchProcessor - Process multiple prompts from CSV/JSON files
File: nodes/bawk_batch_processor.py
"""

import os
import csv
import json
from typing import List, Dict, Any, Tuple

# ComfyUI imports with fallback
try:
    import folder_paths
except ImportError:
    folder_paths = None

# Optional pandas import for advanced CSV handling
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class BawkBatchProcessor:
    """
    Process multiple prompts and settings from CSV or JSON files.
    Supports batch generation with different parameters per image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_file": ("STRING", {
                    "default": "",
                    "tooltip": "Path to CSV or JSON file with batch settings. CSV columns: prompt, seed, steps, guidance, etc."
                }),
                "file_format": (["Auto-detect", "CSV", "JSON"], {
                    "default": "Auto-detect",
                    "tooltip": "File format - Auto-detect will guess from extension"
                }),
                "batch_index": ("INT", {
                    "default": 0, "min": 0, "max": 999999,
                    "tooltip": "Which row/item to process from the batch file (0-based index)"
                }),
                "preview_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show file contents without processing - useful for checking batch files"
                }),
            },
            "optional": {
                # Override defaults
                "default_resolution": ("STRING", {
                    "default": "FHD 16:9 - 1920x1080 - 2.1MP",
                    "tooltip": "Default resolution for items that don't specify one"
                }),
                "default_steps": ("INT", {
                    "default": 30, "min": 1, "max": 100,
                    "tooltip": "Default steps for items that don't specify them"
                }),
                "default_guidance": ("FLOAT", {
                    "default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Default guidance for items that don't specify it"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "seed", "steps", "guidance", "resolution", "batch_info")
    FUNCTION = "process_batch"
    CATEGORY = "BawkNodes/batch"
    DESCRIPTION = "Process prompts and settings from CSV/JSON files for batch generation"

    def process_batch(
        self,
        batch_file,
        file_format="Auto-detect",
        batch_index=0,
        preview_only=False,
        default_resolution="FHD 16:9 - 1920x1080 - 2.1MP",
        default_steps=30,
        default_guidance=3.5
    ):
        """
        Process a batch file and return settings for the specified index
        """
        try:
            print(f"[BawkBatchProcessor] Processing batch file: {batch_file}")

            if not batch_file or not os.path.exists(batch_file):
                raise ValueError(f"Batch file not found: {batch_file}")

            # Load batch data
            batch_data = self._load_batch_file(batch_file, file_format)

            if preview_only:
                return self._preview_batch_file(batch_data)

            # Validate index
            if batch_index >= len(batch_data):
                raise ValueError(f"Batch index {batch_index} out of range. File contains {len(batch_data)} items.")

            # Get the specific batch item
            batch_item = batch_data[batch_index]

            # Extract settings with defaults
            settings = self._extract_settings(
                batch_item, default_resolution, default_steps, default_guidance
            )

            batch_info = f"Processing item {batch_index + 1}/{len(batch_data)} from {os.path.basename(batch_file)}"
            print(f"[BawkBatchProcessor] {batch_info}")

            return (
                settings["prompt"],
                settings["seed"],
                settings["steps"],
                settings["guidance"],
                settings["resolution"],
                batch_info
            )

        except Exception as e:
            error_msg = f"Batch processing failed: {str(e)}"
            print(f"[BawkBatchProcessor] Error: {error_msg}")

            # Return safe defaults on error
            return (
                f"Error: {error_msg}",
                0,
                default_steps,
                default_guidance,
                default_resolution,
                f"Error processing batch file"
            )

    def _load_batch_file(self, file_path: str, file_format: str) -> List[Dict]:
        """Load batch data from CSV or JSON file"""

        # Auto-detect format
        if file_format == "Auto-detect":
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv':
                file_format = "CSV"
            elif ext == '.json':
                file_format = "JSON"
            else:
                raise ValueError(f"Cannot auto-detect format for extension: {ext}")

        if file_format == "CSV":
            return self._load_csv_file(file_path)
        elif file_format == "JSON":
            return self._load_json_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def _load_csv_file(self, file_path: str) -> List[Dict]:
        """Load batch data from CSV file"""
        try:
            batch_data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Clean up the row data
                    clean_row = {k.strip(): v.strip() if isinstance(v, str) else v
                               for k, v in row.items() if k and k.strip()}
                    if clean_row:  # Only add non-empty rows
                        batch_data.append(clean_row)

            print(f"[BawkBatchProcessor] Loaded {len(batch_data)} items from CSV")
            return batch_data

        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {str(e)}")

    def _load_json_file(self, file_path: str) -> List[Dict]:
        """Load batch data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                batch_data = data
            elif isinstance(data, dict):
                # Try common keys for batch data
                for key in ['items', 'prompts', 'batch', 'data']:
                    if key in data and isinstance(data[key], list):
                        batch_data = data[key]
                        break
                else:
                    # Single item wrapped in dict
                    batch_data = [data]
            else:
                raise ValueError("JSON must contain a list or dict with batch data")

            print(f"[BawkBatchProcessor] Loaded {len(batch_data)} items from JSON")
            return batch_data

        except Exception as e:
            raise ValueError(f"Failed to load JSON file: {str(e)}")

    def _extract_settings(
        self, batch_item: Dict, default_resolution: str, default_steps: int, default_guidance: float
    ) -> Dict[str, Any]:
        """Extract and validate settings from a batch item"""

        settings = {}

        # Extract prompt (required)
        prompt_keys = ['prompt', 'text', 'description', 'input']
        settings['prompt'] = ""
        for key in prompt_keys:
            if key in batch_item and batch_item[key]:
                settings['prompt'] = str(batch_item[key])
                break

        if not settings['prompt']:
            raise ValueError(f"No prompt found in batch item. Looked for keys: {prompt_keys}")

        # Extract seed
        seed_keys = ['seed', 'random_seed', 'noise_seed']
        settings['seed'] = 0
        for key in seed_keys:
            if key in batch_item:
                try:
                    settings['seed'] = int(float(batch_item[key]))
                    break
                except (ValueError, TypeError):
                    continue

        # Extract steps
        steps_keys = ['steps', 'sampling_steps', 'iterations']
        settings['steps'] = default_steps
        for key in steps_keys:
            if key in batch_item:
                try:
                    steps = int(float(batch_item[key]))
                    if 1 <= steps <= 100:
                        settings['steps'] = steps
                        break
                except (ValueError, TypeError):
                    continue

        # Extract guidance
        guidance_keys = ['guidance', 'guidance_scale', 'cfg', 'cfg_scale']
        settings['guidance'] = default_guidance
        for key in guidance_keys:
            if key in batch_item:
                try:
                    guidance = float(batch_item[key])
                    if 0.0 <= guidance <= 20.0:
                        settings['guidance'] = guidance
                        break
                except (ValueError, TypeError):
                    continue

        # Extract resolution
        resolution_keys = ['resolution', 'size', 'dimensions', 'preset']
        settings['resolution'] = default_resolution
        for key in resolution_keys:
            if key in batch_item and batch_item[key]:
                settings['resolution'] = str(batch_item[key])
                break

        return settings

    def _preview_batch_file(self, batch_data: List[Dict]) -> Tuple[str, int, int, float, str, str]:
        """Preview the contents of a batch file"""

        preview_info = []
        preview_info.append(f"📁 Batch File Preview - {len(batch_data)} items total")
        preview_info.append("")

        # Show first few items
        preview_count = min(5, len(batch_data))
        for i, item in enumerate(batch_data[:preview_count]):
            preview_info.append(f"🔹 Item {i + 1}:")

            # Show key fields
            if 'prompt' in item:
                prompt_preview = item['prompt'][:100] + "..." if len(item['prompt']) > 100 else item['prompt']
                preview_info.append(f"   Prompt: {prompt_preview}")

            for key in ['seed', 'steps', 'guidance', 'resolution']:
                if key in item:
                    preview_info.append(f"   {key.title()}: {item[key]}")

            preview_info.append("")

        if len(batch_data) > preview_count:
            preview_info.append(f"... and {len(batch_data) - preview_count} more items")

        preview_text = "\\n".join(preview_info)
        batch_info = f"Preview mode: {len(batch_data)} items loaded"

        return (preview_text, 0, 30, 3.5, "FHD 16:9 - 1920x1080 - 2.1MP", batch_info)