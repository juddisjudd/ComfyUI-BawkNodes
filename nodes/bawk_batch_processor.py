"""
BawkBatchProcessor - pull prompt + settings for one row of a CSV / JSON batch file.
"""

import csv
import json
import os

PROMPT_KEYS = ("prompt", "text", "description")
SEED_KEYS = ("seed", "noise_seed")
STEPS_KEYS = ("steps", "sampling_steps")
GUIDANCE_KEYS = ("guidance", "guidance_scale", "cfg")
RESOLUTION_KEYS = ("resolution", "size", "preset")


def _first(item, keys, cast, default):
    for key in keys:
        if key in item and item[key] not in (None, ""):
            try:
                return cast(item[key])
            except (TypeError, ValueError):
                continue
    return default


def _load_rows(path, file_format):
    if file_format == "Auto-detect":
        file_format = "JSON" if path.lower().endswith(".json") else "CSV"
    with open(path, "r", encoding="utf-8-sig") as f:
        if file_format == "JSON":
            data = json.load(f)
            if isinstance(data, dict):
                for key in ("items", "prompts", "batch", "data"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
                else:
                    data = [data]
            return [row for row in data if isinstance(row, dict)]
        rows = []
        for row in csv.DictReader(f):
            clean = {k.strip(): v.strip() for k, v in row.items() if k and isinstance(v, str)}
            if clean:
                rows.append(clean)
        return rows


class BawkBatchProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_file": ("STRING", {
                    "default": "",
                    "dynamicPrompts": False,
                    "tooltip": "Path to a CSV or JSON file. Columns / keys: prompt, seed, steps, guidance, resolution",
                }),
                "file_format": (["Auto-detect", "CSV", "JSON"], {"default": "Auto-detect"}),
                "batch_index": ("INT", {
                    "default": 0, "min": 0, "max": 999999, "control_after_generate": True,
                    "tooltip": "Row to use (0-based). Set control to 'increment' to walk through the file",
                }),
                "preview_only": ("BOOLEAN", {"default": False, "tooltip": "Return a text preview of the file instead of a row"}),
            },
            "optional": {
                "default_resolution": ("STRING", {"default": "1024x1024", "dynamicPrompts": False}),
                "default_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "default_guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "seed", "steps", "guidance", "resolution", "batch_info")
    FUNCTION = "process_batch"
    CATEGORY = "BawkNodes/batch"
    DESCRIPTION = "Read one row of a CSV / JSON batch file. Link 'resolution' into Bawk Sampler's resolution input."

    def process_batch(self, batch_file, file_format="Auto-detect", batch_index=0, preview_only=False,
                      default_resolution="1024x1024", default_steps=20, default_guidance=3.5):
        if not batch_file or not os.path.isfile(batch_file):
            raise FileNotFoundError(f"Batch file not found: {batch_file}")
        rows = _load_rows(batch_file, file_format)
        if not rows:
            raise ValueError(f"No rows found in {batch_file}")

        if preview_only:
            lines = [f"{len(rows)} items in {os.path.basename(batch_file)}", ""]
            for i, row in enumerate(rows[:5]):
                lines.append(f"[{i}] " + ", ".join(f"{k}={str(v)[:80]}" for k, v in row.items()))
            if len(rows) > 5:
                lines.append(f"... and {len(rows) - 5} more")
            return ("\n".join(lines), 0, default_steps, default_guidance, default_resolution, "preview")

        if batch_index >= len(rows):
            raise IndexError(f"batch_index {batch_index} out of range, file has {len(rows)} rows")
        row = rows[batch_index]

        prompt = _first(row, PROMPT_KEYS, str, "")
        if not prompt:
            raise ValueError(f"Row {batch_index} has no prompt (looked for {PROMPT_KEYS})")
        seed = _first(row, SEED_KEYS, lambda v: int(float(v)), 0)
        steps = _first(row, STEPS_KEYS, lambda v: int(float(v)), default_steps)
        guidance = _first(row, GUIDANCE_KEYS, float, default_guidance)
        resolution = _first(row, RESOLUTION_KEYS, str, default_resolution)
        info = f"Row {batch_index + 1}/{len(rows)} from {os.path.basename(batch_file)}"
        return (prompt, seed, steps, guidance, resolution, info)
