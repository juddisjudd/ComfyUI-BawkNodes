# ComfyUI-BawkNodes

Custom node pack for ComfyUI focused only on FLUX models (FLUX.1 dev / Krea / Kontext, FLUX.2 dev / klein, GGUF builds of those). Goal: the fewest nodes needed for multi-LoRA FLUX workflows. FLUX has no negative prompt, so no node exposes one.

## Layout

- `__init__.py` exports `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`, `WEB_DIRECTORY`.
- `nodes/` one file per node (V1 class schema: `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`, `DESCRIPTION`).
- `nodes/lora_stack.py` also holds the shared LoRA helpers (`lora_slot_inputs`, `collect_lora_slots`, `apply_lora_stack`).
- `js/bawk_nodes.js` node colors and prompt textarea sizing only.
- `workflows/` example workflow.

## Nodes

| Node id | Purpose |
| --- | --- |
| `DiffusionModelLoader` | model + VAE + text encoder(s). 2 encoders = FLUX.1 (`CLIPType.FLUX`), 1 encoder = FLUX.2 (`CLIPType.FLUX2`). `.gguf` files go through ComfyUI-GGUF's node classes looked up in `nodes.NODE_CLASS_MAPPINGS` at run time. |
| `BawkLoraStack` | 6 LoRA slots, chainable (`lora_stack` in/out), `enabled` master switch for set switching. |
| `FluxWildcardEncode` | applies `lora_stack` + 6 inline slots, resolves `{a|b}` wildcards, encodes with `encode_from_tokens_scheduled`. |
| `BawkSampler` | builds latent from `model.get_model_object("latent_format")` (16ch/8 for FLUX.1, 128ch/16 for FLUX.2), sets `guidance` in conditioning, optional `reference_latents` for Kontext / klein edit, `cfg` > 1 uses `CFGGuider` with zeroed negative (klein base only), FLUX.1 gets the resolution-dependent shift patch, FLUX.2 uses its own schedule when scheduler is `auto`. |
| `FluxImageSaver` | `[model]-DD-MM-YYYY/` folders, PNG metadata written with `json.dumps` like core `SaveImage`, optional Discord webhook. |
| `BawkImageLoader` | Load Image + resize / pad + size outputs. |
| `BawkBatchProcessor` | one row of a CSV / JSON file -> prompt, seed, steps, guidance, resolution. |

## Conventions

- Only add inputs that matter for FLUX. Defaults follow the official Comfy templates (euler, 20 steps, guidance 3.5 / 2.5 Kontext / 4.0 FLUX.2, klein distilled 4 steps).
- Raise exceptions on failure; do not return fake fallback tensors.
- Use `logging`, not `print`. No startup banners.
- Keep comments to non-obvious constraints only.
- No `eval` / `exec`, no runtime `pip install` (registry rules).
- When changing widget inputs, update `workflows/example_bawknodes_workflow.json` `widgets_values` to match the new order.
