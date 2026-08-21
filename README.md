<div align="center">
<h1>ComfyUI Bawk Nodes</h1>

**Compact FLUX workflow nodes for ComfyUI**

FLUX.1 dev · Krea · Kontext · FLUX.2 dev · FLUX.2 klein 4B / 9B · GGUF

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-brightgreen)](https://github.com/comfyanonymous/ComfyUI)
[![Version](https://img.shields.io/badge/Version-3.0.0-orange)]()

</div>

Five nodes cover a full multi-LoRA FLUX workflow:

```
Bawk Model Loader ─┬─► Bawk Wildcard Encoder ─► Bawk Sampler ─► Bawk Image Saver
Bawk LoRA Stack ───┘
```

FLUX models do not use negative prompts, so there is none anywhere in this pack.

## Nodes

### 🚀 Bawk Model Loader
Loads diffusion model, VAE and text encoder(s) in one node.

| Model | `clip_name1` | `clip_name2` | `vae_name` |
| --- | --- | --- | --- |
| FLUX.1 dev / Krea / Kontext | `t5xxl_*.safetensors` | `clip_l.safetensors` | `ae.safetensors` |
| FLUX.2 dev | `mistral_3_small_flux2_*.safetensors` | `none` | `flux2-vae.safetensors` |
| FLUX.2 klein 4B | `qwen_3_4b.safetensors` | `none` | `flux2-vae.safetensors` |
| FLUX.2 klein 9B | `qwen_3_8b_*.safetensors` | `none` | `flux2-vae.safetensors` |

`.gguf` diffusion models and text encoders show up in the same dropdowns when [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) is installed.

### 🔗 Bawk LoRA Stack
Six LoRA slots per node. Chain stacks through the `lora_stack` input to combine sets, and flip `enabled` on a stack to switch a whole set on or off. Feed the result into the encoder's `lora_stack` input.

### 🎲 Bawk Wildcard Encoder
- Applies the incoming `lora_stack` plus six inline LoRA slots to MODEL and CLIP.
- `{red|green|blue}` wildcards, seeded by `wildcard_seed` (nesting allowed).
- Outputs the resolved prompt for the saver.

### 🐓 Bawk Sampler
Latent, sampling and VAE decode in one node. Detects FLUX.1 vs FLUX.2 from the model and picks the right latent layout and schedule.

| Input | Notes |
| --- | --- |
| `resolution` | Presets, `Match input / reference image`, or `Custom`. A linked `WIDTHxHEIGHT` string also works. |
| `steps` / `guidance` | FLUX.1 dev: 20 / 3.5 · Kontext: 20 / 2.5 · FLUX.2 dev: 20 / 4.0 · klein distilled: 4 (guidance ignored) |
| `scheduler` | `auto` = `simple` for FLUX.1, official FLUX.2 schedule for FLUX.2 |
| `input_image` + `denoise` | img2img |
| `reference_image` / `reference_image_2` | Kontext and FLUX.2 / klein image editing. Images are scaled to the model's preferred size and attached as reference latents. |
| `cfg` | Leave at 1.0 for all distilled models. Klein **base** models use real CFG (~4-5) against an empty negative. |
| `max_shift` / `base_shift` | FLUX.1 only |

### 💾 Bawk Image Saver
Saves to `output/[model]-DD-MM-YYYY/`. PNG files embed the workflow (drag them back into ComfyUI); jpg / webp get a `.json` sidecar. Optional prompt `.txt` and Discord webhook upload.

### 📸 Bawk Image Loader · 📁 Bawk Batch Processor
Load Image with resize / pad-to-square and size outputs. Batch Processor reads one row of a CSV / JSON file (`prompt, seed, steps, guidance, resolution`) so `batch_index` on `increment` walks through a list of prompts.

## Installation

**ComfyUI Manager**: search for "Bawk Nodes" and install.

**Manual**:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/juddisjudd/ComfyUI-BawkNodes.git
```

Requires a ComfyUI build with FLUX.2 support (late 2025 or newer). GGUF loading needs ComfyUI-GGUF.

Example workflows for every supported model family are in [`workflows/`](workflows/): FLUX.1 dev text-to-image (with LoRA sets), img2img, Kontext edit, FLUX.2 dev, klein 4B text-to-image and klein 9B two-reference edit.

## Changes in 3.0.0

- FLUX.2 dev and klein support (latent layout, schedule, single text encoder), GGUF loading, reference images for Kontext / klein edit.
- Fixed: guidance was never applied (wrong conditioning key), PNG metadata was not ComfyUI-loadable, image loader validation was disabled.
- LoRA sets: new Bawk LoRA Stack (chainable, toggleable) replaces the broken dynamic stack + apply pair.
- Removed Z-Image loader (use Bawk Model Loader), the placeholder ControlNet node, the unused AI prompt enhancement code and ~1,200 lines of dead utilities.
- Widget layouts changed, so workflows saved with 2.x need their Bawk nodes re-added.

## License

GPL-3.0 - see [LICENSE](LICENSE).
