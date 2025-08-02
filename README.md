# 🐓 ComfyUI Bawk Nodes

A collection of ComfyUI nodes **focused primarily on FLUX model workflows**. While some nodes may work with other diffusion models, all development and testing is done specifically with FLUX architectures to ensure optimal performance and compatibility.

## ⚡ FLUX-First Design Philosophy

Our nodes are built with FLUX models as the primary target:
- **Optimized for FLUX**: All nodes designed around FLUX model architecture
- **FLUX-Tested**: Extensive testing with FLUX Dev, FLUX Schnell, and FLUX variants
- **Other Models**: May work with SDXL/SD1.5 but **not officially supported or tested**

> **Note**: While these nodes might function with other diffusion models, we only guarantee compatibility and provide support for FLUX models. Use with other models at your own discretion.

## Current Nodes

### 🚀 Diffusion Model Loader (Advanced)
A powerful diffusion model loader specifically optimized for FLUX models with separate component loading.

**FLUX-Optimized Features:**
- **FLUX Model Support**: Loads models from `diffusion_models` directory (FLUX format)
- **Separate Component Loading**: Independent VAE and dual CLIP text encoder support
- **FLUX Weight Types**: Support for FP8, FP16, BF16, and FP32 optimized for FLUX
- **FLUX CLIP Types**: Proper `CLIPType.FLUX` handling for T5 + CLIP-L encoders
- **Model String Output**: Returns clean model name for workflow identification
- **FLUX Validation**: Input validation designed around FLUX model requirements

## 📦 Installation

### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "Bawk Nodes" or "ComfyUI-BawkNodes"
3. Click Install

### Method 2: Manual Installation
1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/juddisjudd/ComfyUI-BawkNodes.git
   ```

3. Restart ComfyUI

## 🎯 Usage

### Diffusion Model Loader (Advanced)
1. Add the "🚀 Diffusion Model Loader (Advanced)" node to your workflow
2. Select your diffusion model from the dropdown
3. Choose your VAE (or use "baked VAE" for none)
4. Select your text encoders (CLIP models)
5. Choose your preferred weight data type
6. Connect the outputs to your workflow

### Advanced Configuration

#### Weight Data Types
- **default**: Automatic selection based on hardware
- **fp8_e4m3fn**: 8-bit floating point (requires modern GPUs)
- **fp8_e4m3fn_fast**: Optimized 8-bit variant
- **fp8_e5m2**: Alternative 8-bit format
- **fp16**: 16-bit floating point (most common)
- **bf16**: Brain floating point 16-bit
- **fp32**: Full precision 32-bit

#### Separate VAE Loading
- Select "baked VAE" to use the VAE included in your model
- Choose a specific VAE file to override the model's VAE

#### Dual Text Encoders (FLUX Models)
- **clip_name1**: First text encoder (typically T5 for FLUX)
- **clip_name2**: Second text encoder (typically CLIP-L for FLUX)

## 🔧 Node Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | STRING | - | Checkpoint file name (required) |
| `weight_dtype` | COMBO | "default" | Weight data type |
| `vae_name` | COMBO | "baked VAE" | VAE model (optional) |
| `clip_name1` | COMBO | "none" | First text encoder (optional) |
| `clip_name2` | COMBO | "none" | Second text encoder (optional) |

## 📤 Node Outputs

| Output | Type | Description |
|--------|------|-------------|
| `MODEL` | MODEL | Loaded diffusion model |
| `VAE` | VAE | Variational autoencoder |
| `CLIP` | CLIP | Text encoder(s) |
| `MODEL_STRING` | STRING | Model information summary |


## 🔮 Planned Nodes (FLUX-Focused)

***TBD***
*All future nodes will maintain our FLUX-first design philosophy*

## Example Workflows

### Basic FLUX Workflow
```
Diffusion Model Loader (Advanced)
├── model_name: "flux1-dev.safetensors"
├── weight_dtype: "fp8_e4m3fn"
├── vae_name: "ae.safetensors"
├── clip_name1: "t5xxl_fp16.safetensors"
└── clip_name2: "clip_l.safetensors"
```

## ⚡ FLUX Performance Tips

1. **Hardware Optimization**:
   - Use FP8 data types on RTX 4000+ series GPUs for maximum VRAM efficiency
   - FLUX models benefit significantly from modern GPU architectures
   - Ensure adequate VRAM (12GB+ recommended for FLUX Dev)

2. **FLUX-Specific Settings**:
   - Use `fp8_e4m3fn` for best quality/memory balance
   - Use `fp8_e4m3fn_fast` for maximum speed
   - T5 + CLIP-L combination provides optimal text understanding

3. **Model Organization**:
   - Keep FLUX models in `ComfyUI/models/diffusion_models/`
   - Use `ae.safetensors` VAE for all FLUX variants
   - Separate text encoders allow better memory management

## Troubleshooting

### FLUX-Specific Issues

**"Model not found"**
- Ensure FLUX models are in `ComfyUI/models/diffusion_models/` (NOT checkpoints!)
- VAE files go in `ComfyUI/models/vae/`
- Text encoders go in `ComfyUI/models/text_encoders/` or `ComfyUI/models/clip/`

**Matrix multiplication errors with FLUX samplers**
- This usually means a checkpoint was loaded instead of a diffusion model
- Ensure your FLUX model is in `diffusion_models` directory
- Our loader is specifically designed to prevent this issue

**"Insufficient VRAM" with FLUX**
- Try `fp8_e4m3fn` or `fp8_e4m3fn_fast` weight types
- FLUX models are large - consider using smaller variants for lower VRAM
- Ensure no other models are loaded in memory

**Empty dropdowns**
- The node only shows files that actually exist in the correct directories
- Check that your FLUX files are in the proper locations
- This loader ONLY shows diffusion models (FLUX format)

**Not compatible with other models**
- Remember: These nodes are designed specifically for FLUX
- Other diffusion models may not work correctly
- Use standard ComfyUI loaders for non-FLUX models


## Support

- **Issues**: [GitHub Issues](https://github.com/juddisjudd/ComfyUI-BawkNodes/issues)

---

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/P5P57KRR9)