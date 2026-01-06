<div align="center">
<h1>ComfyUI Bawk Nodes v2.3.0</h1>

**The Ultimate FLUX & Z-Image Workflow Suite for ComfyUI**

*Transform your AI image generation with powerful, easy-to-use nodes designed for FLUX and Z-Image models*

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-brightgreen)](https://github.com/comfyanonymous/ComfyUI)
[![FLUX](https://img.shields.io/badge/FLUX-Optimized-blue)](https://blackforestlabs.ai/)
[![Z-Image](https://img.shields.io/badge/Z--Image-Supported-cyan)](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
[![Version](https://img.shields.io/badge/Version-2.3.0-orange)]()

</div>

---

## **Why Choose Bawk Nodes?**

### **All-in-One Workflow Solutions**
- **No more node spaghetti!** Each Bawk Node combines multiple functions into clean, powerful tools
- **FLUX & Z-Image optimized** - Nodes designed for both FLUX and Z-Image-Turbo models
- **Professional results** with minimal setup time

### **Perfect For:**
- **Content Creators** - Instagram, TikTok, and social media workflows
- **Artists & Designers** - Professional image generation and experimentation
- **Hobbyists** - Easy-to-use tools without complicated setups
- **Power Users** - Advanced features like batch processing and Discord integration

---

## **What You Get**

### **Bawk Image Loader**
*Your gateway to img2img workflows*
- **Click to browse** - No more typing file paths!
- **Auto-rotation** - Handles phone photos perfectly
- **Smart resizing** - Perfect dimensions every time
- **Multiple formats** - JPG, PNG, WEBP, and more

### **Bawk Wildcard Encoder**
*Text prompts + LoRAs + AI enhancement made simple*
- **6 LoRA slots** with smart recommendations
- **Wildcard support** - Randomize your prompts
- **Optional AI enhancement** - Claude, OpenAI & Grok integration for better prompts
- **Helpful tooltips** - Know exactly what each setting does
- **FLUX-optimized** text encoding

### **Bawk Sampler**
*The heart of your workflow*
- **Text-to-Image & Image-to-Image** in one node
- **Resolution presets** - Instagram, TikTok, 4K, and more
- **Smart validation** - Helpful tips and warnings
- **All-in-one** - Generates, samples, and decodes in one step

### **Bawk Image Saver**
*Save and share like a pro*
- **Organized folders** by model and date
- **Discord integration** - Auto-post your creations
- **Batch support** - Upload multiple images at once
- **Metadata saving** - Never lose your settings

### **Bawk Batch Processor**
*Automate your workflow*
- **CSV/JSON support** - Process hundreds of prompts
- **A/B testing** - Compare different settings easily
- **Preview mode** - Check your files before processing
- **Perfect for** content creation at scale

### **Bawk ControlNet**
*Guided generation made easy*
- **Built-in preprocessing** - Canny, depth, pose, and more
- **No external tools needed** - Everything works out of the box
- **FLUX-optimized** control strength recommendations
- **Multiple control types** in one node

### **Bawk Model Loader**
*Advanced model management*
- **Smart caching** - Faster loading times
- **Memory optimization** - Handle large models efficiently
- **Validation** - Helpful warnings for compatibility issues
- **FLUX-specific** optimizations

### **Z-Image Loader**
*Lightning-fast generation with Z-Image-Turbo*
- **Native ComfyUI support** - Uses built-in Z-Image/Lumina2 architecture
- **Only 8 steps** - State-of-the-art quality with minimal compute
- **No CFG needed** - Guidance scale 0.0 (distilled model)
- **Same workflow** - Drop-in replacement for FLUX loader

---

## **Quick Start Guide**

### **1. Basic Text-to-Image Workflow (FLUX)**
```
Bawk Model Loader → Bawk Wildcard Encoder → Bawk Sampler → Bawk Image Saver
                                            (30 steps, guidance 3.5)
```

### **2. Z-Image Turbo Workflow (8 steps!)**
```
Z-Image Loader → Bawk Wildcard Encoder → Bawk Sampler → Bawk Image Saver
                                         (8 steps, guidance 0.0)
```

### **3. Image-to-Image Workflow**
```
Bawk Image Loader → Bawk Sampler → Bawk Image Saver
                   ↗ (set denoise 0.6-0.8)
```

### **4. AI-Enhanced Prompt Workflow**
```
Bawk Wildcard Encoder → Bawk Sampler → Bawk Image Saver
  (LoRAs + wildcards       (generate)
   + AI enhancement)
```

### **5. Batch Generation Workflow**
```
Bawk Batch Processor → Bawk Wildcard Encoder → Bawk Sampler → Bawk Image Saver
```

---

## **Popular Use Cases**

### **Social Media Content**
- Use **Instagram presets** in Bawk Sampler (1080x1080, 1080x1920)
- Set up **Discord webhooks** to auto-post to your content channels
- **Batch process** multiple variations for A/B testing

### **Art & Design**
- Load reference images with **Bawk Image Loader**
- Use **LoRA slots** for consistent character/style
- Try **different denoise levels** for style transfer effects

### **Workflow Automation**
- Create **CSV files** with prompts and settings
- Use **Bawk Batch Processor** for unattended generation
- **Discord integration** notifies you when batches complete

### **AI-Enhanced Prompting**
- **Built into Wildcard Encoder** - Just enable AI enhancement and add your API key
- Transform **simple prompts** into detailed, professional descriptions
- **Multiple AI providers** - Claude (Anthropic), OpenAI (GPT-4/3.5), and Grok (xAI)
- **8 enhancement styles** - Photography, cinematic, artistic, and more
- **100% optional** - Works perfectly without AI enhancement enabled

---

## **Pro Tips**

### **LoRA Management**
- **Slot 1**: Main character/style (strength 0.8-1.2)
- **Slot 2**: Secondary effects (strength 0.6-1.0)
- **Slot 3**: Clothing/objects (strength 0.4-0.8)
- **Slots 4-6**: Fine details and adjustments (strength 0.2-0.6)

### **Denoise Settings for Img2Img**
- **0.3-0.5**: Subtle improvements, keep original structure
- **0.6-0.7**: Style changes, good balance
- **0.8-0.9**: Major transformations
- **1.0**: Complete replacement (text2img mode)

### **Batch Processing**
Create a CSV file like this:
```csv
prompt,seed,steps,guidance,resolution
"beautiful sunset",12345,30,3.5,"Instagram Square - 1080x1080 - 1.2MP"
"city at night",67890,25,4.0,"Instagram Story - 1080x1920 - 2.1MP"
```

### **AI Enhancement (Built into Wildcard Encoder)**
- **Enable in Wildcard Encoder**: Set ai_enhance_enabled to True and add your API key
- **Start simple**: "cat on chair" → Enhanced: "Elegant domestic cat with sleek fur sitting gracefully on vintage wooden chair, soft natural lighting, professional photography composition"
- **Provider Comparison**: Claude (balanced, cost-effective), GPT-4 (high quality, expensive), GPT-3.5 (fast, budget), Grok (creative, experimental)
- **API Key Security**: Keys are masked in logs and only used when enhancement is enabled
- **Creativity Levels**: 0.3 for subtle enhancement, 0.7 for creative improvement, 1.0 for artistic interpretation

---

## **What's New in v2.3.0**

- **Z-Image-Turbo Support** - Native support for Z-Image models (8 steps, no CFG needed!)
- **Z-Image Loader** - Drop-in replacement for FLUX workflow, same node chain
- **Unified Sampler** - BawkSampler works with both FLUX and Z-Image models
- **Example Workflows** - Included workflows for both FLUX and Z-Image

### Previous Updates (v2.1.0)
- **Integrated AI Enhancement** - Built directly into Bawk Wildcard Encoder (100% optional)
- **Multi-Provider AI Support** - Claude (Anthropic), OpenAI (GPT-4/3.5), and Grok (xAI)
- **8 Enhancement Styles** - Photography, cinematic, artistic, portrait, landscape, and more
- **Simplified Workflow** - Single node handles wildcards, LoRAs, and AI enhancement

---

## **Installation**

### **Method 1: ComfyUI Manager (Recommended)**
1. Open ComfyUI Manager
2. Search for "Bawk Nodes"
3. Click Install
4. Restart ComfyUI

### **Method 2: Manual Installation**
1. Navigate to `ComfyUI/custom_nodes/`
2. Clone this repository:
   ```bash
   git clone https://github.com/juddisjudd/ComfyUI-BawkNodes.git
   ```
3. Restart ComfyUI

---

## **Community & Support**

- **Issues & Feature Requests**: [GitHub Issues](https://github.com/juddisjudd/ComfyUI-BawkNodes/issues)

---

## **License**

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.
