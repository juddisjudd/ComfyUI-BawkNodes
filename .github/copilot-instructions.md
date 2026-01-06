# ComfyUI-BawkNodes - AI Coding Instructions

## Project Overview
A ComfyUI custom node pack optimized for **FLUX** and **Z-Image** models. Each node combines multiple functions into clean, workflow-friendly tools. Integrates with ComfyUI's extension system via `NODE_CLASS_MAPPINGS` and `WEB_DIRECTORY`. Compatible with **Nodes 2.0** (Vue-based rendering).

## Architecture

### Node Registration Flow
```
__init__.py → nodes.py → nodes/*.py (individual node classes)
```
- **Entry point**: `__init__.py` exports `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`, `WEB_DIRECTORY`
- **Node definitions**: Each node is a class in `nodes/` with `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`
- **Frontend**: `js/bawk_nodes.js` provides UI enhancements via ComfyUI's extension API

### Module Structure
- `nodes/` - Individual node implementations (one class per file)
- `modules/` - Shared utilities (`validation.py`, `cache_manager.py`, `model_utils.py`)
- `js/` - Frontend UI enhancements and node-specific widgets

## Node Development Patterns

### Required Class Structure
```python
class MyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { ... },
            "optional": { ... },
            "hidden": { ... }  # For PROMPT, EXTRA_PNGINFO
        }
    
    RETURN_TYPES = ("MODEL", "IMAGE", ...)
    RETURN_NAMES = ("model", "image", ...)
    FUNCTION = "main_method_name"
    CATEGORY = "BawkNodes/subcategory"
    DESCRIPTION = "📝 Emoji + tooltip description shown on hover"
    OUTPUT_NODE = True  # Only for terminal nodes like savers
    
    def main_method_name(self, **inputs):
        # Implementation
        return (output1, output2, ...)
```

### Input Type Conventions
- Use `folder_paths.get_filename_list()` for model/file dropdowns
- Add `"tooltip"` to input dicts for user guidance
- Use `"forceInput": True` for inputs that must come from connections
- **FLUX presets**: steps=30, guidance=3.5, max_shift=1.15, base_shift=0.5
- **Z-Image presets**: steps=8, guidance=0.0 (distilled, no CFG needed)

### Error Handling Pattern
```python
try:
    print(f"[NodeName] Status message")
    # Operation
except Exception as e:
    print(f"[NodeName] ❌ Error: {str(e)}")
    return (safe_defaults,)
```

### Validation
Use `VALIDATE_INPUTS` classmethod for pre-execution checks:
```python
@classmethod
def VALIDATE_INPUTS(cls, **kwargs):
    errors = []
    if not valid: errors.append("message")
    return "; ".join(errors) if errors else True
```

## Model-Specific Parameters

| Parameter | FLUX | Z-Image-Turbo |
|-----------|------|---------------|
| Steps | 30 (20 fast, 40 quality) | **8** |
| Guidance | 3.5 (2.0-7.0 range) | **0.0** (no CFG) |
| Resolution | 1024x1024 optimal | 1024x1024 optimal |
| Shifts | max=1.15, base=0.5 | N/A |
| CLIP type | `CLIPType.FLUX` | Built-in T5 |

## Key Integrations

### ComfyUI APIs Used
- `comfy.sd` - Model/VAE/CLIP loading (`load_diffusion_model`, `load_clip`)
- `comfy.samplers` - Sampler/scheduler lists
- `comfy.model_management` - Memory management
- `comfy.utils` - File loading (`load_torch_file`)
- `folder_paths` - Standard directories (`get_filename_list`, `get_full_path_or_raise`)
- `comfy_extras.nodes_custom_sampler` - Advanced sampling components
- `comfy.supported_models` - Model detection (FLUX, ZImage/Lumina2, etc.)

## Frontend Development (`js/`)

### Extension Registration (Nodes 2.0 Compatible)
```javascript
import { app } from "/scripts/app.js";
app.registerExtension({
    name: "bawk.nodes.feature",
    async beforeRegisterNodeDef(nodeType, nodeData, app) { ... },
    nodeCreated(node, app) { ... },  // Configure new instances
    getCustomWidgets() { return {} },  // Custom widget types
});
```

### Key Hooks
| Hook | Purpose |
|------|---------|
| `beforeRegisterNodeDef` | Modify node prototype/handlers |
| `nodeCreated` | Configure new node instances |
| `getCustomWidgets` | Add custom widget types |

Check for Bawk nodes: compare `nodeData.name` against `isBawkNode()` list in `bawk_nodes.js`.

## Testing & Workflows

- Example workflow: `workflows/example_bawknodes_workflow.json`
- Test in ComfyUI by placing in `custom_nodes/` directory
- Startup messages print to console with colored formatting
- Test with both Nodes 2.0 (Vue) and LiteGraph.js modes

## Adding a New Node

1. Create `nodes/my_new_node.py` with class following patterns above
2. Add import to root `__init__.py`
3. Add to `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`
4. Use emoji prefix in display name (🐓, 📸, 💾, ⚡, etc.)
5. Add node name to `isBawkNode()` in `js/bawk_nodes.js`
6. Add color scheme to `colorSchemes` object in `js/bawk_nodes.js`

## Z-Image Integration

Z-Image is **natively supported** in ComfyUI via `comfy.supported_models.ZImage` (extends Lumina2).
Uses standard `comfy.sd.load_diffusion_model()` - works with BawkSampler like FLUX models.

**Workflow**: ZImageModelLoader → FluxWildcardEncode → BawkSampler → FluxImageSaver
(Same as FLUX workflow, just swap the model loader)

**Recommended settings**: steps=8, guidance=0.0 (distilled model, no CFG needed)

Model files: Place in `ComfyUI/models/diffusion_models/` directory.
