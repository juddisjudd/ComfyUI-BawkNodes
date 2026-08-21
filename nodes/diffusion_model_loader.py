"""
DiffusionModelLoader - FLUX.1 / FLUX.2 model, VAE and text encoder loader.
Supports .safetensors natively and .gguf through ComfyUI-GGUF when installed.
"""

import os

import torch
import comfy.sd
import comfy.utils
import folder_paths

WEIGHT_DTYPES = {
    "default": {},
    "fp8_e4m3fn": {"dtype": torch.float8_e4m3fn},
    "fp8_e4m3fn_fast": {"dtype": torch.float8_e4m3fn, "fp8_optimizations": True},
    "fp8_e5m2": {"dtype": torch.float8_e5m2},
}


def _file_list(folder, gguf_folder):
    files = list(folder_paths.get_filename_list(folder))
    if gguf_folder in folder_paths.folder_names_and_paths:
        files += [f for f in folder_paths.get_filename_list(gguf_folder) if f not in files]
    return files


def _is_gguf(name):
    return name.lower().endswith(".gguf")


def _gguf_node(class_name):
    import nodes as comfy_nodes
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(class_name)
    if cls is None:
        raise RuntimeError("Loading .gguf files needs the ComfyUI-GGUF custom node pack (city96/ComfyUI-GGUF)")
    return cls()


def _clip_type(dual):
    if dual:
        return comfy.sd.CLIPType.FLUX
    clip_type = getattr(comfy.sd.CLIPType, "FLUX2", None)
    if clip_type is None:
        raise RuntimeError("FLUX.2 text encoders need a newer ComfyUI (CLIPType.FLUX2 missing)")
    return clip_type


class DiffusionModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        text_encoders = _file_list("text_encoders", "clip_gguf")
        return {
            "required": {
                "model_name": (_file_list("diffusion_models", "unet_gguf"), {
                    "tooltip": "FLUX diffusion model from models/diffusion_models (.safetensors or .gguf)",
                }),
                "vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "ae.safetensors for FLUX.1, flux2-vae.safetensors for FLUX.2",
                }),
                "clip_name1": (text_encoders, {
                    "tooltip": "FLUX.1: t5xxl. FLUX.2-dev: mistral_3_small. FLUX.2-klein: qwen_3_4b (4B) / qwen_3_8b (9B)",
                }),
                "clip_name2": (["none"] + text_encoders, {
                    "tooltip": "FLUX.1: clip_l. Leave 'none' for FLUX.2 models (single text encoder)",
                }),
                "weight_dtype": (list(WEIGHT_DTYPES), {
                    "tooltip": "Diffusion model precision. fp8 halves VRAM. Ignored for .gguf files",
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "VAE", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "VAE", "CLIP", "MODEL_STRING")
    FUNCTION = "load"
    CATEGORY = "BawkNodes/loaders"
    DESCRIPTION = "Load a FLUX.1 / FLUX.2 diffusion model, VAE and text encoder(s) in one node. GGUF supported via ComfyUI-GGUF."

    def load(self, model_name, vae_name, clip_name1, clip_name2="none", weight_dtype="default"):
        model = self._load_model(model_name, weight_dtype)
        vae = self._load_vae(vae_name)
        clip = self._load_clip(clip_name1, None if clip_name2 == "none" else clip_name2)
        model_string = os.path.splitext(os.path.basename(model_name))[0]
        return (model, vae, clip, model_string)

    def _load_model(self, model_name, weight_dtype):
        if _is_gguf(model_name):
            return _gguf_node("UnetLoaderGGUF").load_unet(model_name)[0]
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        model = comfy.sd.load_diffusion_model(path, model_options=dict(WEIGHT_DTYPES[weight_dtype]))
        if model is None:
            raise RuntimeError(f"Could not detect model type of '{model_name}'")
        return model

    def _load_vae(self, vae_name):
        path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        return comfy.sd.VAE(sd=sd, metadata=metadata)

    def _load_clip(self, clip_name1, clip_name2):
        names = [n for n in (clip_name1, clip_name2) if n]
        dual = len(names) == 2
        if any(_is_gguf(n) for n in names):
            if dual:
                return _gguf_node("DualCLIPLoaderGGUF").load_clip(names[0], names[1], "flux")[0]
            return _gguf_node("CLIPLoaderGGUF").load_clip(names[0], "flux2")[0]
        paths = [folder_paths.get_full_path_or_raise("text_encoders", n) for n in names]
        return comfy.sd.load_clip(
            ckpt_paths=paths,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=_clip_type(dual),
        )
