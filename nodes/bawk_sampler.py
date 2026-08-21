"""
BawkSampler - latent creation, FLUX sampling and VAE decode in one node.
Handles FLUX.1 (dev / Krea / Kontext) and FLUX.2 (dev / klein) automatically.
"""

import logging
import math
import re

import torch
import comfy.model_base
import comfy.model_management
import comfy.model_sampling
import comfy.samplers
import comfy.utils
import node_helpers
from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise, SamplerCustomAdvanced

try:
    from comfy_extras.nodes_flux import get_schedule as flux2_schedule
except ImportError:
    flux2_schedule = None

RESOLUTION_PRESETS = {
    "1:1 Square - 1024x1024": (1024, 1024),
    "1:1 Square Large - 1408x1408": (1408, 1408),
    "4:3 Landscape - 1152x896": (1152, 896),
    "3:4 Portrait - 896x1152": (896, 1152),
    "3:2 Landscape - 1216x832": (1216, 832),
    "2:3 Portrait - 832x1216": (832, 1216),
    "16:9 Landscape - 1344x768": (1344, 768),
    "9:16 Portrait - 768x1344": (768, 1344),
    "1:1 Draft - 512x512": (512, 512),
}
MATCH_IMAGE = "Match input / reference image"
CUSTOM = "Custom (width / height below)"
AUTO_SCHEDULER = "auto"
DIMS_RE = re.compile(r"(\d+)\s*x\s*(\d+)")

KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328), (832, 1248), (880, 1184), (944, 1104),
    (1024, 1024),
    (1104, 944), (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720), (1504, 688), (1568, 672),
]


def _model_family(model):
    flux2_cls = getattr(comfy.model_base, "Flux2", None)
    if flux2_cls is not None and isinstance(model.model, flux2_cls):
        return "flux2"
    if isinstance(model.model, comfy.model_base.Flux):
        return "flux1"
    return "other"


def _round(value, multiple):
    return max(multiple, int(round(value / multiple)) * multiple)


def _crop_to_multiple(image, multiple):
    h, w = image.shape[1], image.shape[2]
    nh, nw = h // multiple * multiple, w // multiple * multiple
    if (nh, nw) == (h, w):
        return image
    y, x = (h - nh) // 2, (w - nw) // 2
    return image[:, y:y + nh, x:x + nw, :]


def _scale_image(image, width, height):
    samples = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", "center")
    return samples.movedim(1, -1)


def _reference_size(image, family, align):
    h, w = image.shape[1], image.shape[2]
    if family == "flux1":
        _, rw, rh = min((abs(w / h - pw / ph), pw, ph) for pw, ph in KONTEXT_RESOLUTIONS)
        return rw, rh
    scale = math.sqrt(1_000_000 / (w * h))
    return _round(w * scale, align), _round(h * scale, align)


def _zero_out(conditioning):
    out = []
    for cond, opts in conditioning:
        opts = opts.copy()
        if opts.get("pooled_output") is not None:
            opts["pooled_output"] = torch.zeros_like(opts["pooled_output"])
        out.append([torch.zeros_like(cond), opts])
    return out


def _patch_flux1_shift(model, max_shift, base_shift, width, height):
    x1, x2 = 256, 4096
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    shift = (width * height / (8 * 8 * 2 * 2)) * mm + b

    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST):
        pass

    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=shift)
    patched = model.clone()
    patched.add_object_patch("model_sampling", model_sampling)
    return patched


def _sigmas(model, family, scheduler, steps, denoise, width, height):
    if denoise <= 0.0:
        return torch.FloatTensor([])
    total_steps = steps if denoise >= 1.0 else int(steps / denoise)
    if scheduler == AUTO_SCHEDULER:
        if family == "flux2" and flux2_schedule is not None:
            sigmas = torch.FloatTensor(flux2_schedule(total_steps, round(width * height / 256)))
            return sigmas[-(steps + 1):]
        scheduler = "simple"
    sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
    return sigmas[-(steps + 1):]


class BawkSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "resolution": (list(RESOLUTION_PRESETS) + [MATCH_IMAGE, CUSTOM], {
                    "default": "1:1 Square - 1024x1024",
                    "tooltip": "Output size. Any 'WIDTHxHEIGHT' string also works when linked from another node",
                }),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "Images per run"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 100,
                    "tooltip": "FLUX.1-dev / Krea / Kontext / FLUX.2-dev: 20-30. FLUX.2 klein distilled: 4",
                }),
                "guidance": ("FLOAT", {
                    "default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Distilled guidance. FLUX.1-dev 3.5, Kontext 2.5, FLUX.2-dev 4.0. Klein distilled ignores it",
                }),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": ([AUTO_SCHEDULER] + comfy.samplers.KSampler.SCHEDULERS, {
                    "default": AUTO_SCHEDULER,
                    "tooltip": "auto = 'simple' for FLUX.1, the official FLUX.2 schedule for FLUX.2",
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "1.0 for text-to-image. 0.5-0.9 for img2img with input_image",
                }),
            },
            "optional": {
                "input_image": ("IMAGE", {"tooltip": "img2img source. Encoded, then denoised by 'denoise'"}),
                "reference_image": ("IMAGE", {
                    "tooltip": "Edit / reference image for FLUX.1 Kontext and FLUX.2 (klein) edit models",
                }),
                "reference_image_2": ("IMAGE", {"tooltip": "Second reference image (FLUX.2 / klein)"}),
                "cfg": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Real CFG against an empty negative. Only for FLUX.2 klein *base* models (~4-5). "
                               "Keep 1.0 for every distilled FLUX model",
                }),
                "max_shift": ("FLOAT", {
                    "default": 1.15, "min": 0.0, "max": 10.0, "step": 0.01, "advanced": True,
                    "tooltip": "FLUX.1 timestep shift at high resolution. Ignored by FLUX.2",
                }),
                "base_shift": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "advanced": True,
                    "tooltip": "FLUX.1 timestep shift at low resolution. Ignored by FLUX.2",
                }),
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "custom_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("images", "latent")
    FUNCTION = "sample"
    CATEGORY = "BawkNodes/sampling"
    DESCRIPTION = (
        "All-in-one FLUX sampler: builds the latent (txt2img / img2img / reference edit), "
        "samples and decodes. Auto-detects FLUX.1 vs FLUX.2."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, resolution):
        if resolution in (MATCH_IMAGE, CUSTOM) or DIMS_RE.search(resolution):
            return True
        return f"Unknown resolution '{resolution}'. Use a preset or a 'WIDTHxHEIGHT' string"

    def sample(self, model, conditioning, vae, resolution, batch_size, seed, steps, guidance, sampler, scheduler,
               denoise, input_image=None, reference_image=None, reference_image_2=None, cfg=1.0,
               max_shift=1.15, base_shift=0.5, custom_width=1024, custom_height=1024):
        family = _model_family(model)
        latent_format = model.get_model_object("latent_format")
        downscale = latent_format.spacial_downscale_ratio
        align = downscale * 2
        references = [r for r in (reference_image, reference_image_2) if r is not None]

        if input_image is not None:
            pixels = _crop_to_multiple(input_image[:, :, :, :3], align)
            samples = vae.encode(pixels)
            if samples.shape[0] != batch_size:
                samples = samples[:1].repeat(batch_size, 1, 1, 1)
            latent = {"samples": samples}
            width, height = pixels.shape[2], pixels.shape[1]
        else:
            width, height = self._target_size(resolution, custom_width, custom_height, references, family, align)
            samples = torch.zeros(
                [batch_size, latent_format.latent_channels, height // downscale, width // downscale],
                device=comfy.model_management.intermediate_device(),
            )
            latent = {"samples": samples}

        cond = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        for ref in references:
            rw, rh = _reference_size(ref, family, align)
            ref_latent = vae.encode(_scale_image(ref[:, :, :, :3], rw, rh))
            cond = node_helpers.conditioning_set_values(cond, {"reference_latents": [ref_latent]}, append=True)

        work_model = _patch_flux1_shift(model, max_shift, base_shift, width, height) if family == "flux1" else model

        if cfg > 1.0:
            guider = comfy.samplers.CFGGuider(work_model)
            guider.set_conds(cond, _zero_out(cond))
            guider.set_cfg(cfg)
        else:
            guider = Guider_Basic(work_model)
            guider.set_conds(cond)

        sigmas = _sigmas(work_model, family, scheduler, steps, denoise, width, height)
        sampler_object = comfy.samplers.sampler_object(sampler)
        noise = Noise_RandomNoise(seed)

        logging.info(
            f"[BawkSampler] {family} {width}x{height} x{batch_size} seed={seed} steps={steps} "
            f"guidance={guidance} cfg={cfg} {sampler}/{scheduler} denoise={denoise} refs={len(references)}"
        )
        out = SamplerCustomAdvanced().sample(noise, guider, sampler_object, sigmas, latent)[0]

        images = vae.decode(out["samples"])
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images, out)

    def _target_size(self, resolution, custom_width, custom_height, references, family, align):
        if resolution == CUSTOM:
            width, height = custom_width, custom_height
        elif resolution == MATCH_IMAGE:
            if not references:
                raise ValueError(f"'{MATCH_IMAGE}' needs input_image or reference_image")
            return _reference_size(references[0], family, align)
        else:
            match = DIMS_RE.search(resolution)
            width, height = int(match.group(1)), int(match.group(2))
        return _round(width, align), _round(height, align)
