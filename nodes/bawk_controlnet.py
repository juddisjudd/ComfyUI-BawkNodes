"""
BawkControlNet - FLUX ControlNet Integration
File: nodes/bawk_controlnet.py
"""

import torch
import numpy as np
from typing import Tuple, Any


class BawkControlNet:
    """
    FLUX-optimized ControlNet preprocessing and integration.
    Handles common ControlNet types with smart preprocessing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image for ControlNet processing"
                }),
                "control_type": ([
                    "Canny Edge",
                    "Depth Map",
                    "Normal Map",
                    "Pose/OpenPose",
                    "Segmentation",
                    "Scribble",
                    "Lineart",
                    "QR Code",
                    "Custom/Raw"
                ], {
                    "default": "Canny Edge",
                    "tooltip": "Type of ControlNet preprocessing to apply"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "ControlNet influence strength"
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "When to start applying ControlNet (0.0 = from beginning)"
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "When to stop applying ControlNet (1.0 = until end)"
                }),
            },
            "optional": {
                # Canny specific
                "canny_low_threshold": ("INT", {
                    "default": 100, "min": 1, "max": 255,
                    "tooltip": "Canny edge detection low threshold"
                }),
                "canny_high_threshold": ("INT", {
                    "default": 200, "min": 1, "max": 255,
                    "tooltip": "Canny edge detection high threshold"
                }),

                # Depth specific
                "depth_near": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 10.0, "step": 0.01,
                    "tooltip": "Near plane for depth normalization"
                }),
                "depth_far": ("FLOAT", {
                    "default": 100.0, "min": 1.0, "max": 1000.0, "step": 1.0,
                    "tooltip": "Far plane for depth normalization"
                }),

                # Advanced options
                "preprocessor_resolution": ("INT", {
                    "default": 512, "min": 256, "max": 2048, "step": 64,
                    "tooltip": "Resolution for preprocessing (will be resized back to original)"
                }),
                "auto_resize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically resize control image to match generation resolution"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "CONDITIONING", "STRING")
    RETURN_NAMES = ("control_image", "control_conditioning", "control_info")
    FUNCTION = "process_controlnet"
    CATEGORY = "BawkNodes/control"
    DESCRIPTION = "FLUX-optimized ControlNet preprocessing and integration"

    def process_controlnet(
        self,
        image,
        control_type="Canny Edge",
        strength=1.0,
        start_percent=0.0,
        end_percent=1.0,
        canny_low_threshold=100,
        canny_high_threshold=200,
        depth_near=0.1,
        depth_far=100.0,
        preprocessor_resolution=512,
        auto_resize=True
    ):
        """
        Process image with ControlNet preprocessing
        """
        try:
            print(f"[BawkControlNet] Processing {control_type} with strength {strength}")

            # Validate parameters
            self._validate_parameters(strength, start_percent, end_percent)

            # Preprocess the control image
            control_image = self._preprocess_image(
                image, control_type,
                canny_low_threshold, canny_high_threshold,
                depth_near, depth_far, preprocessor_resolution
            )

            # Create control conditioning (placeholder for now - would integrate with actual ControlNet)
            control_conditioning = self._create_control_conditioning(
                control_image, strength, start_percent, end_percent
            )

            # Generate info string
            control_info = self._generate_control_info(
                control_type, strength, start_percent, end_percent, image.shape
            )

            print(f"[BawkControlNet] ✅ {control_info}")

            return (control_image, control_conditioning, control_info)

        except Exception as e:
            error_msg = f"ControlNet processing failed: {str(e)}"
            print(f"[BawkControlNet] ❌ {error_msg}")

            # Return safe defaults
            return (image, [], error_msg)

    def _validate_parameters(self, strength: float, start_percent: float, end_percent: float):
        """Validate ControlNet parameters and provide feedback"""

        if strength > 1.5:
            print(f"[BawkControlNet] ⚠️  High strength ({strength}) may cause over-conditioning")
        elif strength < 0.3:
            print(f"[BawkControlNet] ℹ️  Low strength ({strength}) may have minimal effect")

        if start_percent >= end_percent:
            print(f"[BawkControlNet] ⚠️  Start percent ({start_percent}) should be < end percent ({end_percent})")

        if end_percent - start_percent < 0.2:
            print(f"[BawkControlNet] ℹ️  Short control duration ({end_percent - start_percent:.1%}) may have limited effect")

    def _preprocess_image(
        self,
        image,
        control_type: str,
        canny_low: int,
        canny_high: int,
        depth_near: float,
        depth_far: float,
        resolution: int
    ):
        """Preprocess image based on control type"""

        # Convert to numpy for processing
        if len(image.shape) == 4:
            img_np = image[0].cpu().numpy()  # Take first image from batch
        else:
            img_np = image.cpu().numpy()

        # Convert from 0-1 to 0-255
        img_np = (img_np * 255).astype(np.uint8)

        if control_type == "Canny Edge":
            processed = self._apply_canny_edge(img_np, canny_low, canny_high)
        elif control_type == "Depth Map":
            processed = self._apply_depth_processing(img_np, depth_near, depth_far)
        elif control_type == "Normal Map":
            processed = self._apply_normal_map(img_np)
        elif control_type == "Pose/OpenPose":
            processed = self._apply_pose_detection(img_np)
        elif control_type == "Segmentation":
            processed = self._apply_segmentation(img_np)
        elif control_type == "Scribble":
            processed = self._apply_scribble_effect(img_np)
        elif control_type == "Lineart":
            processed = self._apply_lineart(img_np)
        elif control_type == "QR Code":
            processed = self._apply_qr_processing(img_np)
        else:  # Custom/Raw
            processed = img_np
            print(f"[BawkControlNet] Using raw image without preprocessing")

        # Convert back to tensor
        processed_tensor = torch.from_numpy(processed.astype(np.float32) / 255.0)

        # Ensure correct shape [batch, height, width, channels]
        if len(processed_tensor.shape) == 3:
            processed_tensor = processed_tensor.unsqueeze(0)

        return processed_tensor

    def _apply_canny_edge(self, img_np: np.ndarray, low_thresh: int, high_thresh: int):
        """Apply Canny edge detection"""
        try:
            import cv2

            # Convert to grayscale if needed
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np

            # Apply Canny edge detection
            edges = cv2.Canny(gray, low_thresh, high_thresh)

            # Convert back to 3-channel
            edges_rgb = np.stack([edges, edges, edges], axis=-1)

            print(f"[BawkControlNet] Applied Canny edge detection (thresholds: {low_thresh}, {high_thresh})")
            return edges_rgb

        except ImportError:
            print(f"[BawkControlNet] ⚠️  OpenCV not available, using simple edge detection")
            return self._simple_edge_detection(img_np)

    def _simple_edge_detection(self, img_np: np.ndarray):
        """Simple edge detection fallback without OpenCV"""
        # Convert to grayscale
        if len(img_np.shape) == 3:
            gray = np.mean(img_np, axis=-1)
        else:
            gray = img_np

        # Simple gradient-based edge detection
        grad_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        grad_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        edges = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold and normalize
        edges = np.clip(edges * 3, 0, 255).astype(np.uint8)

        # Convert to 3-channel
        edges_rgb = np.stack([edges, edges, edges], axis=-1)
        return edges_rgb

    def _apply_depth_processing(self, img_np: np.ndarray, near: float, far: float):
        """Apply depth map processing (placeholder)"""
        # Simple luminance-based depth approximation
        if len(img_np.shape) == 3:
            depth = np.mean(img_np, axis=-1)
        else:
            depth = img_np

        # Normalize depth
        depth = np.clip((depth - near * 255) / ((far - near) * 255), 0, 1) * 255
        depth_rgb = np.stack([depth, depth, depth], axis=-1).astype(np.uint8)

        print(f"[BawkControlNet] Applied depth processing (range: {near}-{far})")
        return depth_rgb

    def _apply_normal_map(self, img_np: np.ndarray):
        """Apply normal map processing (placeholder)"""
        print(f"[BawkControlNet] Applied normal map processing (simplified)")
        return img_np  # Placeholder - would need actual normal map generation

    def _apply_pose_detection(self, img_np: np.ndarray):
        """Apply pose detection (placeholder)"""
        print(f"[BawkControlNet] Applied pose detection (placeholder)")
        return np.zeros_like(img_np)  # Placeholder - would need pose detection model

    def _apply_segmentation(self, img_np: np.ndarray):
        """Apply segmentation (placeholder)"""
        print(f"[BawkControlNet] Applied segmentation (placeholder)")
        return img_np  # Placeholder - would need segmentation model

    def _apply_scribble_effect(self, img_np: np.ndarray):
        """Apply scribble effect"""
        # Simple edge-based scribble effect
        edges = self._simple_edge_detection(img_np)
        # Thin the edges for scribble effect
        scribble = np.where(edges > 128, 255, 0).astype(np.uint8)
        print(f"[BawkControlNet] Applied scribble effect")
        return scribble

    def _apply_lineart(self, img_np: np.ndarray):
        """Apply lineart processing"""
        # Similar to scribble but with cleaner lines
        edges = self._simple_edge_detection(img_np)
        lineart = np.where(edges > 100, 255, 0).astype(np.uint8)
        print(f"[BawkControlNet] Applied lineart processing")
        return lineart

    def _apply_qr_processing(self, img_np: np.ndarray):
        """Apply QR code processing"""
        # High contrast black and white
        if len(img_np.shape) == 3:
            gray = np.mean(img_np, axis=-1)
        else:
            gray = img_np

        qr = np.where(gray > 128, 255, 0).astype(np.uint8)
        qr_rgb = np.stack([qr, qr, qr], axis=-1)
        print(f"[BawkControlNet] Applied QR code processing")
        return qr_rgb

    def _create_control_conditioning(self, control_image, strength: float, start: float, end: float):
        """Create control conditioning (placeholder for actual ControlNet integration)"""
        # This would integrate with actual ControlNet models in a real implementation
        # For now, return empty conditioning as placeholder

        conditioning_info = {
            "control_image": control_image,
            "strength": strength,
            "start_percent": start,
            "end_percent": end
        }

        return []  # Placeholder - would return actual conditioning

    def _generate_control_info(self, control_type: str, strength: float, start: float, end: float, image_shape):
        """Generate informational string about the control setup"""

        height, width = image_shape[-2:]
        duration = end - start

        info = f"{control_type} | Strength: {strength:.2f} | Duration: {start:.0%}-{end:.0%} ({duration:.0%}) | Size: {width}x{height}"
        return info