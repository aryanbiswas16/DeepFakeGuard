from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Literal

import torch

from .models.dinov3.detector import Detector as DINOv3Detector
from .models.resnet18.detector import ResNet18Detector, detect_video_resnet18
from .models.ivyfake.detector import IvyFakeDetector
from .types import ModalityResult
from .utils.face_crop import FaceCropper
from .utils.preprocess import simulate_compression, stack_frames
from .utils.video_io import read_video_frames
from .utils.weights import load_weights


DetectorFn = Callable[[str], Dict[str, Any]]


class DeepfakeGuard:
    """
    Orchestrates multimodal deepfake detection pipelines.
    
    Supports multiple detector backends:
    - "dinov3": DINOv3-based detector (face cropping, 0.88+ AUROC)
    - "resnet18": ResNet18-based detector (full frames, pretrained ImageNet)
    - "ivyfake": IvyFake detector (CLIP-based with temporal/spatial analysis)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        detector_type: Literal["dinov3", "resnet18", "ivyfake"] = "dinov3"
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector_type = detector_type
        print(f"Initializing DeepfakeGuard ({detector_type}) on device: {self.device}")

        self._pipelines: Dict[str, DetectorFn] = {}
        self._model_info: Dict[str, str] = {}
        
        # Visual detector (switchable)
        self.visual_detector = None
        self.face_cropper = None
        self.resnet_detector = None
        self.ivyfake_detector = None
        self.visual_weights_loaded = False

        if detector_type == "dinov3":
            self._init_dinov3(weights_path)
        elif detector_type == "resnet18":
            self._init_resnet18()
        elif detector_type == "ivyfake":
            self._init_ivyfake()
        else:
            raise ValueError(f"Unknown detector_type: {detector_type}")

        # Register visual modality
        self.register_modality(
            "visual",
            self._run_visual_analysis,
            f"{detector_type.upper()} deepfake detection"
        )

    def _init_dinov3(self, weights_path: Optional[str]) -> None:
        """Initialize DINOv3-based detector."""
        self.visual_detector = DINOv3Detector(device=self.device)
        self.face_cropper = FaceCropper(
            device=self.device,
            margin_px=0.5,
            padding_ratio=0.3
        )
        
        if weights_path:
            self.load_visual_weights(weights_path)

    def _init_resnet18(self) -> None:
        """Initialize ResNet18-based detector (pretrained ImageNet weights)."""
        self.resnet_detector = ResNet18Detector(device=self.device)
        # Note: Uses pretrained ImageNet weights, no custom weights to load
        self.visual_weights_loaded = True
        print("ResNet18 detector initialized (pretrained ImageNet weights)")

    def _init_ivyfake(self) -> None:
        """Initialize IvyFake detector (CLIP-based with explainable features)."""
        self.ivyfake_detector = IvyFakeDetector(device=self.device)
        # Uses pretrained CLIP weights, no custom weights to load
        self.visual_weights_loaded = True
        print("IvyFake detector initialized (CLIP-ViT-B/32 backbone)")

    def set_detector(self, detector_type: Literal["dinov3", "resnet18", "ivyfake"], weights_path: Optional[str] = None) -> None:
        """
        Switch detector backend.
        
        Args:
            detector_type: "dinov3", "resnet18", or "ivyfake"
            weights_path: Path to DINOv3 weights (only used for dinov3)
        """
        if detector_type == self.detector_type:
            print(f"Already using {detector_type} detector")
            return
            
        print(f"Switching detector from {self.detector_type} to {detector_type}")
        self.detector_type = detector_type
        
        # Clear existing pipelines
        self._pipelines.clear()
        self._model_info.clear()
        
        # Reset detectors
        self.visual_detector = None
        self.face_cropper = None
        self.resnet_detector = None
        self.ivyfake_detector = None
        self.visual_weights_loaded = False
        
        # Reinitialize
        if detector_type == "dinov3":
            self._init_dinov3(weights_path)
        elif detector_type == "resnet18":
            self._init_resnet18()
        elif detector_type == "ivyfake":
            self._init_ivyfake()
            
        # Re-register visual modality
        self.register_modality(
            "visual",
            self._run_visual_analysis,
            f"{detector_type.upper()} deepfake detection"
        )

    def register_modality(self, name: str, pipeline: DetectorFn, description: str) -> None:
        """Register an additional modality pipeline."""
        self._pipelines[name] = pipeline
        self._model_info[name] = description

    def load_visual_weights(self, path: str) -> None:
        """Load weights for DINOv3 detector."""
        if self.detector_type != "dinov3":
            print(f"Warning: Cannot load weights for {self.detector_type} detector")
            return
            
        if os.path.exists(path):
            load_weights(self.visual_detector, path)
            self.visual_weights_loaded = True
            print(f"Visual weights loaded from {path}")
            return

        print(f"Warning: Weights file not found at {path}")

    def detect_video(self, video_path: str) -> Dict[str, Any]:
        """Run the full detection pipeline on a video file."""
        result: Dict[str, Any] = {
            "model_info": {
                "version": "1.2.0",
                "detector_type": self.detector_type,
                **self._model_info
            },
            "overall_label": "UNKNOWN",
            "overall_score": 0.0,
            "modality_results": {},
            "errors": [],
        }

        if not os.path.exists(video_path):
            result["errors"].append(f"Video file not found: {video_path}")
            return result

        for name, pipeline in self._pipelines.items():
            try:
                modality_res = pipeline(video_path)
                if "error" in modality_res:
                    result["errors"].append(modality_res["error"])
                    continue
                result["modality_results"][name] = modality_res
            except Exception as e:
                result["errors"].append(f"Error in {name}: {str(e)}")

        overall_score = self._aggregate_scores(result["modality_results"])
        result["overall_score"] = overall_score
        result["overall_label"] = "FAKE" if overall_score > 0.5 else "REAL"

        return result

    def _aggregate_scores(self, modality_results: Dict[str, Dict[str, Any]]) -> float:
        """Aggregate scores from all modalities."""
        scores = [res.get("score") for res in modality_results.values() if isinstance(res, dict)]
        scores = [s for s in scores if isinstance(s, (int, float))]
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def _run_visual_analysis(self, video_path: str) -> Dict[str, Any]:
        """Run visual analysis based on current detector type."""
        if self.detector_type == "dinov3":
            return self._run_dinov3_analysis(video_path)
        elif self.detector_type == "resnet18":
            return self._run_resnet18_analysis(video_path)
        elif self.detector_type == "ivyfake":
            return self._run_ivyfake_analysis(video_path)
        else:
            return {"error": f"Unknown detector type: {self.detector_type}"}

    def _run_dinov3_analysis(self, video_path: str) -> Dict[str, Any]:
        """DINOv3-based visual analysis with face cropping."""
        try:
            frames = read_video_frames(video_path)
        except Exception as exc:
            return {"error": f"Error reading video: {exc}"}

        if not frames:
            return {"error": "Could not read frames from video"}

        cropped_frames = []
        for frame in frames:
            crop = self.face_cropper.crop(frame, return_metadata=False)
            if crop:
                cropped_frames.append(crop)

        if not cropped_frames:
            return {"error": "No faces detected in video"}

        processed_frames = simulate_compression(cropped_frames)
        input_tensor = stack_frames(
            processed_frames,
            self.visual_detector.preprocess,
            self.device
        )

        det_result = self.visual_detector.predict_video(input_tensor)
        det_result = det_result.__dict__ if hasattr(det_result, "__dict__") else det_result

        score = float(det_result.get("score", 0.0))
        label = det_result.get("label", "UNKNOWN")
        details = det_result.get("details", {})
        details["detector_type"] = "dinov3"

        return ModalityResult(score=score, label=label, details=details).__dict__

    def _run_resnet18_analysis(self, video_path: str) -> Dict[str, Any]:
        """ResNet18-based visual analysis (full frames, no face cropping)."""
        result = self.resnet_detector.predict_video(video_path)
        return ModalityResult(
            score=result["score"],
            label=result["label"],
            details=result["details"]
        ).__dict__

    def _run_ivyfake_analysis(self, video_path: str) -> Dict[str, Any]:
        """IvyFake-based visual analysis (CLIP-based with temporal/spatial features)."""
        result = self.ivyfake_detector.predict_video(video_path)
        return ModalityResult(
            score=result["score"],
            label=result["label"],
            details=result["details"]
        ).__dict__