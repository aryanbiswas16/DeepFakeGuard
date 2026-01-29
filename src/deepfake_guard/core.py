from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

from .models.dinov3.detector import Detector
from .utils.face_crop import FaceCropper
from .utils.preprocess import simulate_compression, stack_frames
from .utils.video_io import read_video_frames
from .utils.weights import load_weights


@dataclass(frozen=True)
class ModalityResult:
    score: float
    label: str
    details: Dict[str, Any]


DetectorFn = Callable[[str], Dict[str, Any]]


class DeepfakeGuard:
    """Orchestrates multimodal deepfake detection pipelines."""

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing DeepfakeGuard on device: {self.device}")

        self._pipelines: Dict[str, DetectorFn] = {}
        self._model_info: Dict[str, str] = {}

        # Visual pipeline (DINOv3)
        self.visual_detector = Detector(device=self.device)
        self.face_cropper = FaceCropper(device=self.device, margin_px=0.5, padding_ratio=0.3)
        self.visual_weights_loaded = False

        if weights_path:
            self.load_visual_weights(weights_path)

        self.register_modality("visual", self._run_visual_analysis, "DINO v3 deepfake detection")

    def register_modality(self, name: str, pipeline: DetectorFn, description: str) -> None:
        """Register an additional modality pipeline.

        The pipeline receives a `video_path` and returns a dict payload.
        """
        self._pipelines[name] = pipeline
        self._model_info[name] = description

    def load_visual_weights(self, path: str) -> None:
        if os.path.exists(path):
            load_weights(self.visual_detector, path)
            self.visual_weights_loaded = True
            print(f"Visual weights loaded from {path}")
            return

        print(f"Warning: Weights file not found at {path}. Model initialized with random weights.")

    def detect_video(self, video_path: str) -> Dict[str, Any]:
        """Run the full detection pipeline on a video file."""
        result: Dict[str, Any] = {
            "model_info": {"version": "1.0.0", **self._model_info},
            "overall_label": "UNKNOWN",
            "overall_score": 0.0,
            "modality_results": {},
            "errors": [],
        }

        if not os.path.exists(video_path):
            result["errors"].append(f"Video file not found: {video_path}")
            return result

        for name, pipeline in self._pipelines.items():
            modality_res = pipeline(video_path)
            if "error" in modality_res:
                result["errors"].append(modality_res["error"])
                continue

            result["modality_results"][name] = modality_res

        overall_score = self._aggregate_scores(result["modality_results"])
        result["overall_score"] = overall_score
        result["overall_label"] = "FAKE" if overall_score > 0.5 else "REAL"

        return result

    def _aggregate_scores(self, modality_results: Dict[str, Dict[str, Any]]) -> float:
        scores = [res.get("score") for res in modality_results.values() if isinstance(res, dict)]
        scores = [s for s in scores if isinstance(s, (int, float))]
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def _run_visual_analysis(self, video_path: str) -> Dict[str, Any]:
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
        input_tensor = stack_frames(processed_frames, self.visual_detector.preprocess, self.device)

        det_result = self.visual_detector.predict_video(input_tensor)
        det_result = det_result.__dict__ if hasattr(det_result, "__dict__") else det_result

        score = float(det_result.get("score", 0.0))
        label = det_result.get("label", "UNKNOWN")
        details = det_result.get("details", {})

        return ModalityResult(score=score, label=label, details=details).__dict__
