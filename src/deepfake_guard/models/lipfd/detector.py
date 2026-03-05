"""
LipFD Detector — DeepFakeGuard integration.

Wraps the LipFD model (NeurIPS 2024) as a drop-in detector backend for the
DeepFakeGuard multimodal deepfake detection toolkit.  Detects lip-sync
deepfakes by exploiting temporal inconsistency between audio signals and
visual lip movements.  Pretrained on the AVLips dataset.

Original Paper:
    Liu et al., "Lips Are Lying: Spotting the Temporal Inconsistency
    between Audio and Visual in Lip-Syncing DeepFakes", NeurIPS 2024.
    https://arxiv.org/abs/2401.15668

Usage::

    from deepfake_guard.models.lipfd import LipFDDetector

    det = LipFDDetector(weights_path="src/deepfake_guard/weights/lipfd_ckpt.pth")
    result = det.predict_video("video.mp4")
    print(result["overall_label"], result["overall_score"])

The detector follows the DeepFakeGuard result format so it can be used
interchangeably with DINOv3 and D3 detectors.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .model import LipFD, VALID_ARCH_NAMES, build_model
from .preprocessing import (
    has_ffmpeg_support,
    has_librosa_support,
    preprocess_video,
)


# Default CLIP architecture used by the pretrained LipFD checkpoint
DEFAULT_ARCH = "CLIP:ViT-L/14"

# Detection threshold (logit space → sigmoid ≥ 0.5 is FAKE)
DEFAULT_THRESHOLD = 0.5


class LipFDDetector:
    """
    DeepFakeGuard-compatible detector backend powered by LipFD.

    Detects lip-sync deepfakes using the LipFD model (NeurIPS 2024),
    trained on the AVLips dataset. Exploits temporal inconsistency
    between audio signals and visual lip movements.

    Parameters
    ----------
    weights_path : str | Path | None
        Path to a LipFD checkpoint (``*.pth``).  The checkpoint is expected
        to contain a ``"model"`` key with the state dict.  If *None*, the
        model runs with random weights (useful for testing the pipeline).
    arch : str
        CLIP architecture string, e.g. ``"CLIP:ViT-L/14"``
        (must match the architecture the checkpoint was trained with).
    threshold : float
        Sigmoid probability threshold for labelling a video as FAKE.
    device : str | None
        Torch device.  ``None`` → auto-detect (CUDA if available).
    n_extract : int
        Number of frame groups to sample from the video during
        preprocessing (more = slower but more robust).
    max_composites : int | None
        If set, cap the number of composite images processed per video
        to limit memory / latency.
    """

    DETECTOR_TYPE = "lipfd"

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        arch: str = DEFAULT_ARCH,
        threshold: float = DEFAULT_THRESHOLD,
        device: Optional[str] = None,
        n_extract: int = 10,
        max_composites: Optional[int] = None,
    ):
        self.arch = arch
        self.threshold = threshold
        self.n_extract = n_extract
        self.max_composites = max_composites

        # Resolve device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Build model
        self.model: LipFD = build_model(arch)

        # Load weights if provided
        self._weights_loaded = False
        if weights_path is not None:
            self._load_weights(weights_path)

        self.model.eval()
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def _load_weights(self, weights_path: str | Path) -> None:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"LipFD checkpoint not found: {path}"
            )

        state_dict = torch.load(str(path), map_location="cpu")

        # Handle both raw state-dicts and wrapped checkpoints
        if "model" in state_dict:
            state_dict = state_dict["model"]

        self.model.load_state_dict(state_dict, strict=False)
        self._weights_loaded = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_video(
        self,
        video_path: str,
        batch_size: int = 4,
    ) -> Dict[str, Any]:
        """
        Analyse a video for lip-sync deepfake manipulation.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        batch_size : int
            Number of composite samples to process in one forward pass.
            Reduce if running out of GPU memory.

        Returns
        -------
        result : dict
            DeepFakeGuard-standardised result dictionary with keys:
            ``overall_label``, ``overall_score``, ``model_info``,
            ``modality_results``, ``errors``.
        """
        errors: List[str] = []

        if not self._weights_loaded:
            errors.append(
                "No pretrained weights loaded — predictions will be unreliable."
            )

        if not has_librosa_support():
            return self._error_result(
                "Missing dependency: librosa. LipFD requires librosa for "
                "audio preprocessing. Install with: pip install librosa"
            )

        if not has_ffmpeg_support():
            return self._error_result(
                "Missing dependency: ffmpeg not found on PATH. LipFD "
                "requires ffmpeg to extract audio from video."
            )

        # --- Preprocess ---
        try:
            full_imgs, crops, has_audio = preprocess_video(
                video_path,
                n_extract=self.n_extract,
                max_composites=self.max_composites,
            )
        except Exception as e:
            return self._error_result(str(e))

        if full_imgs is None or crops is None:
            return self._error_result(
                "Video could not be preprocessed (too short or unreadable)."
            )

        if not has_audio:
            return self._error_result(
                "No usable audio track found in this video. LipFD requires "
                "audio to detect lip-sync deepfakes — cannot produce a "
                "reliable prediction."
            )

        num_samples = full_imgs.shape[0]

        # Safety: cap sample count to prevent GPU OOM on very long videos
        _MAX_SAMPLES = 20
        if num_samples > _MAX_SAMPLES:
            full_imgs = full_imgs[:_MAX_SAMPLES]
            crops = [
                [c[:_MAX_SAMPLES] for c in scale_crops]
                for scale_crops in crops
            ]
            num_samples = _MAX_SAMPLES

        # --- Inference (in mini-batches) ---
        all_scores: List[float] = []

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)

            batch_imgs = full_imgs[start:end].to(self.device)
            batch_crops = [
                [crops[s][f][start:end].to(self.device) for f in range(len(crops[0]))]
                for s in range(len(crops))
            ]

            # Global CLIP features
            features = self.model.get_features(batch_imgs).to(self.device)

            # Region-aware prediction
            pred_logits, _, _ = self.model(batch_crops, features)
            probs = torch.sigmoid(pred_logits).flatten().cpu().tolist()
            all_scores.extend(probs)

            # Free GPU memory between batches to prevent OOM
            del batch_imgs, batch_crops, features, pred_logits
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # --- Aggregate scores ---
        if not all_scores:
            return self._error_result("No valid predictions produced.")

        scores_arr = np.array(all_scores)
        overall_score = float(np.mean(scores_arr))
        overall_label = "FAKE" if overall_score >= self.threshold else "REAL"

        # Per-sample labels for detail
        sample_labels = [
            "FAKE" if s >= self.threshold else "REAL" for s in all_scores
        ]
        fake_ratio = sum(1 for l in sample_labels if l == "FAKE") / len(sample_labels)

        return {
            "overall_label": overall_label,
            "overall_score": round(overall_score, 4),
            "model_info": {
                "detector_type": self.DETECTOR_TYPE,
                "architecture": self.arch,
                "weights_loaded": self._weights_loaded,
                "threshold": self.threshold,
                "paper": "Liu et al., NeurIPS 2024",
            },
            "modality_results": {
                "audio_visual": {
                    "score": round(overall_score, 4),
                    "label": overall_label,
                    "details": {
                        "method": "LipFD — lip-audio temporal inconsistency",
                        "num_samples": num_samples,
                        "sample_scores": [round(s, 4) for s in all_scores],
                        "fake_ratio": round(fake_ratio, 4),
                        "score_std": round(float(np.std(scores_arr)), 4),
                        "score_min": round(float(np.min(scores_arr)), 4),
                        "score_max": round(float(np.max(scores_arr)), 4),
                    },
                }
            },
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _error_result(message: str) -> Dict[str, Any]:
        """Return a result dict indicating an error."""
        return {
            "overall_label": "ERROR",
            "overall_score": 0.0,
            "model_info": {"detector_type": "lipfd"},
            "modality_results": {},
            "errors": [message],
        }

    def __repr__(self) -> str:
        return (
            f"LipFDDetector(arch={self.arch!r}, "
            f"threshold={self.threshold}, "
            f"device={self.device}, "
            f"weights_loaded={self._weights_loaded})"
        )
