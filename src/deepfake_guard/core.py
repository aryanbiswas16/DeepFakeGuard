from __future__ import annotations

import gc
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Literal

logger = logging.getLogger(__name__)

import torch

from .models.dinov3.detector import Detector as DINOv3Detector
from .models.d3.detector import D3Detector
from .models.lipfd.detector import LipFDDetector
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
    - "d3": D3 detector (training-free, second-order temporal features)
    - "lipfd": LipFD detector (audio-visual lip-sync detection, NeurIPS 2024)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        detector_type: Literal["dinov3", "d3", "lipfd"] = "dinov3"
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector_type = detector_type


        self._pipelines: Dict[str, DetectorFn] = {}
        self._model_info: Dict[str, str] = {}
        
        # Visual detector (switchable)
        self.visual_detector = None
        self.face_cropper = None
        self.d3_detector = None
        self.lipfd_detector = None
        self.visual_weights_loaded = False

        if detector_type == "dinov3":
            self._init_dinov3(weights_path)
        elif detector_type == "d3":
            self._init_d3()
        elif detector_type == "lipfd":
            self._init_lipfd(weights_path)
        else:
            raise ValueError(f"Unknown detector_type: {detector_type}")

        # Register visual modality
        self.register_modality(
            "visual",
            self._run_visual_analysis,
            f"{detector_type.upper()} deepfake detection"
        )

    # Paths to the weights bundled inside the installed package
    _BUNDLED_DINOV3_WEIGHTS = Path(__file__).parent / "weights" / "dinov3_best_v3.pth"
    _BUNDLED_LIPFD_WEIGHTS  = Path(__file__).parent / "weights" / "lipfd_ckpt.pth"

    def _init_dinov3(self, weights_path: Optional[str] = None) -> None:
        """Initialize DINOv3-based detector.

        Weights are resolved in this order:
          1. Explicit ``weights_path`` argument (if provided and exists)
          2. Bundled weights shipped with the package
        """
        self.visual_detector = DINOv3Detector(device=self.device)
        self.face_cropper = FaceCropper(
            device=self.device,
            padding_ratio=0.3,
        )

        resolved = None
        if weights_path and os.path.exists(weights_path):
            resolved = weights_path
        elif self._BUNDLED_DINOV3_WEIGHTS.exists():
            resolved = str(self._BUNDLED_DINOV3_WEIGHTS)
        else:
            import warnings
            warnings.warn("No DINOv3 weights found — running with random initialisation.")

        if resolved:
            self.load_visual_weights(resolved)

    def _init_d3(self, encoder: str = "xclip-16") -> None:
        """Initialize D3 detector (training-free, second-order temporal features)."""
        self.d3_detector = D3Detector(encoder_name=encoder, device=self.device)
        self.visual_weights_loaded = True

    def _init_lipfd(self, weights_path: Optional[str] = None) -> None:
        """Initialize LipFD detector (audio-visual lip-sync deepfake detection, NeurIPS 2024).
        Weights are resolved automatically:
          1. Explicit ``weights_path`` argument (if provided and exists)
          2. Bundled weights at ``src/deepfake_guard/weights/lipfd_ckpt.pth``
          3. Auto-download from GitHub Releases (~1.68 GB, first run only)
        """
        from .utils.weights import resolve_lipfd_weights
        from .models.lipfd.detector import DEFAULT_ARCH
        resolved = resolve_lipfd_weights(weights_path)
        if resolved is None:
            import warnings
            warnings.warn(
                "LipFD weights not available — running without weights "
                "(accuracy will be poor). See README for download instructions."
            )
            # No weights available: use the smaller ViT-B/32 to save ~900 MB RAM.
            # Accuracy is equally poor regardless of CLIP variant without weights.
            arch = "CLIP:ViT-B/32"
        else:
            arch = DEFAULT_ARCH
        self.lipfd_detector = LipFDDetector(weights_path=resolved, device=self.device, arch=arch)
        self.visual_weights_loaded = resolved is not None

    def set_detector(
        self,
        detector_type: Literal["dinov3", "d3", "lipfd"],
        weights_path: Optional[str] = None,
    ) -> None:
        """
        Switch detector backend.
        
        Args:
            detector_type: "dinov3", "d3", or "lipfd"
            weights_path: Path to weights (used for dinov3 and lipfd)
        """
        if detector_type == self.detector_type:
            return
            
        self.detector_type = detector_type

        # Reset all detector handles
        self._pipelines.clear()
        self._model_info.clear()
        self.visual_detector = None
        self.face_cropper = None
        self.d3_detector = None
        self.lipfd_detector = None
        self.visual_weights_loaded = False
        
        # Reinitialize
        if detector_type == "dinov3":
            self._init_dinov3(weights_path)
        elif detector_type == "d3":
            self._init_d3()
        elif detector_type == "lipfd":
            self._init_lipfd(weights_path)
            
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
            import warnings
            warnings.warn(f"load_visual_weights is only applicable to the dinov3 detector (current: {self.detector_type}).")
            return
            
        if os.path.exists(path):
            load_weights(self.visual_detector, path)
            self.visual_weights_loaded = True
            return

        import warnings
        warnings.warn(f"Weights file not found: {path}")

    def detect_video(self, video_path: str) -> Dict[str, Any]:
        """Run the full detection pipeline on a video file."""
        result: Dict[str, Any] = {
            "model_info": {
                "version": "0.4.0",
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

    # ─── Per-detector trust priors ───────────────────────────────────────
    #
    # Domain awareness:
    #   DINOv3  – trained *strictly* on face-swap deepfakes (FaceForensics++,
    #             Celeb-DF).  Strong on face manipulation artefacts, but may
    #             false-positive on legitimate AI-generated video that wasn't
    #             designed to impersonate someone.
    #   D3      – training-free; measures second-order temporal volatility
    #             (ICCV 2025).  Complementary to DINOv3: good at catching
    #             AI-generated video, but can miss face-swap deepfakes that
    #             preserve natural motion.
    #   LipFD   – audio-visual lip-sync detector (NeurIPS 2024, 95.3% on
    #             AVLips).  Unique modality—only detector that uses audio.
    #             Trained on Wav2Lip / SadTalker / TalkLip / MakeItTalk.
    #             Irrelevant for videos without speech or face.
    #
    _DETECTOR_TRUST: Dict[str, float] = {
        "dinov3": 1.0,     # highest—trained, well-calibrated
        "lipfd":  0.9,     # NeurIPS 2024, strong on its domain
        "d3":     0.65,    # training-free, principled but noisier
    }

    # Which *forgery families* each detector is an authority on.
    _DETECTOR_DOMAIN: Dict[str, str] = {
        "dinov3": "face-swap deepfake",
        "lipfd":  "lip-sync deepfake",
        "d3":     "AI-generated video",
    }

    # ─── Single-detector aggregation (unchanged API) ───────────────────
    def _aggregate_scores(self, modality_results: Dict[str, Dict[str, Any]]) -> float:
        """Confidence-weighted aggregate across modalities.

        Each score is weighted by:
          trust_prior(detector) × certainty(score)

        where certainty = 2 * |score − 0.5|  (0 = uncertain, 1 = maximal).
        """
        weighted_sum = 0.0
        weight_total = 0.0
        for res in modality_results.values():
            if not isinstance(res, dict):
                continue
            score = res.get("score")
            if not isinstance(score, (int, float)):
                continue
            det_type = res.get("details", {}).get("detector_type", self.detector_type)
            trust = self._DETECTOR_TRUST.get(det_type, 0.5)
            certainty = abs(float(score) - 0.5) * 2.0
            weight = trust * (0.1 + 0.9 * certainty)
            weighted_sum += float(score) * weight
            weight_total += weight
        if weight_total == 0.0:
            return 0.0
        return float(weighted_sum / weight_total)

    # ─── Ensemble detection (multi-detector) ───────────────────────────
    @staticmethod
    def ensemble_detect_video(
        guards: Dict[str, "DeepfakeGuard"],
        video_path: str,
        threshold: float = 0.5,
        vlm_backend: str = "openai",
        vlm_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run all supplied detectors and produce a fused ensemble result.

        This is the *proper* ensemble entry-point.  It:
          1. Runs each detector sequentially (with memory cleanup between).
          2. Applies domain-aware trust × certainty weighting.
          3. Detects outlier detectors and down-weights them (veto logic).
          4. Checks cross-modal agreement / disagreement.
          5. Generates a natural-language explanation.

        Args:
            guards: ``{"dinov3": guard, "d3": guard, …}``
            video_path: Path to the video file.
            threshold: Score above this → FAKE.

        Returns:
            A rich result dict with ensemble verdict, per-detector
            results, agreement analysis, and explanation.
        """
        if not os.path.exists(video_path):
            return {"error": f"Video not found: {video_path}"}

        # ── 1. Run each detector (with cleanup) ──────────────────────
        per_detector: Dict[str, Dict[str, Any]] = {}
        for name, guard in guards.items():
            try:
                per_detector[name] = guard.detect_video(video_path)
            except Exception as exc:
                per_detector[name] = {
                    "overall_score": 0.5,
                    "overall_label": "ERROR",
                    "errors": [str(exc)],
                }
            # Free memory between detectors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Collect per-detector scores & labels
        scores: Dict[str, float] = {}
        labels: Dict[str, str] = {}
        for n, r in per_detector.items():
            scores[n] = r.get("overall_score", 0.5)
            labels[n] = r.get("overall_label", "UNKNOWN")

        # ── 1b. Applicability gating ────────────────────────────────
        applicability: Dict[str, Dict[str, Any]] = {}
        for name, r in per_detector.items():
            applicability[name] = DeepfakeGuard._assess_applicability(name, r)

        # ── 2. Outlier veto ───────────────────────────────────────────
        #  If exactly one detector is highly confident FAKE (>0.8) but
        #  ALL others are confidently REAL (<0.35), the lone detector is
        #  likely a domain mismatch (e.g. DINOv3 seeing "face artefacts"
        #  in a legitimate AI video).  Down-weight it heavily.
        outliers: Dict[str, bool] = {n: False for n in scores}
        if len(scores) >= 3:
            for candidate, cand_score in scores.items():
                if cand_score <= 0.8:
                    continue
                others = {k: v for k, v in scores.items() if k != candidate}
                others_strongly_real = all(s < 0.35 for s in others.values())
                others_have_opinion = any(
                    abs(s - 0.5) > 0.1 for s in others.values()
                )
                if others_strongly_real and others_have_opinion:
                    outliers[candidate] = True

        # ── 3. Domain-aware weighted fusion ───────────────────────────
        trust_map = DeepfakeGuard._DETECTOR_TRUST
        weighted_sum = 0.0
        weight_total = 0.0
        contributions: Dict[str, Dict[str, float]] = {}

        for name, score in scores.items():
            if labels[name] == "ERROR":
                contributions[name] = {
                    "score": score,
                    "weight": 0.0,
                    "outlier": False,
                    "trust": trust_map.get(name, 0.5),
                    "certainty": 0.0,
                    "applicability_factor": 0.0,
                    "applicability_level": "unavailable",
                    "applicability_reason": applicability.get(name, {}).get("reason", "Detector failed"),
                }
                continue
            trust = trust_map.get(name, 0.5)
            certainty = abs(score - 0.5) * 2.0
            weight = trust * (0.1 + 0.9 * certainty)

            app_meta = applicability.get(name, {})
            app_factor = float(app_meta.get("factor", 1.0))
            weight *= app_factor

            # Outlier penalty — multiply by 0.05 (95 % reduction)
            if outliers.get(name, False):
                weight *= 0.05

            weighted_sum += score * weight
            weight_total += weight
            contributions[name] = {
                "score": round(score, 4),
                "weight": round(weight, 4),
                "trust": trust,
                "certainty": round(certainty, 4),
                "outlier": outliers.get(name, False),
                "applicability_factor": round(app_factor, 4),
                "applicability_level": app_meta.get("level", "high"),
                "applicability_reason": app_meta.get("reason", "Fully applicable"),
            }

        raw_score = weighted_sum / weight_total if weight_total > 0 else 0.5

        # Gentle sigmoid sharpening (k=4): pushes scores away from 0.5
        # but keeps them interpretable—nothing like the k=200 insanity.
        k = 4.0
        try:
            ensemble_score = 1.0 / (1.0 + math.exp(-k * (raw_score - 0.5)))
        except OverflowError:
            ensemble_score = 0.0 if raw_score < 0.5 else 1.0

        ensemble_label = "FAKE" if ensemble_score > threshold else "REAL"

        # ── 4. Agreement analysis ─────────────────────────────────────
        non_error = {n: l for n, l in labels.items() if l != "ERROR"}
        unique_labels = set(non_error.values()) - {"UNKNOWN"}
        if len(unique_labels) <= 1 and len(non_error) > 1:
            agreement = "unanimous"
        elif len(unique_labels) == 2:
            fake_count = sum(1 for l in non_error.values() if l == "FAKE")
            real_count = sum(1 for l in non_error.values() if l == "REAL")
            if fake_count > real_count:
                agreement = "majority-fake"
            elif real_count > fake_count:
                agreement = "majority-real"
            else:
                agreement = "split"
        else:
            agreement = "inconclusive"

        # ── 5. Natural-language explanation ────────────────────────────
        explanation = DeepfakeGuard._build_ensemble_explanation(
            scores, labels, outliers, contributions, applicability,
            ensemble_score, ensemble_label, agreement, threshold,
        )

        # ── 5b. VLM semantic explainability (FAKE only, post-hoc) ─────
        vlm_explanation = None
        if ensemble_label == "FAKE":
            try:
                from deepfake_guard.explainability import VLMExplainer
                explainer = VLMExplainer(backend=vlm_backend, api_key=vlm_api_key)
                vlm_explanation = explainer.explain(video_path, ensemble_score)
            except Exception as exc:
                logger.debug("VLM explainability unavailable: %s", exc)
                vlm_explanation = {
                    "available": False,
                    "explanation": "VLM backend not configured",
                }

        # ── 6. Assemble result ────────────────────────────────────────
        all_errors = []
        for r in per_detector.values():
            all_errors.extend(r.get("errors", []))

        return {
            "overall_score": round(ensemble_score, 4),
            "overall_label": ensemble_label,
            "raw_score": round(raw_score, 4),
            "threshold": threshold,
            "agreement": agreement,
            "explanation": explanation,
            "vlm_explanation": vlm_explanation,
            "detector_results": per_detector,
            "contributions": contributions,
            "applicability": applicability,
            "scores": scores,
            "labels": labels,
            "outliers": outliers,
            "errors": all_errors,
            "model_info": {
                "version": "0.5.0",
                "mode": "ensemble",
                "detectors": list(guards.keys()),
                "fusion": "domain-aware trust × certainty × applicability with outlier veto",
            },
        }

    @staticmethod
    def _assess_applicability(detector_name: str, detector_result: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate whether a detector is applicable to this specific video."""
        if detector_result.get("overall_label") == "ERROR":
            err = "; ".join(detector_result.get("errors", []))
            return {
                "level": "unavailable",
                "factor": 0.0,
                "reason": err or "Detector failed for this video",
            }

        details = detector_result.get("modality_results", {}).get("visual", {}).get("details", {})

        if detector_name == "lipfd":
            av_details = detector_result.get("modality_results", {}).get("audio_visual", {}).get("details", {})
            n_samples = av_details.get("num_samples")
            if isinstance(n_samples, int):
                if n_samples >= 8:
                    return {"level": "high", "factor": 1.0, "reason": f"{n_samples} lip-sync samples analysed"}
                if n_samples >= 4:
                    return {"level": "medium", "factor": 0.75, "reason": f"Limited lip-sync samples ({n_samples})"}
                return {"level": "low", "factor": 0.45, "reason": f"Very few lip-sync samples ({n_samples})"}
            return {"level": "medium", "factor": 0.7, "reason": "Audio-visual samples available"}

        if detector_name == "d3":
            frame_count = details.get("frame_count")
            volatility = details.get("volatility")
            vol_threshold = details.get("threshold", 1.8)
            if isinstance(frame_count, int) and frame_count < 8:
                return {"level": "low", "factor": 0.5, "reason": f"Low frame count ({frame_count})"}
            if isinstance(volatility, (int, float)) and isinstance(vol_threshold, (int, float)) and vol_threshold > 0:
                motion_ratio = float(volatility) / float(vol_threshold)
                if motion_ratio < 0.45:
                    return {"level": "low", "factor": 0.45, "reason": "Very low temporal motion"}
                if motion_ratio < 0.75:
                    return {"level": "medium", "factor": 0.7, "reason": "Limited temporal motion"}
            return {"level": "high", "factor": 1.0, "reason": "Sufficient temporal dynamics"}

        if detector_name == "dinov3":
            frame_count = details.get("frame_count")
            if isinstance(frame_count, int):
                if frame_count >= 8:
                    return {"level": "high", "factor": 1.0, "reason": f"Faces found across {frame_count} frames"}
                if frame_count >= 4:
                    return {"level": "medium", "factor": 0.75, "reason": "Moderate face coverage"}
                return {"level": "low", "factor": 0.5, "reason": "Sparse/uncertain face coverage"}
            return {"level": "high", "factor": 1.0, "reason": "Face-based analysis available"}

        return {"level": "high", "factor": 1.0, "reason": "Applicable"}

    @staticmethod
    def _build_ensemble_explanation(
        scores: Dict[str, float],
        labels: Dict[str, str],
        outliers: Dict[str, bool],
        contributions: Dict[str, Dict[str, float]],
        applicability: Dict[str, Dict[str, Any]],
        ensemble_score: float,
        ensemble_label: str,
        agreement: str,
        threshold: float,
    ) -> str:
        """Generate a human-readable explanation of the ensemble verdict."""
        domain = DeepfakeGuard._DETECTOR_DOMAIN
        parts: List[str] = []

        # Headline
        certainty_pct = abs(ensemble_score - 0.5) * 200
        parts.append(
            f"The ensemble verdict is {ensemble_label} "
            f"(score {ensemble_score:.1%}, certainty {certainty_pct:.0f}%)."
        )

        # Agreement summary
        if agreement == "unanimous":
            parts.append("All detectors agree on this verdict.")
        elif agreement == "split":
            parts.append(
                "Detectors are evenly split — treat this result with caution."
            )
        elif agreement.startswith("majority"):
            majority_side = agreement.split("-")[1].upper()
            dissenters = [
                n.upper() for n, l in labels.items()
                if l != majority_side and l not in ("UNKNOWN", "ERROR")
            ]
            if dissenters:
                parts.append(
                    f"Majority says {majority_side}, "
                    f"but {', '.join(dissenters)} dissent{'s' if len(dissenters) == 1 else ''}."
                )

        # Per-detector rationale
        for name in sorted(scores.keys()):
            s = scores[name]
            lab = labels[name]
            dom = domain.get(name, "unknown")
            is_outlier = outliers.get(name, False)

            if lab == "ERROR":
                parts.append(f"{name.upper()} ({dom}): skipped — could not analyse this video.")
                continue

            direction = "fake" if s > 0.5 else "real"
            strength = abs(s - 0.5) * 2.0
            if strength > 0.7:
                adverb = "strongly"
            elif strength > 0.3:
                adverb = "moderately"
            else:
                adverb = "weakly"

            line = f"{name.upper()} ({dom}): {adverb} indicates {direction} ({s:.1%})."

            if is_outlier:
                line += (
                    " This contradicts all other detectors and was "
                    "down-weighted as a likely domain mismatch."
                )

            app_meta = applicability.get(name, {})
            app_level = str(app_meta.get("level", "high"))
            app_reason = str(app_meta.get("reason", ""))
            if app_level in ("low", "medium"):
                line += f" Applicability is {app_level}: {app_reason}."

            # Domain-specific caveats
            if name == "dinov3" and lab == "FAKE":
                line += (
                    " Note: DINOv3 was trained on face-swap deepfakes "
                    "(not AI-generated video); a FAKE label here "
                    "specifically indicates face manipulation artefacts."
                )
            elif name == "lipfd" and lab == "REAL":
                line += (
                    " LipFD only detects lip-sync forgeries; a REAL "
                    "result does not rule out other manipulation types."
                )
            elif name == "d3" and lab == "FAKE":
                line += (
                    " D3 detects AI-generated content by motion "
                    "volatility; this suggests the video has "
                    "unnaturally smooth temporal dynamics."
                )

            parts.append(line)

        return " ".join(parts)

    def _run_visual_analysis(self, video_path: str) -> Dict[str, Any]:
        """Run visual analysis based on current detector type."""
        if self.detector_type == "dinov3":
            return self._run_dinov3_analysis(video_path)
        elif self.detector_type == "d3":
            return self._run_d3_analysis(video_path)
        elif self.detector_type == "lipfd":
            return self._run_lipfd_analysis(video_path)
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
        # Free original frames — we only need crops from here
        del frames

        if not cropped_frames:
            return {"error": "No faces detected in video"}

        processed_frames = simulate_compression(cropped_frames)
        del cropped_frames  # free pre-compression copies

        # Build tensor inside no_grad to avoid storing unnecessary
        # computation graphs that bloat memory.
        with torch.no_grad():
            input_tensor = stack_frames(
                processed_frames,
                self.visual_detector.preprocess,
                self.device
            )
            del processed_frames

            det_result = self.visual_detector.predict_video(input_tensor)
            del input_tensor

        # Free GPU cache after inference
        if self.device == "cuda":
            torch.cuda.empty_cache()

        det_result = det_result.__dict__ if hasattr(det_result, "__dict__") else det_result

        score = float(det_result.get("score", 0.0))
        label = det_result.get("label", "UNKNOWN")
        details = det_result.get("details", {})
        details["detector_type"] = "dinov3"

        return ModalityResult(score=score, label=label, details=details).__dict__

    def _run_d3_analysis(self, video_path: str) -> Dict[str, Any]:
        """D3-based visual analysis (training-free, second-order temporal features)."""
        result = self.d3_detector.predict_video(video_path)
        # Free GPU cache after inference
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return ModalityResult(
            score=result["score"],
            label=result["label"],
            details=result["details"]
        ).__dict__

    def _run_lipfd_analysis(self, video_path: str) -> Dict[str, Any]:
        """LipFD audio-visual lip-sync deepfake analysis (NeurIPS 2024)."""
        if self.lipfd_detector is None:
            return {"error": "LipFD detector not initialised"}

        try:
            result = self.lipfd_detector.predict_video(video_path)
        except Exception as exc:
            return {"error": f"LipFD analysis failed: {exc}"}
        finally:
            # Free GPU cache after inference regardless of success/failure
            if self.device == "cuda":
                torch.cuda.empty_cache()

        if result.get("overall_label") == "ERROR":
            errors = result.get("errors", ["Unknown LipFD error"])
            return {"error": "; ".join(errors)}

        score = float(result.get("overall_score", 0.0))
        label = result.get("overall_label", "UNKNOWN")
        details = result.get("modality_results", {}).get("audio_visual", {}).get("details", {})
        details["detector_type"] = "lipfd"
        details["model"] = result.get("model_info", {})

        return ModalityResult(score=score, label=label, details=details).__dict__