"""
IvyFake-based video detector (CLIP-based explainable AIGC detection)
Uses CLIP vision encoder with temporal and spatial artifact analyzers
Original: https://github.com/HamzaKhan760/IvyFakeGenDetector
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

from ...types import ModalityResult


class TemporalArtifactAnalyzer(nn.Module):
    """Analyzes temporal inconsistencies in video frames."""
    
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
        
    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_features: (batch, num_frames, embed_dim)
        Returns:
            temporal_features: (batch, embed_dim)
        """
        # Transpose for Conv1d: (batch, embed_dim, num_frames)
        x = frame_features.transpose(1, 2)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)  # Back to (batch, num_frames, embed_dim)
        
        # Self-attention across frames
        x = x.transpose(0, 1)  # (num_frames, batch, embed_dim)
        attn_out, _ = self.attention(x, x, x)
        attn_out = attn_out.transpose(0, 1)  # (batch, num_frames, embed_dim)
        
        # Global average pooling
        return attn_out.mean(dim=1)


class SpatialArtifactAnalyzer(nn.Module):
    """Analyzes spatial artifacts in individual frames."""
    
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_features: (batch, num_frames, embed_dim)
        Returns:
            spatial_features: (batch, embed_dim)
        """
        # Apply attention and pool
        attention_weights = self.spatial_attention(frame_features)
        attended_features = frame_features * attention_weights
        return attended_features.mean(dim=1)


class IvyXDetector(nn.Module):
    """
    Explainable AIGC detector using a frozen CLIP vision backbone.

    The temporal and spatial heads (TemporalArtifactAnalyzer,
    SpatialArtifactAnalyzer, fusion, classifier) are intentionally bypassed
    during inference because they are randomly initialised and have no trained
    weights.  Instead, the detector computes two principled signals directly
    from the frozen CLIP embeddings:

      * Temporal consistency  — mean cosine similarity between consecutive
        frame embeddings.  Real videos have high frame-to-frame semantic
        continuity; AI-generated artefacts cause dips.
      * Spatial anomaly       — per-frame deviation from the clip-mean
        embedding across the sequence (Mahalanobis-like outlier score).

    Both signals are fused via a simple calibrated sigmoid to produce a
    P(fake) value in [0, 1].
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        num_classes: int = 2,
        embed_dim: int = 512,
        freeze_backbone: bool = True,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.device = device
        
        # Load CLIP model as vision backbone
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Temporal and spatial artifact analyzers
        self.temporal_analyzer = TemporalArtifactAnalyzer(embed_dim)
        self.spatial_analyzer = SpatialArtifactAnalyzer(embed_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head
        self.classifier = nn.Linear(embed_dim // 2, num_classes)
        
        self.to(device)
        self.eval()
        
    def extract_frame_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from frames using CLIP vision encoder."""
        vision_outputs = self.clip_model.vision_model(pixel_values=images)
        return vision_outputs.pooler_output
    
    # Sigmoid sharpness — controls how fast P(fake) transitions around the
    # temporal-consistency decision boundary.  Tuned so that a ~0.05 drop in
    # mean cosine similarity below the boundary maps to ~0.75 P(fake).
    _SIG_K: float = 20.0
    _TEMPORAL_BOUNDARY: float = 0.85   # mean cos-sim below this → suspicious

    def forward(
        self,
        images: torch.Tensor,
        return_explanations: bool = False
    ) -> Dict[str, Any]:
        """
        Args:
            images: ``(batch, num_frames, C, H, W)`` or ``(batch, C, H, W)``

        Returns:
            dict with keys ``fake_scores`` ``[B]``, ``temporal_sim`` ``[B]``,
            ``spatial_anomaly`` ``[B]``, and ``per_frame_fake_probs``.
        """
        if images.dim() == 5:  # video
            batch_size, num_frames = images.shape[:2]
            images_flat = images.view(-1, *images.shape[2:])
        else:  # single frame batch
            batch_size = images.shape[0]
            num_frames = 1
            images_flat = images

        # ── frozen CLIP features ─────────────────────────────────────────────
        raw_feats = self.extract_frame_features(images_flat)          # (B*T, D)
        feats = torch.nn.functional.normalize(raw_feats, dim=-1)      # L2-normalise
        feats = feats.view(batch_size, num_frames, -1)                 # (B, T, D)

        # ── temporal signal: mean consecutive cosine similarity ──────────────
        if num_frames > 1:
            cos_sim = (feats[:, :-1, :] * feats[:, 1:, :]).sum(-1)    # (B, T-1)
            temporal_sim = cos_sim.mean(-1)                            # (B,)
            # Per-frame fake proxy: deviation of each sim from boundary
            # (low similarity = higher per-frame anomaly)
            per_frame_sim = torch.cat([
                cos_sim[:, :1],   # pad first frame
                cos_sim
            ], dim=1)  # (B, T)
        else:
            temporal_sim = torch.ones(batch_size, device=images.device)
            per_frame_sim = torch.ones(batch_size, 1, device=images.device)

        # ── spatial signal: per-frame distance from sequence mean ────────────
        seq_mean = feats.mean(dim=1, keepdim=True)                     # (B, 1, D)
        spatial_dist = torch.norm(feats - seq_mean, dim=-1).mean(-1)   # (B,)
        # Normalise spatial_dist to [0, 1] via tanh
        spatial_anomaly = torch.tanh(spatial_dist * 2.0)

        # ── fuse: low temporal consistency + high spatial anomaly → fake ─────
        # Invert temporal_sim so that high sim = low fake score
        temporal_fake = 1.0 - temporal_sim          # (B,)  ∈ [0, 1] approx
        combined = 0.7 * temporal_fake + 0.3 * spatial_anomaly

        # Sigmoid centred at 0.5 * (1 - boundary) to calibrate
        fake_scores = torch.sigmoid(self._SIG_K * (combined - (1.0 - self._TEMPORAL_BOUNDARY)))

        # Per-frame fake probability (sigmoid of per-frame sim deviation)
        per_frame_fake = torch.sigmoid(
            self._SIG_K * ((1.0 - per_frame_sim) - (1.0 - self._TEMPORAL_BOUNDARY))
        )  # (B, T)

        return {
            "fake_scores": fake_scores,
            "temporal_sim": temporal_sim,
            "spatial_anomaly": spatial_anomaly,
            "per_frame_fake_probs": per_frame_fake,
        }

    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (predictions [B], fake_probs [B]) compatible with old callers."""
        self.eval()
        with torch.no_grad():
            output = self.forward(images)
            fake_p = output["fake_scores"]           # P(fake) in [0,1]
            probs = torch.stack([1 - fake_p, fake_p], dim=-1)  # [B, 2]
            preds = (fake_p > 0.5).long()
        return preds, probs


class IvyFakeDetector:
    """
    IvyFake-based video detector for DeepfakeGuard.
    
    Features:
    - CLIP-based vision encoder
    - Temporal artifact analysis for videos
    - Spatial artifact analysis for frames
    - Explainable outputs
    """
    
    def __init__(
        self,
        device: str = "cpu",
        num_frames: int = 16,
        threshold: float = 0.5,
        model_name: str = "openai/clip-vit-base-patch32"
    ):
        self.device = device
        self.num_frames = num_frames
        self.threshold = threshold
        
        # Initialize IvyX model
        self.model = IvyXDetector(
            model_name=model_name,
            device=device
        )
        
        print(f"Loaded IvyFake detector on {device}")
    
    def extract_frames(self, video_path: str) -> list:
        """Uniformly sample frames from video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return []
            
        step = max(total // self.num_frames, 1)
        
        idx = 0
        while cap.isOpened() and len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            idx += 1
        
        cap.release()
        return frames
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for CLIP input."""
        # Resize to 224x224 (CLIP's expected input)
        frame_resized = cv2.resize(frame, (224, 224))
        
        # Convert to tensor and normalize (CLIP normalization)
        frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC to CHW
        
        # CLIP normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        frame_tensor = (frame_tensor - mean) / std
        
        return frame_tensor
    
    @torch.no_grad()
    def predict_video(self, video_path: str) -> Dict[str, Any]:
        """
        Detect AI-generated content in video.
        
        Returns:
            dict with:
                - score: P(fake) 0-1
                - label: "FAKE" or "REAL"
                - details: per-frame info and explanations
        """
        # Extract frames
        frames = self.extract_frames(video_path)
        if not frames:
            return {
                "score": 0.0,
                "label": "UNKNOWN",
                "details": {"error": "Could not extract frames"}
            }
        
        # Preprocess frames → (1, T, 3, H, W) so forward treats as one video
        inputs = torch.stack([self.preprocess_frame(f) for f in frames]).to(self.device)
        inputs = inputs.unsqueeze(0)   # (T,3,H,W) → (1,T,3,H,W)
        
        # Inference — uses frozen CLIP features; untrained heads are bypassed
        self.model.eval()
        with torch.no_grad():
            output = self.model.forward(inputs)

        mean_score = float(output["fake_scores"].mean().item())
        per_frame_t = output["per_frame_fake_probs"][0]   # (T,) for first (only) batch
        per_frame = [float(p) for p in per_frame_t.cpu().tolist()]
        temporal_sim = float(output["temporal_sim"].mean().item())
        spatial_anomaly = float(output["spatial_anomaly"].mean().item())

        return {
            "score": mean_score,
            "label": "FAKE" if mean_score > self.threshold else "REAL",
            "details": {
                "per_frame_fake_probs": per_frame,
                "frame_count": len(frames),
                "detector_type": "ivyfake",
                "backbone": "CLIP-ViT-B/32 (frozen)",
                "features": ["temporal_consistency", "spatial_anomaly"],
                "temporal_sim": round(temporal_sim, 4),
                "spatial_anomaly": round(spatial_anomaly, 4),
                "note": "Principled CLIP-based detector: frame-to-frame cosine similarity + embedding variance"
            }
        }


def create_ivyfake_detector(
    device: str = "cpu",
    **kwargs
) -> IvyFakeDetector:
    """Factory function for IvyFake detector."""
    return IvyFakeDetector(device=device, **kwargs)


# Detection function for DeepfakeGuard integration
def detect_video_ivyfake(video_path: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Standalone detection function for DeepfakeGuard modality registration.
    
    Args:
        video_path: Path to video file
        device: "cpu" or "cuda"
        
    Returns:
        dict compatible with DeepfakeGuard format
    """
    detector = IvyFakeDetector(device=device)
    result = detector.predict_video(video_path)
    
    # Convert to ModalityResult format
    return ModalityResult(
        score=result["score"],
        label=result["label"],
        details=result["details"]
    ).__dict__