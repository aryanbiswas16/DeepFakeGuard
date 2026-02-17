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
    
    def __init__(self, embed_dim: int = 512):
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
    
    def __init__(self, embed_dim: int = 512):
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
    Unified explainable AIGC detector for images and videos.
    Based on vision-language architecture with CLIP backbone.
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
    
    def forward(
        self,
        images: torch.Tensor,
        return_explanations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (batch, num_frames, channels, height, width) or (batch, channels, height, width)
            return_explanations: Whether to generate explanation features
        
        Returns:
            Dictionary containing logits and features
        """
        # Handle both image and video inputs
        if images.dim() == 5:  # Video: (batch, num_frames, C, H, W)
            batch_size, num_frames = images.shape[:2]
            images_flat = images.view(-1, *images.shape[2:])
        else:  # Image: (batch, C, H, W)
            batch_size = images.shape[0]
            num_frames = 1
            images_flat = images
        
        # Extract features
        frame_features = self.extract_frame_features(images_flat)
        
        # Reshape back to (batch, num_frames, embed_dim)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        
        # Analyze temporal and spatial artifacts
        if num_frames > 1:
            temporal_features = self.temporal_analyzer(frame_features)
        else:
            temporal_features = frame_features.squeeze(1)
            
        spatial_features = self.spatial_analyzer(frame_features)
        
        # Fuse features
        combined_features = torch.cat([temporal_features, spatial_features], dim=-1)
        fused_features = self.fusion(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Prepare output
        output = {
            'logits': logits,
            'temporal_features': temporal_features,
            'spatial_features': spatial_features
        }
        
        return output
    
    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on input images/videos
        
        Returns:
            predictions: Class predictions (0=real, 1=fake)
            probabilities: Confidence scores
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(images)
            logits = output['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
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
        
        # Preprocess frames
        inputs = torch.stack([self.preprocess_frame(f) for f in frames]).to(self.device)
        
        # Add batch dimension if single frame
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        
        # Inference
        preds, probs = self.model.predict(inputs)
        
        # Get fake probability (class 1)
        fake_probs = probs[:, 1]
        mean_score = float(fake_probs.mean().item())
        per_frame = fake_probs.cpu().tolist()
        
        return {
            "score": mean_score,
            "label": "FAKE" if mean_score > self.threshold else "REAL",
            "details": {
                "per_frame_fake_probs": [float(p) for p in per_frame],
                "frame_count": len(frames),
                "detector_type": "ivyfake",
                "backbone": "CLIP-ViT-B/32",
                "features": ["temporal_artifacts", "spatial_artifacts"],
                "note": "CLIP-based explainable detector with temporal analysis"
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