"""
D3 Detector - Detection by Difference of Differences
Training-free AI-generated video detection using second-order temporal features.

Based on: "D3: Training-Free AI-Generated Video Detection Using Second-Order Features"
ICCV 2025 - Zheng et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List
import cv2
from pathlib import Path


class D3FeatureExtractor:
    """Extract second-order temporal features from video frames."""
    
    def __init__(self, encoder_name: str = "xclip-16", device: str = "cpu"):
        self.device = device
        self.encoder_name = encoder_name
        self.encoder = self._load_encoder(encoder_name)
        self.encoder.eval()
        
    def _load_encoder(self, encoder_name: str):
        """Load pre-trained encoder model."""
        encoder_name = encoder_name.lower()
        
        if "xclip" in encoder_name:
            try:
                from transformers import CLIPModel, CLIPProcessor
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
                # Use vision encoder only
                return model.vision_model.to(self.device)
            except:
                # Fallback to ViT
                import timm
                model = timm.create_model("vit_base_patch16_clip_224", pretrained=True, num_classes=0)
                return model.to(self.device)
                
        elif "resnet" in encoder_name:
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            model.fc = nn.Identity()  # Remove classifier
            return model.to(self.device)
            
        elif "mobilenet" in encoder_name:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            model.classifier = nn.Identity()
            return model.to(self.device)
            
        else:
            # Default to CLIP ViT
            import timm
            model = timm.create_model("vit_base_patch16_clip_224", pretrained=True, num_classes=0)
            return model.to(self.device)
    
    @torch.no_grad()
    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract features from frames.
        Args:
            frames: [T, C, H, W] tensor
        Returns:
            features: [T, D] tensor
        """
        frames = frames.to(self.device)
        features = []
        
        # Process in batches to avoid OOM
        batch_size = 32
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            feat = self.encoder(batch)
            features.append(feat.cpu())
        
        return torch.cat(features, dim=0)  # [T, D]
    
    def compute_first_order_diff(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute first-order differences (temporal gradients).
        Args:
            features: [T, D]
        Returns:
            first_diff: [T-1, D]
        """
        return features[1:] - features[:-1]  # [T-1, D]
    
    def compute_second_order_diff(self, first_diff: torch.Tensor) -> torch.Tensor:
        """
        Compute second-order differences (difference of differences).
        Args:
            first_diff: [T-1, D]
        Returns:
            second_diff: [T-2, D]
        """
        return first_diff[1:] - first_diff[:-1]  # [T-2, D]
    
    def compute_volatility(self, features: torch.Tensor) -> float:
        """
        Compute volatility score from second-order differences.
        Higher volatility = more natural/real video.
        Lower volatility = more AI-generated.
        
        Args:
            features: [T, D] extracted features
        Returns:
            volatility: scalar score
        """
        if len(features) < 3:
            return 0.0
        
        # First-order differences
        first_diff = self.compute_first_order_diff(features)  # [T-1, D]
        
        # Second-order differences (D3 core)
        second_diff = self.compute_second_order_diff(first_diff)  # [T-2, D]
        
        # Compute L2 norm of second-order differences
        second_diff_norm = torch.norm(second_diff, p=2, dim=1)  # [T-2]
        
        # Volatility = mean of second-order differences
        volatility = second_diff_norm.mean().item()
        
        return volatility


class D3Detector:
    """
    D3: Detection by Difference of Differences
    Training-free AI-generated video detector.
    """
    
    def __init__(self, 
                 encoder_name: str = "xclip-16",
                 threshold: float = 0.5,
                 device: str = "cpu",
                 calibrate: bool = False):
        """
        Initialize D3 detector.
        
        Args:
            encoder_name: Encoder to use (xclip-16, xclip-32, resnet-18, mobilenet-v3)
            threshold: Threshold for real vs fake (calibrated per encoder)
            device: Device to run on
            calibrate: Whether to auto-calibrate threshold
        """
        self.device = device
        self.encoder_name = encoder_name
        self.calibrate = calibrate
        
        # Default thresholds per encoder (from paper/empirical)
        self.default_thresholds = {
            "xclip-16": 0.15,
            "xclip-32": 0.12,
            "resnet-18": 0.18,
            "mobilenet-v3": 0.20,
        }
        
        self.threshold = threshold if threshold else self.default_thresholds.get(encoder_name, 0.15)
        
        # Initialize feature extractor
        self.feature_extractor = D3FeatureExtractor(encoder_name, device)
        
        print(f"D3 Detector initialized with {encoder_name} encoder")
        print(f"Threshold: {self.threshold:.3f}")
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for encoder."""
        # Resize to 224x224
        frame = cv2.resize(frame, (224, 224))
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize (ImageNet stats)
        frame = frame.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = (frame - mean) / std
        
        # To tensor [C, H, W]
        frame = torch.from_numpy(frame).permute(2, 0, 1)
        
        return frame
    
    def predict_video(self, video_path: str, sample_frames: int = 32) -> Dict[str, Any]:
        """
        Predict if video is real or AI-generated.
        
        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample
        
        Returns:
            Dictionary with prediction results
        """
        # Extract frames
        frames = self._extract_frames(video_path, sample_frames)
        
        if len(frames) < 3:
            return {
                "score": 0.5,
                "label": "UNKNOWN",
                "details": {"error": "Insufficient frames"}
            }
        
        # Preprocess frames
        processed = torch.stack([self.preprocess_frame(f) for f in frames])
        
        # Extract features
        features = self.feature_extractor.extract_features(processed)
        
        # Compute volatility (D3 core)
        volatility = self.feature_extractor.compute_volatility(features)
        
        # Normalize volatility to [0, 1] for scoring
        # Higher volatility = more real
        max_expected_volatility = 0.5  # Empirical upper bound
        normalized_score = min(volatility / max_expected_volatility, 1.0)
        
        # Determine label
        # If volatility > threshold: Real (high motion variance)
        # If volatility <= threshold: Fake (constrained motion)
        is_real = volatility > self.threshold
        
        # Convert to "fake probability" for consistency with other detectors
        # Higher score = more fake
        fake_score = 1.0 - normalized_score
        
        return {
            "score": fake_score,
            "label": "REAL" if is_real else "FAKE",
            "details": {
                "volatility": volatility,
                "threshold": self.threshold,
                "encoder": self.encoder_name,
                "frame_count": len(frames),
                "detector_type": "d3"
            }
        }
    
    def _extract_frames(self, video_path: str, num_frames: int = 32) -> List[np.ndarray]:
        """Extract evenly spaced frames from video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        # Sample frame indices
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames


def create_d3_detector(encoder: str = "xclip-16", device: str = "cpu") -> D3Detector:
    """Factory function to create D3 detector."""
    return D3Detector(encoder_name=encoder, device=device)
