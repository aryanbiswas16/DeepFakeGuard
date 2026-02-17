"""
IvyFake-based detector (CLIP-based explainable AIGC detection)
Uses CLIP vision encoder with temporal and spatial artifact analyzers
"""

from .detector import IvyFakeDetector, create_ivyfake_detector, detect_video_ivyfake

__all__ = ["IvyFakeDetector", "create_ivyfake_detector", "detect_video_ivyfake"]