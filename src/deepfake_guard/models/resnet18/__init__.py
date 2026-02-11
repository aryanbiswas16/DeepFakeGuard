"""ResNet18-based video detector (Friend's implementation)"""

from .detector import ResNet18Detector, create_resnet18_detector, detect_video_resnet18

__all__ = ["ResNet18Detector", "create_resnet18_detector", "detect_video_resnet18"]