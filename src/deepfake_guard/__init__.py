from .core import DeepfakeGuard
from .models.dinov3.detector import Detector as DINOv3Detector
from .models.resnet18.detector import ResNet18Detector, detect_video_resnet18

__version__ = "0.2.0"
__all__ = ["DeepfakeGuard", "DINOv3Detector", "ResNet18Detector", "detect_video_resnet18"]