from .core import DeepfakeGuard
from .types import ModalityResult
from .models.dinov3.detector import Detector as DINOv3Detector
from .models.d3.detector import D3Detector, create_d3_detector

__version__ = "0.4.0"
__all__ = [
    "DeepfakeGuard",
    "ModalityResult",
    "DINOv3Detector",
    "D3Detector",
    "create_d3_detector",
]