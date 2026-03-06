from .core import DeepfakeGuard
from .types import ModalityResult
from .models.dinov3.detector import Detector as DINOv3Detector
from .models.d3.detector import D3Detector, create_d3_detector

# VLM explainability — optional (requires openai / anthropic / qwen-vl-utils)
try:
    from .explainability import VLMExplainer, VLMExplanation
except ImportError:
    VLMExplainer = None  # type: ignore[assignment,misc]
    VLMExplanation = None  # type: ignore[assignment,misc]

__version__ = "0.5.0"
__all__ = [
    "DeepfakeGuard",
    "ModalityResult",
    "DINOv3Detector",
    "D3Detector",
    "create_d3_detector",
    "VLMExplainer",
    "VLMExplanation",
]