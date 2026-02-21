"""
DeepFakeGuard API Server
FastAPI backend for multi-detector deepfake detection

Supports: DINOv3, ResNet18, IvyFake, D3
"""

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import tempfile
import os

app = FastAPI(
    title="DeepFakeGuard API",
    description="Multimodal deepfake detection with multiple detector backends",
    version="0.4.0"
)

# Configuration
DEFAULT_DETECTOR = os.environ.get("DEEPFAKE_GUARD_DETECTOR", "dinov3")
WEIGHTS_PATH = os.environ.get("DEEPFAKE_GUARD_WEIGHTS", "weights/dinov3_best_v3.pth")

# Global guard instance (will be initialized on first use)
_guard = None
_current_detector = None

def get_guard(detector_type: str = "dinov3"):
    """Get or create DeepfakeGuard instance."""
    global _guard, _current_detector
    
    import sys
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from deepfake_guard import DeepfakeGuard
    
    # Reinitialize if detector type changed
    if _guard is None or _current_detector != detector_type:
        print(f"Initializing DeepfakeGuard with {detector_type} detector...")
        
        weights = WEIGHTS_PATH if detector_type == "dinov3" else None
        _guard = DeepfakeGuard(
            weights_path=weights if weights and os.path.exists(weights) else None,
            detector_type=detector_type
        )
        _current_detector = detector_type
    
    return _guard

@app.post("/detect")
async def detect_endpoint(
    file: UploadFile = File(..., description="Video file to analyze (MP4, MOV, AVI, MKV)"),
    detector: str = Query(
        default="dinov3",
        enum=["dinov3", "resnet18", "ivyfake", "d3"],
        description="Detector backend to use"
    )
):
    """
    Detect deepfakes in an uploaded video file.
    
    Args:
        file: Video file to analyze
        detector: Detector backend ("dinov3", "resnet18", "ivyfake", "d3")
    
    Returns:
        Detection results with score, label, and detailed analysis
    """
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        # Get guard with specified detector
        guard = get_guard(detector)
        
        # Run detection
        result = guard.detect_video(tmp_path)
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detector": detector}
        )
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/")
def read_root():
    """
    Health check and API info.
    
    Returns:
        API status, version, and available endpoints
    """
    global _current_detector
    
    return {
        "status": "DeepfakeGuard API is running",
        "version": "0.4.0",
        "current_detector": _current_detector or DEFAULT_DETECTOR,
        "available_detectors": ["dinov3", "resnet18", "ivyfake", "d3"],
        "endpoints": {
            "detect": "POST /detect?detector=dinov3|resnet18|ivyfake|d3",
            "health": "GET /",
            "switch_detector": "GET /switch/{detector_type}",
            "list_detectors": "GET /detectors"
        },
        "documentation": "https://github.com/aryanbiswas16/DeepfakeVidDetection#readme"
    }

@app.get("/switch/{detector_type}")
def switch_detector(detector_type: str):
    """
    Switch the active detector backend.
    
    Args:
        detector_type: "dinov3", "resnet18", "ivyfake", or "d3"
    
    Returns:
        Status of the switch operation
    """
    global _guard, _current_detector
    
    valid_detectors = ["dinov3", "resnet18", "ivyfake", "d3"]
    
    if detector_type not in valid_detectors:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid detector_type. Must be one of: {valid_detectors}"}
        )
    
    try:
        # Force reinitialization
        _guard = None
        guard = get_guard(detector_type)
        
        return {
            "status": "success",
            "detector": detector_type,
            "weights_loaded": guard.visual_weights_loaded if detector_type == "dinov3" else True,
            "message": f"Switched to {detector_type.upper()} detector"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/detectors")
def list_detectors():
    """
    List available detectors and their info.
    
    Returns:
        Dictionary with detector specifications and recommendations
    """
    return {
        "detectors": {
            "dinov3": {
                "name": "DINOv3 Vision Transformer",
                "description": "Face-based detection with DINOv3 ViT-B/16",
                "requires_weights": True,
                "features": ["Face cropping", "768-dim embeddings", "LayerNorm tuning"],
                "accuracy": "0.88+ AUROC",
                "speed": "Moderate",
                "best_for": "High accuracy deepfake detection"
            },
            "resnet18": {
                "name": "ResNet18 CNN",
                "description": "Full-frame detection with ResNet18",
                "requires_weights": False,
                "features": ["Full frame analysis", "No face cropping", "Lightweight"],
                "accuracy": "Baseline",
                "speed": "Fast",
                "best_for": "Quick analysis without face dependency"
            },
            "ivyfake": {
                "name": "IvyFake CLIP Detector",
                "description": "CLIP-based explainable AIGC detection with temporal/spatial analysis",
                "requires_weights": False,
                "features": [
                    "CLIP ViT-B/32 backbone",
                    "Temporal artifact analysis",
                    "Spatial artifact analysis",
                    "Explainable outputs"
                ],
                "accuracy": "Good",
                "speed": "Moderate",
                "best_for": "Explainable detection with artifact analysis"
            },
            "d3": {
                "name": "D3 - Detection by Difference of Differences",
                "description": "Training-free detection using second-order temporal features (ICCV 2025)",
                "requires_weights": False,
                "features": [
                    "Training-free",
                    "Second-order temporal features",
                    "Motion volatility analysis",
                    "Multiple encoder support"
                ],
                "accuracy": "Varies by encoder",
                "speed": "Fast",
                "best_for": "Training-free detection with temporal consistency analysis",
                "citation": "Zheng et al., ICCV 2025"
            }
        }
    }
