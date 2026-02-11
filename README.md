# Deepfake Guard Toolkit v0.2.0

A multimodal Python library for deepfake detection with **multiple detector backends**, ready for production and research.

## 🆕 New in v0.2.0 - Multi-Detector Support

Now supports multiple detection backends:
- **🧠 DINOv3** (ViT-B/16) - Face-based detection with 0.88+ AUROC
- **🎯 ResNet18** (CNN) - Full-frame detection, lightweight

Switch between detectors at runtime via API, GUI, or Python API!

---

## Features
- **🔄 Multiple Detectors:** Switch between DINOv3 and ResNet18 backends
- **🎛️ Toggle Interface:** GUI with detector selection sidebar
- **📊 Visual Analysis:** State-of-the-art detection using DINOv3 Vision Transformers
- **🔌 Multimodal Ready:** Architecture designed to easily plug in Audio and Metadata detectors
- **🚀 Production Ready:** Includes FastAPI server and Streamlit demo UI
- **💻 Easy API:** Simple Python interface for integrating into existing workflows

---

## Quick Start

### Prerequisites
Python 3.8+ and PyTorch 2.0+

### 1. Install the library
```bash
pip install -e .
```

### 2. Run the Enhanced UI (with detector toggle)
```bash
streamlit run ui/enhanced_gui.py
```

### 3. Or run the API Server
```bash
# Using DINOv3 (requires weights)
export DEEPFAKE_GUARD_DETECTOR=dinov3
export DEEPFAKE_GUARD_WEIGHTS="weights/dinov3_best_v3.pth"
python -m uvicorn app.main:app --reload

# Or using ResNet18 (no weights needed)
export DEEPFAKE_GUARD_DETECTOR=resnet18
python -m uvicorn app.main:app --reload
```

---

## 🔄 Switching Detectors

### Python API
```python
from deepfake_guard import DeepfakeGuard

# Initialize with specific detector
guard = DeepfakeGuard(detector_type="dinov3", weights_path="weights/dinov3_best_v3.pth")

# Detect
result = guard.detect_video("video.mp4")

# Switch to the ResNet18 detector at runtime
guard.set_detector("resnet18")

# Detect with ResNet18
result2 = guard.detect_video("video.mp4")
```

### API (HTTP)
```bash
# Detect with DINOv3
curl -X POST "http://localhost:8000/detect?detector=dinov3" -F "file=@video.mp4"

# Detect with ResNet18
curl -X POST "http://localhost:8000/detect?detector=resnet18" -F "file=@video.mp4"

# Switch default detector
curl http://localhost:8000/switch/resnet18

# List available detectors
curl http://localhost:8000/detectors
```

### GUI
Launch the enhanced GUI and use the sidebar to toggle between detectors:
```bash
streamlit run ui/enhanced_gui.py
```

---

## 🧠 Detector Comparison

| Feature | DINOv3 (ViT-B/16) | ResNet18 (CNN) |
|---------|---------------------|---------------------------|
| **Architecture** | Vision Transformer (ViT-B/16) | CNN (ResNet18) |
| **Face Detection** | ✅ MTCNN face cropping | ❌ Full frame analysis |
| **Embeddings** | 768-dim | 512-dim |
| **Training** | Fine-tuned on deepfakes | Pretrained ImageNet |
| **Accuracy** | 0.88+ AUROC | Baseline |
| **Speed** | Slower | Faster |
| **Weights Required** | ✅ Yes | ❌ No |

---

## 📁 File Structure

```
DeepFakeGuard/
├── src/deepfake_guard/
│   ├── core.py                  # Main orchestrator with detector switching
│   ├── models/
│   │   ├── dinov3/              # DINOv3 detector
│   │   │   ├── detector.py
│   │   │   ├── frame_encoder.py
│   │   │   └── classifier_head.py
│   │   └── resnet18/            # ResNet18 detector (NEW)
│   │       ├── detector.py
│   │       └── __init__.py
│   └── utils/                   # Shared utilities
├── app/
│   └── main.py                  # FastAPI with detector endpoints (UPDATED)
├── ui/
│   ├── demo_frontend.py         # Basic UI
│   └── enhanced_gui.py          # UI with detector toggle (NEW)
└── weights/                     # Model weights storage
```

---

## 🔌 Integrating New Detectors

To add a new detector (e.g., Audio):

1. **Create detector module:**
```python
# src/deepfake_guard/models/audio/detector.py

def detect_audio_deepfake(video_path: str) -> Dict[str, Any]:
    # Extract audio
    # Run your detector
    return {
        "score": 0.65,
        "label": "FAKE",
        "details": {"audio_score": 0.65}
    }
```

2. **Register in DeepfakeGuard:**
```python
guard.register_modality("audio", detect_audio_deepfake, "Audio detection")
```

3. **Or add as primary detector:**
Modify `core.py` to add new `detector_type` option.

---

## 🛠️ Advanced Usage

### Custom Threshold
```python
guard = DeepfakeGuard(detector_type="dinov3")
result = guard.detect_video("video.mp4")

# Custom classification
threshold = 0.7
is_fake = result["overall_score"] > threshold
```

### Batch Processing with Different Detectors
```python
import os

guard = DeepfakeGuard()
videos = ["vid1.mp4", "vid2.mp4", "vid3.mp4"]

for detector in ["dinov3", "resnet18"]:
    guard.set_detector(detector)
    print(f"\n=== Using {detector.upper()} ===")
    
    for vid in videos:
        result = guard.detect_video(vid)
        print(f"{vid}: {result['overall_label']} ({result['overall_score']:.2%})")
```

---

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check & API info |
| `/detect` | POST | Analyze video (use `?detector=` to select) |
| `/switch/{type}` | GET | Switch default detector |
| `/detectors` | GET | List available detectors |

---

## 🔧 Troubleshooting

**"No module named 'torch'"**
```bash
pip install torch torchvision
```

**"Weights file not found"**
- For DINOv3: Ensure `weights/dinov3_best_v3.pth` exists
- For ResNet18: No weights needed (uses pretrained ImageNet)

**"Cannot switch detector"**
```python
# Make sure to clear cache when switching
guard.set_detector("resnet18")
```

---

## 📄 License

MIT License

---

**Star this repository if you find it useful!**