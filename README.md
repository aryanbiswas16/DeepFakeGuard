# Deepfake Guard Toolkit v0.4.0

A multimodal Python library for deepfake detection with **multiple detector backends**, ready for production and research.

## 🆕 New in v0.4.0 - D3 Integration

Now supports **4 detection backends**:
- **🧠 DINOv3** (ViT-B/16) - Face-based detection with 0.88+ AUROC
- **🎯 ResNet18** (CNN) - Full-frame detection, lightweight
- **🌿 IvyFake** (CLIP) - Explainable AIGC detection
- **📊 D3** (XCLIP/ResNet) - Training-free detection using second-order temporal features (ICCV 2025)

Switch between detectors at runtime via API, GUI, or Python API!

---

## Features
- **🔄 Multiple Detectors:** Switch between DINOv3, ResNet18, IvyFake, and D3 backends
- **🎛️ Toggle Interface:** GUI with detector selection sidebar
- **📊 Visual Analysis:** State-of-the-art detection using DINOv3 Vision Transformers
- **📈 Training-Free Option:** D3 detector requires no training (ICCV 2025 method)
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

# Or use D3 (training-free, second-order temporal features)
guard.set_detector("d3")
result3 = guard.detect_video("video.mp4")
```

### API (HTTP)
```bash
# Detect with DINOv3
curl -X POST "http://localhost:8000/detect?detector=dinov3" -F "file=@video.mp4"

# Detect with ResNet18
curl -X POST "http://localhost:8000/detect?detector=resnet18" -F "file=@video.mp4"

# Detect with D3 (training-free)
curl -X POST "http://localhost:8000/detect?detector=d3" -F "file=@video.mp4"

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

| Feature | DINOv3 (ViT-B/16) | ResNet18 (CNN) | IvyFake (CLIP) | D3 (Temporal) |
|---------|-------------------|----------------|----------------|---------------|
| **Architecture** | Vision Transformer (ViT-B/16) | CNN (ResNet18) | CLIP ViT-B/32 | XCLIP/ResNet |
| **Face Detection** | ✅ MTCNN face cropping | ❌ Full frame analysis | ❌ Full frame analysis | ❌ Full frame analysis |
| **Embeddings** | 768-dim | 512-dim | 512-dim | Varies |
| **Training** | Fine-tuned on deepfakes | Pretrained ImageNet | Pretrained CLIP | Training-free |
| **Accuracy** | 0.88+ AUROC | Baseline | Good | Varies by encoder |
| **Speed** | Slower | Faster | Moderate | Fast |
| **Weights Required** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Explainability** | Limited | Limited | High (artifacts) | High (volatility) |

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
│   │   ├── resnet18/            # ResNet18 detector
│   │   │   ├── detector.py
│   │   │   └── __init__.py
│   │   ├── ivyfake/             # IvyFake CLIP detector
│   │   │   ├── detector.py
│   │   │   └── __init__.py
│   │   └── d3/                  # D3 detector (NEW)
│   │       ├── detector.py
│   │       └── __init__.py
│   └── utils/                   # Shared utilities
├── app/
│   └── main.py                  # FastAPI with detector endpoints
├── ui/
│   ├── demo_frontend.py         # Basic UI
│   └── enhanced_gui.py          # UI with detector toggle
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

## 📚 Citations

If you use DeepFakeGuard in your research, please cite:

```bibtex
@software{deepfake_guard_2026,
  author = {Biswas, Aryan},
  title = {DeepFakeGuard: A Unified Framework for Deepfake Detection},
  url = {https://github.com/aryanbiswas16/DeepfakeVidDetection},
  year = {2026}
}
```

### Integrated Methods

**D3 Detector (ICCV 2025):**
```bibtex
@inproceedings{zheng2025d3,
  title={D3: Training-Free AI-Generated Video Detection Using Second-Order Features},
  author={Zheng, Chende and Suo, Ruiqi and Lin, Chenhao and Zhao, Zhengyu and 
          Yang, Le and Liu, Shuai and Yang, Minghui and Wang, Cong and Shen, Chao},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025},
  url={https://arxiv.org/abs/2508.00701}
}
```

**IvyFake:**
```bibtex
@software{ivyfake2024,
  author = {Khan, Hamza and et al.},
  title = {IvyFake: An Explainable and Optimized AI-Generated Image Detection},
  url = {https://github.com/HamzaKhan760/IvyFakeGenDetector},
  year = {2024}
}
```

**DINOv3:**
```bibtex
@inproceedings{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and 
          Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and 
          Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**⭐ Star this repository if you find it useful!**

*Built with ❤️ for the deepfake detection research community.*