# Deepfake Guard Toolkit

A multimodal Python library for deepfake detection, ready for production and research.

## Features
- **Visual Analysis:** State-of-the-art detection using DINOv3 Vision Transformers.
- **Multimodal Ready:** Architecture designed to easily plug in Audio and Metadata detectors.
- **Production Ready:** Includes FastAPI server and Streamlit demo UI.
- **Easy API:** Simple Python interface for integrating into existing workflows.

## 0. Quick Start

**Prerequisites:** Python 3.8+ and PyTorch 2.0+

1. **Install the library**
   ```bash
   pip install -e .
   ```

2. **Run the API Server**
   ```bash
   # Windows PowerShell
   $env:DEEPFAKE_GUARD_WEIGHTS="weights\dinov3_best_v3.pth"
   python -m uvicorn app.main:app --reload
   ```

3. **Run the Demo UI**
   ```bash
   streamlit run ui/demo_frontend.py
   ```

## 1. Usage as a Library

```python
from deepfake_guard import DeepfakeGuard

# Initialize (loads DINOv3 visual model)
guard = DeepfakeGuard(weights_path="weights/dinov3_best_v3.pth")

# Detect
result = guard.detect_video("path/to/video.mp4")

print(f"Verdict: {result['overall_label']}")
print(f"Confidence: {result['overall_score']}")
```

## 2. Usage as an API

The toolkit comes with a production-ready FastAPI server.

**Start the server:**
```bash
python -m uvicorn app.main:app 
```

**Query the API:**
```bash
curl -X POST "http://localhost:8000/detect" -F "file=@your_video.mp4"
```

## 3. Extending the Toolkit (Multimodal)

To add a new modality (e.g., Audio), simply:
1. Create a detector class in `src/deepfake_guard/models/`.
2. Register it in `src/deepfake_guard/core.py`.

## License
MIT License

**Prerequisites:**
```bash
pip install streamlit requests
```

**Run the UI:**
Open a **new terminal**, activate the environment again, and run:

```bash
cd deepfake-guard
streamlit run ui/client.py
```

## Library Usage (Direct)

```python
from deepfake_guard import DeepfakeGuard

# Initialize
# Paths can be relative or absolute
guard = DeepfakeGuard(weights_path="../DeepfakeDetector_v3/training/weights/dinov3_best_v3.pth")

# Detect
result = guard.detect_video("suspicious.mp4")
print(f"Fake Score: {result['score']}")
```
