# DeepFakeGuard

A multi-detector deepfake detection Python library with ensemble fusion, optional VLM-powered explainability, and a Streamlit demo for live walkthroughs.

## Features

- **Three complementary detectors** - DINOv3 (face-swap), D3 (temporal consistency), LipFD (lip-sync)
- **Ensemble fusion** - domain-aware trust x certainty weighting with outlier veto
- **VLM explainability** *(optional)* - post-hoc natural-language explanations via OpenAI, Anthropic, or local Qwen2-VL
- **Streamlit demo UI** - side-by-side detector comparison with VLM integration

## Installation

### Core library

```bash
pip install -e .
```

### With VLM explainability (OpenAI)

```bash
pip install -e ".[vlm]"
```

### With Anthropic backend

```bash
pip install -e ".[vlm-anthropic]"
```

### With local Qwen2-VL backend

```bash
pip install -e ".[vlm-local]"
```

### Everything (demo + all VLM backends)

```bash
pip install -e ".[all]"
```

## Quick Start

### Library usage

```python
from deepfake_guard import DeepfakeGuard

# Single detector (choose "dinov3", "d3", or "lipfd")
guard = DeepfakeGuard(detector_type="dinov3")
result = guard.detect_video("video.mp4")
print(result["overall_label"], result["overall_score"])

# Ensemble (all detectors)
guards = {
    "dinov3": DeepfakeGuard(detector_type="dinov3"),
    "d3":     DeepfakeGuard(detector_type="d3"),
    "lipfd":  DeepfakeGuard(detector_type="lipfd"),
}
result = DeepfakeGuard.ensemble_detect_video(guards, "video.mp4")
print(result["overall_label"], result["overall_score"])

# Ensemble + VLM explanation
result = DeepfakeGuard.ensemble_detect_video(
    guards, "video.mp4",
    vlm_backend="openai",
    vlm_api_key="sk-..."
)
print(result["vlm_explanation"]["explanation"])
```

### Streamlit demo

```bash
streamlit run ui/enhanced_gui.py
```

Then open `http://localhost:8501`.

## Detectors

| Backend | Method | Domain | Reference |
|---------|--------|--------|-----------|
| `dinov3` | DINOv2 ViT-B/16 + classifier head | Face-swap | - |
| `d3` | Training-free temporal consistency | General | ICCV 2025 |
| `lipfd` | Audio-visual lip forgery | Lip-sync | NeurIPS 2024 |

The demo also supports `all` mode to run every detector and display an ensemble result.

## VLM Explainability

The optional explainability module generates natural-language forensic reports by feeding keyframe grids and detector scores to a vision-language model. Three backends are supported:

- **OpenAI** (`openai`) - GPT-4o mini via API
- **Anthropic** (`anthropic`) - Claude via API
- **Local** (`qwen2vl`) - Qwen2-VL-7B-Instruct on-device

VLM is fully optional - the library works without it. Install the extras you need:

```bash
pip install -e ".[vlm]"           # OpenAI
pip install -e ".[vlm-anthropic]" # Anthropic
pip install -e ".[vlm-local]"     # Qwen2-VL (requires GPU)
```

## Project Layout

```text
deepfake-guard/
├── src/deepfake_guard/
│   ├── core.py                    # Main DeepfakeGuard class + ensemble logic
│   ├── types.py                   # Result dataclasses
│   ├── models/                    # Detector implementations
│   │   ├── dinov3/
│   │   ├── d3/
│   │   └── lipfd/
│   ├── explainability/            # VLM explanation module (optional)
│   │   ├── vlm_explainer.py
│   │   ├── prompts.py
│   │   └── grid.py
│   ├── utils/                     # Face crop, video I/O, preprocessing
│   └── weights/                   # Bundled model weights
├── ui/enhanced_gui.py             # Streamlit demo app
├── pyproject.toml
└── requirements.txt
```

## License

MIT - see [LICENSE](LICENSE).
