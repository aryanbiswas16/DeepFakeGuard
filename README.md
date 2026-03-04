# DeepFakeGuard (Conference Demo Edition)

DeepFakeGuard is a multi-detector deepfake detection toolkit with a Streamlit demo designed for live walkthroughs.

## What’s Included

- Streamlit demo UI for side-by-side detector usage
- Core detection library in `src/deepfake_guard`
- Single model weights folder in `src/deepfake_guard/weights/`

## Quick Start

### 1) Install

```bash
pip install -e .
```

If you want demo extras:

```bash
pip install -e ".[all]"
```

### 2) Run the Streamlit demo (recommended)

```bash
streamlit run ui/enhanced_gui.py
```

Then open `http://localhost:8501`.

## Detector Backends

- `dinov3` — face-based ViT detector
- `d3` — training-free temporal consistency detector
- `lipfd` — audio-visual lip forgery detector

The demo also supports `all` mode to run every detector and compare outputs in one pass.

## Project Layout

```text
deepfake-guard/
├── src/deepfake_guard/        # Core library + detector implementations
│   └── weights/               # All bundled model weights
├── ui/                        # Streamlit demo app
├── pyproject.toml
└── requirements.txt
```

## License

MIT — see [LICENSE](LICENSE).
