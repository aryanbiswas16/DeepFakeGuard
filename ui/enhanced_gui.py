"""
Enhanced Deepfake Guard GUI with detector toggle
Supports DINOv3, D3, and LipFD detectors
"""

import streamlit as st
from pathlib import Path
import tempfile
import os


# ── Chart helpers ────────────────────────────────────────────────────────────

def _get_per_frame_probs(result_dict: dict):
    """Pull per_frame_fake_probs out of a detect_video result, or return None."""
    for res in result_dict.get("modality_results", {}).values():
        probs = res.get("details", {}).get("per_frame_fake_probs")
        if probs:
            return probs
    return None


def render_frame_chart(per_frame_probs: list, threshold: float = 0.5, title: str = ""):
    """Render a per-frame fake-probability timeline."""
    try:
        import plotly.graph_objects as go
        x = list(range(len(per_frame_probs)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=per_frame_probs,
            mode="lines+markers",
            name="P(fake)",
            line=dict(color="#e74c3c", width=2),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(231,76,60,0.12)"
        ))
        fig.add_hline(
            y=threshold, line_dash="dash", line_color="orange",
            annotation_text=f"threshold ({threshold:.0%})",
            annotation_position="bottom right"
        )
        fig.update_layout(
            title=f"Per-Frame P(fake) — {title}" if title else "Per-Frame P(fake)",
            xaxis_title="Frame index",
            yaxis_title="P(fake)",
            yaxis=dict(range=[0, 1]),
            height=280,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        import pandas as pd
        st.line_chart(pd.DataFrame({"P(fake)": per_frame_probs}))


def render_ensemble_chart(scores: dict, labels: dict, threshold: float = 0.5):
    """Render a side-by-side bar chart comparing all detector scores."""
    try:
        import plotly.graph_objects as go
        names = list(scores.keys())
        vals = [scores[n] for n in names]
        colors = ["#e74c3c" if labels[n] == "FAKE" else "#2ecc71" for n in names]
        fig = go.Figure(go.Bar(
            x=[n.upper() for n in names],
            y=vals,
            marker_color=colors,
            text=[f"{v:.1%}" for v in vals],
            textposition="outside"
        ))
        fig.add_hline(
            y=threshold, line_dash="dash", line_color="orange",
            annotation_text=f"threshold ({threshold:.0%})",
            annotation_position="top left"
        )
        fig.update_layout(
            yaxis=dict(range=[0, 1.15], title="P(fake)"),
            xaxis_title="Detector",
            height=340,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        import pandas as pd
        st.bar_chart(pd.DataFrame({"Score": scores}))


# Page config
st.set_page_config(
    page_title="Deepfake Guard",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ Deepfake Guard")
st.markdown("Multimodal Deepfake Detection Toolkit")

# Sidebar for detector selection
st.sidebar.header("⚙️ Detector Settings")

detector_type = st.sidebar.selectbox(
    "Select Detector",
    options=["dinov3", "d3", "lipfd", "all"],
    format_func=lambda x: {
        "dinov3":   "🧠 DINOv3 (ViT-B/16 - 0.88 AUROC)",
        "d3":       "📊 D3 (Training-Free - Temporal)",
        "lipfd":    "🎤 LipFD (Audio-Visual - NeurIPS 2024)",
        "all":      "🔀 All Detectors (Ensemble)"
    }[x],
    help="Switch between different detection backends, or run all for an ensemble comparison"
)

detector_info = {
    "dinov3": {
        "name": "DINOv3 Vision Transformer",
        "description": "Face-based detection with DINOv3 ViT-B/16",
        "features": ["Face cropping (MTCNN)", "768-dim embeddings", "LayerNorm tuning", "0.88+ AUROC"],
        "pros": "Higher accuracy, trained on deepfakes",
        "cons": "Requires face detection, slower",
        "requires_weights": False
    },
    "d3": {
        "name": "D3 - Detection by Difference of Differences",
        "description": "Training-free AI video detection using second-order temporal features",
        "features": [
            "Training-free detection",
            "Second-order temporal features",
            "Motion volatility analysis",
            "No face cropping required",
            "ICCV 2025 method"
        ],
        "pros": "No training needed, analyzes temporal consistency",
        "cons": "Sensitivity varies by video type",
        "requires_weights": False
    },
    "lipfd": {
        "name": "LipFD — Lip Forgery Detection",
        "description": "Audio-visual lip-sync deepfake detection (NeurIPS 2024)",
        "features": [
            "CLIP ViT-L/14 global audio-visual features",
            "Region-Aware ResNet-50 (multi-scale attention)",
            "Mel-spectrogram + frame composite input",
            "RA-Loss (lip-region focus)",
            "91.2% acc / 0.962 AUROC on FakeAVCeleb"
        ],
        "pros": "Only audio-visual detector — catches lip-sync fakes others miss",
        "cons": "Requires 1.68 GB weights; modern Wav2Lip may evade",
        "requires_weights": True
    }
}

# Show detector info
if detector_type == "all":
    st.sidebar.markdown("**🔀 Ensemble Mode**")
    st.sidebar.markdown("_Domain-aware fusion of all 3 detectors._")
    st.sidebar.markdown("**Detectors:**")
    st.sidebar.markdown("- 🧠 DINOv3 → face-swap deepfakes")
    st.sidebar.markdown("- 📊 D3 → AI-generated video")
    st.sidebar.markdown("- 🎤 LipFD → lip-sync deepfakes")
    st.sidebar.markdown("**Fusion features:**")
    st.sidebar.markdown("- Trust × certainty weighting")
    st.sidebar.markdown("- Outlier veto (domain mismatch)")
    st.sidebar.markdown("- Cross-modal agreement analysis")
    st.sidebar.markdown("- Natural-language explanation")
    st.sidebar.markdown("**📝 Applicability Notes:**")
    st.sidebar.markdown("- 🧠 DINOv3 is most reliable when a clear face is visible")
    st.sidebar.markdown("- 🎤 LipFD requires visible lips and usable audio")
    st.sidebar.markdown("- 📊 D3 is most informative when there is temporal movement")
    st.sidebar.markdown("- ⚠️ If one modality is inapplicable, rely on the others")
    st.sidebar.markdown("✅ **Pros:** Covers all forgery families")
    st.sidebar.markdown("⚠️ **Cons:** Slower — loads all models")
else:
    info = detector_info[detector_type]
    st.sidebar.markdown(f"**{info['name']}**")
    st.sidebar.markdown(f"_{info['description']}_")
    st.sidebar.markdown("**Features:**")
    for feat in info['features']:
        st.sidebar.markdown(f"- {feat}")
    st.sidebar.markdown(f"✅ **Pros:** {info['pros']}")
    st.sidebar.markdown(f"⚠️ **Cons:** {info['cons']}")

# LipFD options
if detector_type in ("lipfd", "all"):
    st.sidebar.divider()
    lipfd_weights_input = st.sidebar.text_input(
        "LipFD Weights Path (optional)",
        value="",
        placeholder="src/deepfake_guard/weights/lipfd_ckpt.pth",
        help="Leave blank to auto-detect src/deepfake_guard/weights/lipfd_ckpt.pth, or enter a custom path."
    )
    st.sidebar.caption("Download: github.com/AaronComo/LipFD")
else:
    lipfd_weights_input = ""

# VLM Explainability options (ensemble mode only)
if detector_type == "all":
    st.sidebar.divider()
    st.sidebar.markdown("**🧠 VLM Explainability** *(FAKE verdicts only)*")
    vlm_backend = st.sidebar.selectbox(
        "VLM Backend",
        options=["openai", "anthropic", "qwen2vl", "disabled"],
        format_func=lambda x: {
            "openai":    "☁️ GPT-4o mini (OpenAI API)",
            "anthropic": "☁️ Claude Opus 4.6 (Anthropic API)",
            "qwen2vl":   "🖥️ Qwen2-VL-7B (local — requires GPU)",
            "disabled":  "🚫 Disabled",
        }[x],
        help="Provides a natural language explanation of why the video was flagged FAKE.",
    )
    vlm_api_key_input = ""
    if vlm_backend == "openai":
        import os as _os
        vlm_api_key_input = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Get your key at platform.openai.com. Stored only in this session.",
        ) or _os.environ.get("OPENAI_API_KEY", "")
        if not vlm_api_key_input:
            st.sidebar.caption("⚠️ No API key — VLM will be skipped.")
    elif vlm_backend == "anthropic":
        import os as _os
        vlm_api_key_input = st.sidebar.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            help="Get your key at console.anthropic.com. Stored only in this session.",
        ) or _os.environ.get("ANTHROPIC_API_KEY", "")
        if not vlm_api_key_input:
            st.sidebar.caption("⚠️ No API key — VLM will be skipped.")
    elif vlm_backend == "qwen2vl":
        st.sidebar.caption("Requires CUDA GPU + PyTorch 2.4+ + ~15 GB download on first run.")
else:
    vlm_backend = "disabled"
    vlm_api_key_input = ""

# D3 options (shown for d3 or ensemble)
if detector_type in ("d3", "all"):
    st.sidebar.divider()
    d3_encoder = st.sidebar.selectbox(
        "D3 Encoder",
        options=["xclip-16", "xclip-32", "clip-16", "clip-32", "dino-base", "dino-large", "resnet-18", "mobilenet-v3"],
        format_func=lambda x: {
            "xclip-16":     "XCLIP-16 (Recommended)",
            "xclip-32":     "XCLIP-32",
            "clip-16":      "CLIP ViT-B/16",
            "clip-32":      "CLIP ViT-B/32",
            "dino-base":    "DINOv2-Base (facebook/dinov2-base)",
            "dino-large":   "DINOv2-Large (facebook/dinov2-large)",
            "resnet-18":    "ResNet-18",
            "mobilenet-v3": "MobileNet-v3 (Fastest)",
        }[x],
        help="Encoder backbone for D3 feature extraction"
    )
    d3_threshold = st.sidebar.slider(
        "D3 Volatility Threshold",
        min_value=0.5, max_value=6.0, value=2.5, step=0.1,
        help="Lower = more sensitive. Real videos tend to have higher motion volatility than AI-generated ones."
    )
else:
    d3_encoder = None
    d3_threshold = 2.5


# ── Detector loading (each type cached independently) ────────────────────────
@st.cache_resource
def _load_single_detector(det_type: str, d3_enc: str = "xclip-16", lipfd_wts: str = ""):
    import sys
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from deepfake_guard import DeepfakeGuard
    try:
        weights = None
        if det_type == "lipfd":
            from deepfake_guard.utils.weights import resolve_lipfd_weights
            # Priority: user input → default location → auto-download
            weights = resolve_lipfd_weights(
                lipfd_wts if lipfd_wts else None
            )
        guard = DeepfakeGuard(detector_type=det_type, weights_path=weights)
        if det_type == "d3" and d3_enc:
            guard._init_d3(encoder=d3_enc)
        return guard, None
    except Exception as e:
        return None, str(e)


def _load_all_detectors(d3_enc: str = "xclip-16", lipfd_wts: str = ""):
    """Load all 3 detectors, each using the shared cache."""
    guards, errors = {}, {}
    for t in ["dinov3", "d3", "lipfd"]:
        g, e = _load_single_detector(t, d3_enc, lipfd_wts)
        if e:
            errors[t] = e
        else:
            guards[t] = g
    return guards, errors


# Initialise / retrieve from cache
if detector_type == "all":
    with st.spinner("Loading all 3 detectors (results are cached after first run)..."):
        guards, load_errors = _load_all_detectors(d3_encoder, lipfd_weights_input)
    for det, err in load_errors.items():
        st.sidebar.warning(f"⚠️ {det.upper()} failed: {err}")
    if not guards:
        st.error("No detectors could be loaded.")
        st.stop()
    loaded_names = ", ".join(k.upper() for k in guards)
    st.success(f"✅ Loaded: {loaded_names}")
else:
    with st.spinner(f"Loading {detector_type.upper()} detector..."):
        _guard, _error = _load_single_detector(detector_type, d3_encoder, lipfd_weights_input)
    if _error:
        st.error(f"Failed to load detector: {_error}")
        st.stop()
    guards = {detector_type: _guard}
    st.success(f"✅ {detector_type.upper()} detector loaded!")

# Main content
st.divider()

# File upload
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi", "mkv"],
        help="Supported formats: MP4, MOV, AVI, MKV"
    )

with col2:
    st.subheader("🔍 Detection Settings")
    threshold = st.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Score above this = FAKE, below = REAL"
    )

# Analysis
if uploaded_file is not None:
    st.divider()
    
    # Save uploaded file — stream in chunks to avoid doubling RAM usage
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        # Write in 8 MB chunks instead of getvalue() which loads everything at once
        uploaded_file.seek(0)
        while True:
            chunk = uploaded_file.read(8 * 1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        video_path = tmp.name
    
    # Display video
    video_size_mb = os.path.getsize(video_path) / 1024 / 1024
    col_video, col_results = st.columns([1, 1])
    
    with col_video:
        st.subheader("📺 Video Preview")
        # Stream from the temp file path instead of re-reading the
        # UploadedFile into memory a second time.
        st.video(video_path)
        st.caption(f"File: {uploaded_file.name} ({video_size_mb:.1f} MB)")
    
    with col_results:
        st.subheader("🎯 Detection Results")
        
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            with st.spinner("Analyzing video... This may take a moment."):
                try:
                    # Apply user-configured D3 threshold before running
                    if "d3" in guards:
                        d3g = guards["d3"]
                        if hasattr(d3g, "d3_detector") and d3g.d3_detector:
                            d3g.d3_detector.threshold = d3_threshold

                    if detector_type == "all":
                        # ── Ensemble mode (domain-aware fusion) ────────────
                        from deepfake_guard import DeepfakeGuard as _DG

                        ensemble = _DG.ensemble_detect_video(
                            guards, video_path, threshold=threshold,
                            vlm_backend=vlm_backend,
                            vlm_api_key=vlm_api_key_input or None,
                        )

                        ensemble_score = ensemble["overall_score"]
                        ensemble_label = ensemble["overall_label"]
                        scores = ensemble["scores"]
                        labels = ensemble["labels"]
                        agreement = ensemble["agreement"]
                        explanation = ensemble["explanation"]
                        outliers = ensemble.get("outliers", {})
                        contribs = ensemble.get("contributions", {})
                        applicability = ensemble.get("applicability", {})
                        all_results = ensemble["detector_results"]

                        fake_votes = sum(1 for l in labels.values() if l == "FAKE")

                        # ── Verdict banner ──────────────────────────────
                        if ensemble_label == "FAKE":
                            st.error("## 🚨 ENSEMBLE VERDICT: FAKE")
                        else:
                            st.success("## ✅ ENSEMBLE VERDICT: REAL")

                        # ── Top-line metrics ────────────────────────────
                        vcols = st.columns(4)
                        with vcols[0]:
                            st.metric("Ensemble Score", f"{ensemble_score:.1%}")
                        with vcols[1]:
                            st.metric("Fake Votes", f"{fake_votes} / {len(guards)}")
                        with vcols[2]:
                            certainty = abs(ensemble_score - 0.5) * 2
                            st.metric("Certainty", f"{certainty:.1%}")
                        with vcols[3]:
                            _agree_icons = {
                                "unanimous": "✅ Unanimous",
                                "majority-fake": "⚠️ Majority FAKE",
                                "majority-real": "✅ Majority REAL",
                                "split": "🔶 Split",
                                "inconclusive": "❓ Inconclusive",
                            }
                            st.metric("Agreement", _agree_icons.get(agreement, agreement))

                        # ── Natural-language explanation ─────────────
                        st.markdown("### 💬 Explanation")
                        st.info(explanation)
                        st.caption(
                            "Applicability notes: DINOv3 works best with clear faces; "
                            "LipFD needs visible lips + usable audio; D3 is strongest "
                            "when there is enough temporal movement."
                        )

                        # ── VLM Semantic Explanation ─────────────────
                        vlm_data = ensemble.get("vlm_explanation")
                        if vlm_data and vlm_data.get("available"):
                            st.markdown("### 🧠 Semantic Analysis (VLM)")

                            if vlm_data.get("artifacts_found"):
                                conf = vlm_data.get("confidence", "medium")
                                conf_icons = {
                                    "high": "🔴 High",
                                    "medium": "🟡 Medium",
                                    "low": "🟢 Low",
                                }
                                st.warning(
                                    f"**Visual artifacts detected** "
                                    f"(confidence: {conf_icons.get(conf, conf)})\n\n"
                                    f"{vlm_data['explanation']}"
                                )
                                categories = vlm_data.get("artifact_categories", [])
                                if categories:
                                    st.caption(
                                        "Artifact types: "
                                        + ", ".join(
                                            c.replace("_", " ").title()
                                            for c in categories
                                        )
                                    )
                                kf = vlm_data.get("key_frames", [])
                                if kf:
                                    st.caption(
                                        "Key frames: " + ", ".join(str(f) for f in kf)
                                    )
                            else:
                                st.success(
                                    f"**No visual artifacts detected**\n\n"
                                    f"{vlm_data.get('explanation', '')}"
                                )

                            st.caption(
                                f"VLM backend: {vlm_data.get('backend', '?')}"
                            )

                        # ── Detector comparison chart ───────────────
                        st.markdown("### 🔀 Detector Comparison")
                        render_ensemble_chart(scores, labels, threshold)

                        # ── Contribution breakdown ───────────────────
                        st.markdown("### ⚖️ Fusion Weights")
                        _domain_labels = {
                            "dinov3": "Face-Swap Deepfake",
                            "d3": "AI-Generated Video",
                            "lipfd": "Lip-Sync Deepfake",
                        }
                        _app_icons = {
                            "high": "🟢",
                            "medium": "🟡",
                            "low": "🟠",
                            "unavailable": "⚫",
                        }
                        for det_name in sorted(contribs.keys()):
                            c = contribs[det_name]
                            dom = _domain_labels.get(det_name, "Unknown")
                            outlier_tag = " 🚫 OUTLIER (vetoed)" if c.get("outlier") else ""
                            icon = "🔴" if labels.get(det_name) == "FAKE" else "🟢"
                            app = applicability.get(det_name, {})
                            app_level = app.get("level", "high")
                            app_reason = app.get("reason", "")
                            app_icon = _app_icons.get(app_level, "🟢")
                            st.markdown(
                                f"{icon} **{det_name.upper()}** ({dom}) — "
                                f"Score: {c.get('score', 0):.1%} · "
                                f"Trust: {c.get('trust', 0):.0%} · "
                                f"Certainty: {c.get('certainty', 0):.1%} · "
                                f"Applicability: {app_icon} {str(app_level).upper()} · "
                                f"Effective weight: {c.get('weight', 0):.3f}"
                                f"{outlier_tag}"
                            )
                            if app_reason:
                                st.caption(f"{det_name.upper()} applicability note: {app_reason}")

                        # ── Per-detector detail panels ──────────────
                        st.markdown("### 📈 Per-Detector Details")
                        for det_name, res in all_results.items():
                            per_frame = _get_per_frame_probs(res)
                            det_label = labels.get(det_name, "?")
                            det_score = scores.get(det_name, 0)
                            is_outlier = outliers.get(det_name, False)
                            app = applicability.get(det_name, {})
                            app_level = app.get("level", "high")
                            app_icon = _app_icons.get(app_level, "🟢")
                            icon = "🔴" if det_label == "FAKE" else "🟢"
                            outlier_suffix = " ⛔ OUTLIER" if is_outlier else ""
                            with st.expander(
                                f"{icon} {det_name.upper()} — {det_score:.1%} ({det_label}) · {app_icon} {str(app_level).upper()}{outlier_suffix}",
                                expanded=True
                            ):
                                if is_outlier:
                                    st.warning(
                                        f"⚠️ {det_name.upper()} was flagged as an "
                                        "outlier (contradicts all other detectors) "
                                        "and was heavily down-weighted in the "
                                        "ensemble score."
                                    )
                                app_reason = app.get("reason", "")
                                if app_reason:
                                    st.caption(f"Applicability: {app_reason}")
                                if per_frame:
                                    render_frame_chart(per_frame, threshold, det_name.upper())
                                else:
                                    # D3: volatility scalar
                                    details = res.get("modality_results", {}).get("visual", {}).get("details", {})
                                    vol = details.get("volatility")
                                    if vol is not None:
                                        thr = details.get("threshold", d3_threshold)
                                        import math
                                        try:
                                            proxy = 1.0 / (1.0 + math.exp(1.5 * (vol - thr)))
                                        except OverflowError:
                                            proxy = 0.0
                                        st.caption(f"Motion Volatility: {vol:.4f}  |  threshold: {thr:.2f}")
                                        st.progress(proxy, text=f"Fake-likelihood (sigmoid): {proxy:.1%}")
                                    # LipFD: audio-visual sample stats
                                    av = res.get("modality_results", {}).get("audio_visual", {}).get("details", {})
                                    fake_ratio = av.get("fake_ratio")
                                    if fake_ratio is not None:
                                        n_samples = av.get("num_samples", "?")
                                        st.caption(f"Lip-sync samples: {n_samples}  |  fake ratio: {fake_ratio:.1%}")
                                        st.progress(det_score, text=f"Mean P(fake): {det_score:.1%}")

                        all_errs = ensemble.get("errors", [])
                        if all_errs:
                            with st.expander("⚠️ Errors"):
                                for err in all_errs:
                                    st.warning(err)

                        with st.expander("📄 Raw JSON (ensemble)"):
                            st.json(ensemble)

                    else:
                        # ── Single-detector mode ────────────────────────────
                        result = guards[detector_type].detect_video(video_path)

                        overall_score = result.get("overall_score", 0)
                        overall_label = result.get("overall_label", "UNKNOWN")

                        if overall_label == "FAKE":
                            st.error("## 🚨 FAKE DETECTED")
                        elif overall_label == "REAL":
                            st.success("## ✅ REAL VIDEO")
                        else:
                            st.warning("## ⚠️ UNKNOWN")

                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Confidence", f"{overall_score:.1%}")
                        with cols[1]:
                            st.metric("Threshold", f"{threshold:.0%}")
                        with cols[2]:
                            confidence = abs(overall_score - 0.5) * 2
                            st.metric("Certainty", f"{confidence:.1%}")

                        st.progress(overall_score)

                        # Per-frame probability timeline
                        per_frame = _get_per_frame_probs(result)
                        if per_frame:
                            st.markdown("### 📈 Per-Frame Timeline")
                            render_frame_chart(per_frame, threshold, detector_type.upper())

                        # Modality details
                        st.markdown("### 📊 Detailed Analysis")
                        for modality, res in result.get("modality_results", {}).items():
                            with st.expander(f"{modality.title()} Analysis", expanded=True):
                                mcols = st.columns([1, 2])
                                with mcols[0]:
                                    score = res.get("score", 0)
                                    label = res.get("label", "UNKNOWN")
                                    if label == "FAKE":
                                        st.error(f"**{label}**\n\nScore: {score:.3f}")
                                    else:
                                        st.success(f"**{label}**\n\nScore: {score:.3f}")
                                with mcols[1]:
                                    details = res.get("details", {})
                                    det_type_str = details.get("detector_type", detector_type)
                                    st.caption(f"Detector: {det_type_str.upper()}")
                                    if det_type_str == "lipfd":
                                        # LipFD-specific fields
                                        n_samples = details.get("num_samples")
                                        if n_samples is not None:
                                            st.caption(f"Lip-sync samples: {n_samples}")
                                        fake_ratio = details.get("fake_ratio")
                                        if fake_ratio is not None:
                                            st.caption(f"Fake sample ratio: {fake_ratio:.1%}")
                                        score_std = details.get("score_std")
                                        if score_std is not None:
                                            st.caption(f"Score std dev: {score_std:.4f}")
                                        score_min = details.get("score_min")
                                        score_max = details.get("score_max")
                                        if score_min is not None and score_max is not None:
                                            st.caption(f"Score range: {score_min:.3f} – {score_max:.3f}")
                                        model_meta = details.get("model", {})
                                        arch = model_meta.get("architecture") or details.get("architecture")
                                        if arch:
                                            st.caption(f"Architecture: {arch}")
                                        wts = model_meta.get("weights_loaded")
                                        if wts is not None:
                                            st.caption(f"Pretrained weights: {'✅ yes' if wts else '⚠️ no (demo mode)'}")
                                        st.caption("Paper: Liu et al., NeurIPS 2024")
                                    else:
                                        frame_count = details.get("frame_count") or details.get("per_frame_fake_probs", [])
                                        if isinstance(frame_count, list):
                                            frame_count = len(frame_count)
                                        st.caption(f"Frames analyzed: {frame_count}")
                                        volatility = details.get("volatility")
                                        if volatility is not None:
                                            st.caption(f"Motion Volatility: {volatility:.4f}")
                                        encoder = details.get("encoder")
                                        if encoder:
                                            st.caption(f"Encoder: {encoder}")
                                        features = details.get("features", [])
                                        if features:
                                            st.caption(f"Analysis: {', '.join(features)}")
                                        backbone = details.get("backbone")
                                        if backbone:
                                            st.caption(f"Backbone: {backbone}")
                                        instability = details.get("instability")
                                        if instability is not None and volatility is None:
                                            st.caption(f"Frame variance: {instability:.4f}")
                                        temporal_sim = details.get("temporal_sim")
                                        if temporal_sim is not None:
                                            st.caption(f"Temporal consistency (cos-sim): {temporal_sim:.4f}")
                                        spa = details.get("spatial_anomaly")
                                        if spa is not None:
                                            st.caption(f"Spatial anomaly: {spa:.4f}")
                                        note = details.get("note")
                                        if note:
                                            st.info(note)

                        errors = result.get("errors", [])
                        if errors:
                            st.error("### ⚠️ Errors")
                            for err in errors:
                                st.error(err)

                        with st.expander("📄 Raw JSON Response"):
                            st.json(result)

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Cleanup
    try:
        os.unlink(video_path)
    except:
        pass

else:
    # Show instructions when no file uploaded
    st.info("👆 Upload a video file to begin analysis")
    
    st.divider()
    
    # Show all 3 detector options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🧠 DINOv3")
        st.markdown("""
        **Best for:** High-accuracy detection
        
        **Features:**
        - Face cropping (MTCNN)
        - 768-dim embeddings
        - LayerNorm tuning
        - 0.88+ AUROC
        
        **Requires:** Trained weights
        """)
    
    with col2:
        st.subheader("📊 D3")
        st.markdown("""
        **Best for:** Training-free detection
        
        **Features:**
        - Second-order features
        - Temporal volatility
        - No training needed
        - ICCV 2025
        
        **Requires:** No weights
        """)
    
    with col3:
        st.subheader("🎤 LipFD")
        st.markdown("""
        **Best for:** Lip-sync deepfakes
        
        **Features:**
        - Audio-visual analysis
        - CLIP + ResNet-50
        - Mel-spectrogram input
        - NeurIPS 2024
        
        **Requires:** src/deepfake_guard/weights/lipfd_ckpt.pth
        """)

# Footer
st.divider()
st.caption("Deepfake Guard v0.5.0 | Ensemble: domain-aware trust × certainty with outlier veto | DINOv3, D3, LipFD")
