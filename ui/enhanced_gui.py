"""
Enhanced Deepfake Guard GUI with detector toggle
Supports DINOv3, ResNet18, IvyFake, and D3 detectors
"""

import streamlit as st
from pathlib import Path
import tempfile
import os

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
    options=["dinov3", "resnet18", "ivyfake", "d3"],
    format_func=lambda x: {
        "dinov3": "🧠 DINOv3 (ViT-B/16 - 0.88 AUROC)",
        "resnet18": "🎯 ResNet18 (CNN - Pretrained)",
        "ivyfake": "🌿 IvyFake (CLIP - Explainable)",
        "d3": "📊 D3 (Training-Free - Temporal)"
    }[x],
    help="Switch between different detection backends"
)

detector_info = {
    "dinov3": {
        "name": "DINOv3 Vision Transformer",
        "description": "Face-based detection with DINOv3 ViT-B/16",
        "features": ["Face cropping (MTCNN)", "768-dim embeddings", "LayerNorm tuning", "0.88+ AUROC"],
        "pros": "Higher accuracy, trained on deepfakes",
        "cons": "Requires face detection, slower",
        "requires_weights": True
    },
    "resnet18": {
        "name": "ResNet18 CNN",
        "description": "Full-frame detection with ResNet18",
        "features": ["Full frame analysis", "No face cropping", "Lightweight", "Pretrained on ImageNet"],
        "pros": "Faster, no face dependency",
        "cons": "Lower accuracy (not fine-tuned)",
        "requires_weights": False
    },
    "ivyfake": {
        "name": "IvyFake CLIP Detector",
        "description": "CLIP-based explainable AIGC detection",
        "features": [
            "CLIP ViT-B/32 backbone",
            "Temporal artifact analysis",
            "Spatial artifact analysis",
            "Explainable outputs",
            "No face cropping required"
        ],
        "pros": "Explainable, artifact detection, pretrained",
        "cons": "First run downloads CLIP model",
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
    }
}

# Show detector info
info = detector_info[detector_type]
st.sidebar.markdown(f"**{info['name']}**")
st.sidebar.markdown(f"_{info['description']}_")
st.sidebar.markdown("**Features:**")
for feat in info['features']:
    st.sidebar.markdown(f"- {feat}")

st.sidebar.markdown(f"✅ **Pros:** {info['pros']}")
st.sidebar.markdown(f"⚠️ **Cons:** {info['cons']}")

# D3 encoder selection
if detector_type == "d3":
    st.sidebar.divider()
    d3_encoder = st.sidebar.selectbox(
        "D3 Encoder",
        options=["xclip-16", "xclip-32", "resnet-18", "mobilenet-v3"],
        format_func=lambda x: {
            "xclip-16": "XCLIP-16 (Recommended)",
            "xclip-32": "XCLIP-32",
            "resnet-18": "ResNet-18",
            "mobilenet-v3": "MobileNet-v3 (Fastest)"
        }[x],
        help="Encoder backbone for D3 feature extraction"
    )
else:
    d3_encoder = None

# Weights path (only for DINOv3)
weights_path = None
if detector_type == "dinov3":
    st.sidebar.divider()
    weights_path = st.sidebar.text_input(
        "Weights Path",
        value="weights/dinov3_best_v3.pth",
        help="Path to DINOv3 weights file"
    )

# Initialize detector
@st.cache_resource
def load_detector(det_type: str, weights: str = None, d3_enc: str = "xclip-16"):
    """Load detector with caching."""
    import sys
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from deepfake_guard import DeepfakeGuard
    
    try:
        # For D3, pass encoder info through detector type
        guard = DeepfakeGuard(
            weights_path=weights if weights and os.path.exists(weights) else None,
            detector_type=det_type
        )
        # If D3, initialize with specific encoder
        if det_type == "d3" and d3_enc:
            guard._init_d3(encoder=d3_enc)
        return guard, None
    except Exception as e:
        return None, str(e)

with st.spinner(f"Loading {detector_type.upper()} detector..."):
    guard, error = load_detector(detector_type, weights_path, d3_encoder)

if error:
    st.error(f"Failed to load detector: {error}")
    st.stop()

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
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.getvalue())
        video_path = tmp.name
    
    # Display video
    col_video, col_results = st.columns([1, 1])
    
    with col_video:
        st.subheader("📺 Video Preview")
        st.video(uploaded_file)
        st.caption(f"File: {uploaded_file.name} ({len(uploaded_file.getvalue()) / 1024 / 1024:.1f} MB)")
    
    with col_results:
        st.subheader("🎯 Detection Results")
        
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            with st.spinner("Analyzing video... This may take a moment."):
                try:
                    # Run detection
                    result = guard.detect_video(video_path)
                    
                    # Display results
                    overall_score = result.get("overall_score", 0)
                    overall_label = result.get("overall_label", "UNKNOWN")
                    
                    # Main result card
                    if overall_label == "FAKE":
                        st.error(f"## 🚨 FAKE DETECTED")
                    elif overall_label == "REAL":
                        st.success(f"## ✅ REAL VIDEO")
                    else:
                        st.warning(f"## ⚠️ UNKNOWN")
                    
                    # Score metrics
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Confidence", f"{overall_score:.1%}")
                    with cols[1]:
                        st.metric("Threshold", f"{threshold:.0%}")
                    with cols[2]:
                        confidence = abs(overall_score - 0.5) * 2
                        st.metric("Certainty", f"{confidence:.1%}")
                    
                    # Progress bar
                    st.progress(overall_score)
                    
                    # Modality results
                    st.markdown("### 📊 Detailed Analysis")
                    for modality, res in result.get("modality_results", {}).items():
                        with st.expander(f"{modality.title()} Analysis", expanded=True):
                            cols = st.columns([1, 2])
                            with cols[0]:
                                score = res.get("score", 0)
                                label = res.get("label", "UNKNOWN")
                                
                                if label == "FAKE":
                                    st.error(f"**{label}**

Score: {score:.3f}")
                                else:
                                    st.success(f"**{label}**

Score: {score:.3f}")
                            
                            with cols[1]:
                                details = res.get("details", {})
                                
                                # Show detector type
                                det_type = details.get("detector_type", detector_type)
                                st.caption(f"Detector: {det_type.upper()}")
                                
                                # Show frame count
                                frame_count = details.get("frame_count") or details.get("per_frame_fake_probs", [])
                                if isinstance(frame_count, list):
                                    frame_count = len(frame_count)
                                st.caption(f"Frames analyzed: {frame_count}")
                                
                                # Show D3-specific info
                                volatility = details.get("volatility")
                                if volatility is not None:
                                    st.caption(f"Motion Volatility: {volatility:.4f}")
                                
                                encoder = details.get("encoder")
                                if encoder:
                                    st.caption(f"Encoder: {encoder}")
                                
                                # Show features for IvyFake
                                features = details.get("features", [])
                                if features:
                                    st.caption(f"Analysis: {', '.join(features)}")
                                
                                # Show backbone info
                                backbone = details.get("backbone")
                                if backbone:
                                    st.caption(f"Backbone: {backbone}")
                                
                                # Show instability if available
                                instability = details.get("instability")
                                if instability is not None and volatility is None:
                                    st.caption(f"Frame variance: {instability:.4f}")
                                
                                # Show any notes
                                note = details.get("note")
                                if note:
                                    st.info(note)
                    
                    # Errors
                    errors = result.get("errors", [])
                    if errors:
                        st.error("### ⚠️ Errors")
                        for err in errors:
                            st.error(err)
                    
                    # Full JSON
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
    
    # Show all 4 detector options
    col1, col2, col3, col4 = st.columns(4)
    
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
        st.subheader("🎯 ResNet18")
        st.markdown("""
        **Best for:** Quick analysis
        
        **Features:**
        - Full frame analysis
        - No face cropping
        - Lightweight
        - ImageNet pretrained
        
        **Requires:** No weights
        """)
    
    with col3:
        st.subheader("🌿 IvyFake")
        st.markdown("""
        **Best for:** Explainable detection
        
        **Features:**
        - CLIP ViT-B/32
        - Temporal artifacts
        - Spatial artifacts
        - Explainable outputs
        
        **Requires:** No weights
        """)
    
    with col4:
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

# Footer
st.divider()
st.caption("Deepfake Guard v0.4.0 | Multi-detector support: DINOv3, ResNet18, IvyFake, D3")
