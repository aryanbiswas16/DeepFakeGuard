"""
Enhanced Deepfake Guard GUI with detector toggle
Supports both DINOv3 and ResNet18 detectors
"""

import streamlit as st
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
    options=["dinov3", "resnet18"],
    format_func=lambda x: {
        "dinov3": "🧠 DINOv3 (Your Model - 0.88 AUROC)",
        "resnet18": "🎯 ResNet18 (Friend's Model - Pretrained)"
    }[x],
    help="Switch between different detection backends"
)

detector_info = {
    "dinov3": {
        "name": "DINOv3 Vision Transformer",
        "description": "Face-based detection with DINOv3 ViT-B/16",
        "features": ["Face cropping (MTCNN)", "768-dim embeddings", "LayerNorm tuning", "0.88+ AUROC"],
        "pros": "Higher accuracy, trained on deepfakes",
        "cons": "Requires face detection, slower"
    },
    "resnet18": {
        "name": "ResNet18 CNN",
        "description": "Full-frame detection with ResNet18",
        "features": ["Full frame analysis", "No face cropping", "Lightweight", "Pretrained on ImageNet"],
        "pros": "Faster, no face dependency",
        "cons": "Lower accuracy (not fine-tuned)"
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
def load_detector(det_type: str, weights: str = None):
    """Load detector with caching."""
    import sys
    sys.path.insert(0, '/Users/drdeathwish/.openclaw/workspace/DeepFakeGuard/src')
    
    from deepfake_guard import DeepfakeGuard
    
    try:
        guard = DeepfakeGuard(
            weights_path=weights if weights and os.path.exists(weights) else None,
            detector_type=det_type
        )
        return guard, None
    except Exception as e:
        return None, str(e)

with st.spinner(f"Loading {detector_type.upper()} detector..."):
    guard, error = load_detector(detector_type, weights_path)

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
                                    st.error(f"**{label}**\n\nScore: {score:.3f}")
                                else:
                                    st.success(f"**{label}**\n\nScore: {score:.3f}")
                            
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
                                
                                # Show instability if available
                                instability = details.get("instability")
                                if instability is not None:
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 DINOv3 Detector (Your Model)")
        st.markdown("""
        **Best for:** High-accuracy detection
        
        **How it works:**
        1. Detects and crops faces from video
        2. Analyzes facial features with DINOv3
        3. Detects subtle artifacts in deepfakes
        
        **Requires:** Trained weights file
        **Accuracy:** 0.88+ AUROC
        """)
    
    with col2:
        st.subheader("🎯 ResNet18 Detector (Friend's Model)")
        st.markdown("""
        **Best for:** Quick analysis, no face dependency
        
        **How it works:**
        1. Samples frames uniformly from video
        2. Analyzes full frames with ResNet18
        3. Uses pretrained ImageNet features
        
        **Requires:** No weights (pretrained)
        **Accuracy:** Baseline (not fine-tuned)
        """)

# Footer
st.divider()
st.caption("Deepfake Guard v0.2.0 | Multi-detector support enabled")