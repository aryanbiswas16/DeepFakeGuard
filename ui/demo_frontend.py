import streamlit as st
import requests

st.set_page_config(page_title="Deepfake Guard API Client")

st.title("🕵️ Deepfake Guard API Client")

# Configuration
API_URL = st.text_input("API URL", "http://localhost:8000")

# Detector selection
detector = st.selectbox(
    "Select Detector",
    options=["dinov3", "resnet18", "ivyfake", "d3"],
    format_func=lambda x: {
        "dinov3": "\U0001f9e0 DINOv3 (ViT-B/16 - High Accuracy)",
        "resnet18": "\U0001f3af ResNet18 (CNN - Fast)",
        "ivyfake": "\U0001f33f IvyFake (CLIP - Explainable)",
        "d3": "\U0001f4ca D3 (Training-Free - Temporal)"
    }[x],
    help="Choose the detection backend"
)

# Show detector info
info_text = {
    "dinov3": "Face-based detection with DINOv3. Requires trained weights. Best accuracy (0.88+ AUROC).",
    "resnet18": "Full-frame detection with ResNet18. Uses pretrained ImageNet weights. Fast but less accurate.",
    "ivyfake": "CLIP-based detection with temporal and spatial artifact analysis. Explainable outputs, no weights needed.",
    "d3": "Training-free detection using second-order temporal features (ICCV 2025). Analyzes motion volatility."
}
st.caption(f"**{detector.upper()}:** {info_text[detector]}")

st.divider()

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("Analyze Video"):
        with st.spinner(f"Sending video to API ({detector} detector)..."):
            try:
                # Prepare the file for upload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Call API with detector parameter
                detect_url = f"{API_URL.rstrip('/')}/detect"
                response = requests.post(
                    detect_url, 
                    files=files,
                    params={"detector": detector}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.subheader("Results")

                    # Display Model Info
                    model_info = result.get("model_info", {})
                    if model_info:
                        det_type = model_info.get("detector_type", detector)
                        st.caption(f"Detector: {det_type.upper()} | Version: {model_info.get('version', '?.?')}")
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    
                    score = result.get("overall_score")
                    label = result.get("overall_label")
                    
                    if score is not None:
                        with col1:
                            st.metric("Score", f"{score:.4f}")
                        with col2:
                            if label == "FAKE":
                                st.error(f"Verdict: {label}")
                            else:
                                st.success(f"Verdict: {label}")
                                
                        st.progress(score)
                    
                    # Show detailed modality results
                    modality_results = result.get("modality_results", {})
                    if modality_results:
                        st.subheader("Detailed Analysis")
                        for modality, res in modality_results.items():
                            with st.expander(f"{modality.title()} Details"):
                                st.write(f"Score: {res.get('score', 'N/A')}")
                                st.write(f"Label: {res.get('label', 'N/A')}")
                                
                                # Show details if available
                                details = res.get("details", {})
                                if details:
                                    det_type = details.get("detector_type")
                                    if det_type:
                                        st.caption(f"Detector Type: {det_type}")
                                    
                                    frame_count = details.get("frame_count")
                                    if frame_count:
                                        st.caption(f"Frames: {frame_count}")
                                    
                                    features = details.get("features", [])
                                    if features:
                                        st.caption(f"Features: {', '.join(features)}")
                                    
                                    backbone = details.get("backbone")
                                    if backbone:
                                        st.caption(f"Backbone: {backbone}")
                                    
                                    note = details.get("note")
                                    if note:
                                        st.info(note)
                    
                    # Show errors if any
                    errors = result.get("errors", [])
                    if errors:
                        st.error("Errors occurred:")
                        for err in errors:
                            st.error(err)
                    
                    # Show full details
                    with st.expander("Full JSON Response"):
                        st.json(result)
                        
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to API. Is the server running? (uvicorn app.main:app --reload)")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")