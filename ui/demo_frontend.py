import streamlit as st
import requests

st.set_page_config(page_title="Deepfake Guard Client")

st.title("🕵️ Deepfake Guard API Client")

# Configuration
API_URL = st.text_input("API URL", "http://localhost:8000/detect")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("Analyze Video"):
        with st.spinner("Sending video to API..."):
            try:
                # Prepare the file for upload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Call API
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.subheader("Results")

                    # Display Model Info
                    model_info = result.get("model_info", {})
                    if model_info:
                        st.caption(f"Model: {model_info.get('visual_detector', 'Unknown')} (v{model_info.get('version', '?.?')})")
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    
                    score = result.get("score")
                    label = result.get("label")
                    
                    if score is not None:
                        with col1:
                            st.metric("Score", f"{score:.4f}")
                        with col2:
                            if label == "FAKE":
                                st.error("Verdict: FAKE")
                            else:
                                st.success("Verdict: REAL")
                                
                        st.progress(score)
                    
                    # Show full details
                    with st.expander("Full JSON Response"):
                        st.json(result)
                        
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to API. Is the server running? (uvicorn app.main:app --reload)")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
