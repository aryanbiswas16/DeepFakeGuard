from fastapi import FastAPI, UploadFile, File
from deepfake_guard import DeepfakeGuard
import shutil
import tempfile
import os

app = FastAPI()

# Configuration
# Users should set DEEPFAKE_GUARD_WEIGHTS env var or place weights in the running directory
WEIGHTS_PATH = os.environ.get("DEEPFAKE_GUARD_WEIGHTS", "dinov3_best_v3.pth")

print(f"Loading model with weights: {WEIGHTS_PATH}")
# Initialize the guard
# Note: DeepfakeGuard handles the case where weights are missing by printing a warning
guard = DeepfakeGuard(weights_path=WEIGHTS_PATH)

@app.post("/detect")
async def detect_endpoint(file: UploadFile = File(...)):
    """
    Detect deepfakes in an uploaded video file.
    """
    # Create a temp file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        # Run detection
        result = guard.detect_video(tmp_path)
        return result
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/")
def read_root():
    return {"status": "DeepfakeGuard API is running", "model_loaded": guard.visual_weights_loaded}

