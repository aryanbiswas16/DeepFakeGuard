#!/usr/bin/env python3
"""
Download benchmark datasets for DeepFakeGuard.
Tries multiple sources: Kaggle, Google Drive, direct links.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def check_kaggle():
    """Check if Kaggle CLI is available and configured."""
    try:
        result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def download_kaggle_deepfake():
    """Download deepfake detection challenge data from Kaggle."""
    print("Attempting to download from Kaggle...")
    
    # Deepfake Detection Challenge dataset
    datasets = [
        "deepfake-detection-challenge",
        "deepfake-detection-faces",
    ]
    
    for dataset in datasets:
        cmd = f"kaggle competitions download -c {dataset} -p ./benchmark_data/kaggle/"
        print(f"Trying: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Success! Downloaded {dataset}")
            return True
        else:
            print(f"Failed: {result.stderr[:200]}")
    
    return False

def download_sample_videos():
    """Download sample deepfake videos from publicly available sources."""
    import urllib.request
    
    # Some public sample URLs (if available)
    # For now, we'll create a placeholder
    print("Creating sample structure...")
    
    os.makedirs("./benchmark_data/real", exist_ok=True)
    os.makedirs("./benchmark_data/fake", exist_ok=True)
    
    return True

def main():
    print("="*60)
    print("DeepFakeGuard Benchmark Data Downloader")
    print("="*60)
    
    # Check Kaggle
    if check_kaggle():
        print("✓ Kaggle CLI found")
        if download_kaggle_deepfake():
            print("\n✓ Download complete!")
            return 0
    else:
        print("✗ Kaggle CLI not found or not configured")
        print("  Install: pip install kaggle")
        print("  Setup: https://github.com/Kaggle/kaggle-api#api-credentials")
    
    print("\n" + "="*60)
    print("ALTERNATIVE: Manual Download Required")
    print("="*60)
    print("""
To get benchmark data, you have these options:

1. FACE FORENSICS++ (Recommended)
   - Request access: https://github.com/ondyari/FaceForensics
   - Fill out Google Form
   - Download script will be emailed

2. CELEB-DF
   - Request access: https://github.com/yuezunli/celeb-deepfakeforensics
   - Fill out Google Form

3. KAGGLE DEEPFAKE DETECTION CHALLENGE
   - https://www.kaggle.com/c/deepfake-detection-challenge
   - Requires Kaggle account + competition rules acceptance
   - Run: kaggle competitions download -c deepfake-detection-challenge

4. CREATE SMALL TEST SET
   - Collect 10-20 real videos from YouTube/phone
   - Collect 10-20 fake videos from online sources
   - Organize in benchmark_data/real/ and benchmark_data/fake/
""")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
