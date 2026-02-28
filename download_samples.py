#!/usr/bin/env python3
"""
Download sample deepfake videos from publicly available sources.
"""

import os
import sys
import urllib.request
import ssl
from pathlib import Path

# Disable SSL verification for some hosts
ssl._create_default_https_context = ssl._create_unverified_context

SAMPLE_DIR = Path(__file__).parent / "benchmark_data" / "samples"

def download_file(url, output_path):
    """Download a file from URL."""
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"  ✓ Saved to {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def main():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    (SAMPLE_DIR / "real").mkdir(exist_ok=True)
    (SAMPLE_DIR / "fake").mkdir(exist_ok=True)
    
    print("="*60)
    print("Downloading Sample Deepfake Videos")
    print("="*60)
    
    # These are placeholder URLs - actual deepfake samples need to come from
    # legitimate sources. For now, we'll document the process.
    
    print("""
Publicly available deepfake video sources:

1. Kaggle Deepfake Detection Challenge (requires account):
   https://www.kaggle.com/c/deepfake-detection-challenge/data
   - Sample submission videos available
   - ~4GB for sample set

2. Dessa Deepfake Detection:
   https://github.com/dessa-oss/deepfake-detection-challenge
   - Pre-trained models and sample data

3. OpenForensics:
   https://github.com/ondyari/FaceForensics (already cloned)
   - Scripts only, videos require form approval

4. Synthetic Data (already generated):
   - 10 synthetic "real" videos in benchmark_data/synthetic/real/
   - 10 synthetic "fake" videos in benchmark_data/synthetic/fake/

For immediate testing, use synthetic data:
   python3 universal_benchmark.py --synthetic

To get real deepfake videos:
1. Sign up for Kaggle
2. Download from: https://www.kaggle.com/c/deepfake-detection-challenge/data
3. Place in benchmark_data/samples/
""")
    
    # Check what we have
    real_videos = list((SAMPLE_DIR / "real").glob("*.mp4"))
    fake_videos = list((SAMPLE_DIR / "fake").glob("*.mp4"))
    
    print(f"\nCurrent samples:")
    print(f"  Real videos: {len(real_videos)}")
    print(f"  Fake videos: {len(fake_videos)}")
    
    if not real_videos and not fake_videos:
        print("\nNo real sample videos available yet.")
        print("Using synthetic data for now...")
        
        # Link synthetic data as fallback
        synth_dir = Path(__file__).parent / "benchmark_data" / "synthetic"
        if synth_dir.exists():
            print(f"\n✓ Synthetic data available: {synth_dir}")
            print("Run: python3 universal_benchmark.py --synthetic")

if __name__ == "__main__":
    main()
