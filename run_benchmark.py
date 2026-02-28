#!/usr/bin/env python3
"""
DeepFakeGuard - Complete Benchmark Runner
Handles data download and evaluation automatically.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent / "benchmark_data"
RESULTS_FILE = Path(__file__).parent / "benchmark_results.json"

def log(msg):
    print(f"[BENCHMARK] {msg}")

def setup_environment():
    """Setup all necessary directories and dependencies."""
    log("Setting up environment...")
    BENCHMARK_DIR.mkdir(exist_ok=True)
    (BENCHMARK_DIR / "real").mkdir(exist_ok=True)
    (BENCHMARK_DIR / "fake").mkdir(exist_ok=True)
    log("✓ Directories created")

def check_kaggle_auth():
    """Check if Kaggle is authenticated."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists() or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))

def download_from_kaggle():
    """Attempt to download deepfake dataset from Kaggle."""
    if not check_kaggle_auth():
        log("✗ Kaggle not authenticated")
        return False
    
    log("Attempting Kaggle download...")
    
    # Try deepfake detection challenge
    competitions = [
        "deepfake-detection-challenge",
        "deepfake-detection-faces"
    ]
    
    for comp in competitions:
        log(f"Trying competition: {comp}")
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", comp, "-p", str(BENCHMARK_DIR)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            log(f"✓ Downloaded {comp}")
            return True
        else:
            log(f"✗ Failed: {result.stderr[:100]}")
    
    return False

def create_sample_benchmark():
    """Create a minimal benchmark with instructions."""
    log("Creating sample benchmark structure...")
    
    readme_path = BENCHMARK_DIR / "README.txt"
    with open(readme_path, "w") as f:
        f.write("""DEEPFAKE BENCHMARK DATASET
==========================

This directory should contain:
- real/ : Real videos (YouTube, phone recordings, etc.)
- fake/ : Deepfake/manipulated videos

QUICK START:
1. Download sample videos from:
   - FaceForensics++: https://github.com/ondyari/FaceForensics
   - Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics
   - Kaggle: https://www.kaggle.com/c/deepfake-detection-challenge

2. Place videos in appropriate folders

3. Run: python3 run_benchmark.py

MINIMUM REQUIREMENTS:
- 10 real videos
- 10 fake videos
- For meaningful results: 50+ each
""")
    
    log(f"✓ Created {readme_path}")
    return True

def generate_synthetic_report():
    """Generate a report explaining what results would look like."""
    report = {
        "status": "NO_REAL_DATA",
        "message": "Real benchmark data not available",
        "instructions": "Download real deepfake datasets to get actual metrics",
        "expected_structure": {
            "benchmark_data/real/": "Real videos",
            "benchmark_data/fake/": "Fake/deepfake videos"
        },
        "datasets": [
            {
                "name": "FaceForensics++",
                "url": "https://github.com/ondyari/FaceForensics",
                "size": "1.8TB (full), ~50GB (c23 quality)",
                "videos": "1000 real + 4000 fake"
            },
            {
                "name": "Celeb-DF",
                "url": "https://github.com/yuezunli/celeb-deepfakeforensics",
                "size": "~8GB",
                "videos": "590 real + 5639 fake"
            },
            {
                "name": "Kaggle Deepfake Detection",
                "url": "https://www.kaggle.com/c/deepfake-detection-challenge",
                "size": "~4GB",
                "videos": "Training set available"
            }
        ],
        "what_youll_get": {
            "dinov3": {
                "auroc": "TBD (target: 0.85-0.90)",
                "accuracy": "TBD",
                "inference_ms": "~8ms per frame"
            },
            "xception_baseline": {
                "literature_auroc": 0.81,
                "source": "FaceForensics++ paper"
            },
            "efficientnet_b4": {
                "literature_auroc": 0.84,
                "source": "Literature estimate"
            }
        }
    }
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(report, f, indent=2)
    
    log(f"✓ Report saved to {RESULTS_FILE}")
    return report

def run_full_pipeline():
    """Run the complete benchmark pipeline."""
    log("="*60)
    log("Starting DeepFakeGuard Benchmark Pipeline")
    log("="*60)
    
    # Setup
    setup_environment()
    
    # Try to download data
    data_available = False
    
    if check_kaggle_auth():
        log("Kaggle credentials found, attempting download...")
        data_available = download_from_kaggle()
    
    if not data_available:
        # Check for existing data
        real_videos = list((BENCHMARK_DIR / "real").glob("*.mp4"))
        fake_videos = list((BENCHMARK_DIR / "fake").glob("*.mp4"))
        
        if real_videos and fake_videos:
            log(f"✓ Found existing data: {len(real_videos)} real, {len(fake_videos)} fake")
            data_available = True
        else:
            log("No real benchmark data found")
            create_sample_benchmark()
    
    if not data_available:
        log("\n" + "="*60)
        log("CANNOT RUN FULL BENCHMARK")
        log("="*60)
        log("Real deepfake datasets require authentication.")
        log("See benchmark_data/README.txt for instructions.")
        
        report = generate_synthetic_report()
        
        log("\n" + "="*60)
        log("NEXT STEPS:")
        log("="*60)
        log("1. Download a dataset (FaceForensics++ recommended)")
        log("2. Extract to benchmark_data/real/ and benchmark_data/fake/")
        log("3. Run: python3 run_benchmark.py")
        log("\nQuick test with synthetic data:")
        log("  python3 test_detectors.py")
        
        return 1
    
    # If we have data, run actual benchmark
    log("\n" + "="*60)
    log("RUNNING BENCHMARK")
    log("="*60)
    
    # Import and run benchmark
    sys.path.insert(0, str(Path(__file__).parent))
    from benchmark_real import BenchmarkRunner, load_labels_from_csv
    
    # Create labels CSV from directory structure
    labels_csv = BENCHMARK_DIR / "labels.csv"
    with open(labels_csv, "w") as f:
        f.write("video_path,label\n")
        for v in (BENCHMARK_DIR / "real").glob("*.mp4"):
            f.write(f"real/{v.name},REAL\n")
        for v in (BENCHMARK_DIR / "fake").glob("*.mp4"):
            f.write(f"fake/{v.name},FAKE\n")
    
    video_paths, ground_truth = load_labels_from_csv(str(labels_csv))
    video_paths = [str(BENCHMARK_DIR / p) for p in video_paths]
    
    runner = BenchmarkRunner()
    results = runner.run_full_benchmark(video_paths, ground_truth)
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    log("\n" + "="*60)
    log("RESULTS SUMMARY")
    log("="*60)
    for detector, metrics in results.items():
        log(f"\n{detector.upper()}:")
        if "error" in metrics:
            log(f"  ERROR: {metrics['error']}")
        else:
            log(f"  AUROC: {metrics.get('auroc', 'N/A')}")
            log(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
            log(f"  Inference: {metrics.get('avg_inference_time', 0):.3f}s")
    
    log(f"\n✓ Full results saved to {RESULTS_FILE}")
    return 0

if __name__ == "__main__":
    sys.exit(run_full_pipeline())
