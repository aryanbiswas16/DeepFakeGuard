#!/usr/bin/env python3
"""
FaceForensics++ Dataset Download Setup

This script sets up the infrastructure to download FaceForensics++ dataset.
The actual download requires approval via Google Form.

Steps:
1. Fill out Google Form: https://forms.gle/xxxx (link in README)
2. Wait for email with download script link
3. Run this script to download the dataset
4. Run benchmarks
"""

import os
import sys
import json
import subprocess
from pathlib import Path

FF_DATASET_DIR = Path(__file__).parent / "benchmark_data" / "faceforensics"
FF_TOOLS_DIR = Path(__file__).parent / "faceforensics_tools"


def setup_directories():
    """Create FaceForensics directory structure."""
    print("Setting up FaceForensics++ directory structure...")
    
    # Create directories
    (FF_DATASET_DIR / "original_sequences" / "youtube" / "c23" / "videos").mkdir(parents=True, exist_ok=True)
    (FF_DATASET_DIR / "manipulated_sequences" / "Deepfakes" / "c23" / "videos").mkdir(parents=True, exist_ok=True)
    (FF_DATASET_DIR / "manipulated_sequences" / "Face2Face" / "c23" / "videos").mkdir(parents=True, exist_ok=True)
    (FF_DATASET_DIR / "manipulated_sequences" / "FaceSwap" / "c23" / "videos").mkdir(parents=True, exist_ok=True)
    (FF_DATASET_DIR / "manipulated_sequences" / "NeuralTextures" / "c23" / "videos").mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created: {FF_DATASET_DIR}")


def check_download_script():
    """Check if download script is available."""
    script_path = FF_TOOLS_DIR / "dataset" / "download-FaceForensics.py"
    
    if script_path.exists():
        print(f"✓ Download script found: {script_path}")
        return str(script_path)
    
    # Check if user has placed it manually
    manual_paths = [
        Path.home() / "Downloads" / "download-FaceForensics.py",
        Path.home() / "download-FaceForensics.py",
        Path(__file__).parent / "download-FaceForensics.py",
    ]
    
    for path in manual_paths:
        if path.exists():
            print(f"✓ Download script found: {path}")
            return str(path)
    
    return None


def download_dataset(download_script: str, num_videos: int = None):
    """Download FaceForensics++ dataset."""
    print("\n" + "="*60)
    print("Downloading FaceForensics++ Dataset")
    print("="*60)
    
    cmd = [
        "python3", download_script,
        str(FF_DATASET_DIR),
        "-d", "all",
        "-c", "c23",
        "-t", "videos"
    ]
    
    if num_videos:
        cmd.extend(["--num_videos", str(num_videos)])
        print(f"Downloading {num_videos} videos (sample mode)")
    else:
        print("Downloading full dataset (~10GB for c23 quality)")
    
    print(f"Command: {' '.join(cmd)}")
    print("\nStarting download...")
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def create_benchmark_labels():
    """Create labels.csv for benchmark from FaceForensics splits."""
    splits_dir = FF_TOOLS_DIR / "dataset" / "splits"
    
    if not splits_dir.exists():
        print("✗ Splits directory not found")
        return False
    
    # Load test split
    with open(splits_dir / "test.json") as f:
        test_videos = json.load(f)
    
    # Create labels for benchmark
    benchmark_dir = Path(__file__).parent / "benchmark_data"
    labels_file = benchmark_dir / "faceforensics_labels.csv"
    
    with open(labels_file, "w") as f:
        f.write("video_path,label\n")
        
        # Original videos (real)
        for pair in test_videos:
            for vid_id in pair:
                vid_path = f"faceforensics/original_sequences/youtube/c23/videos/{vid_id}.mp4"
                f.write(f"{vid_path},REAL\n")
        
        # Manipulated videos (fake) - Deepfakes
        for pair in test_videos:
            vid_path = f"faceforensics/manipulated_sequences/Deepfakes/c23/videos/{pair[0]}_{pair[1]}.mp4"
            f.write(f"{vid_path},FAKE\n")
            vid_path = f"faceforensics/manipulated_sequences/Deepfakes/c23/videos/{pair[1]}_{pair[0]}.mp4"
            f.write(f"{vid_path},FAKE\n")
    
    print(f"✓ Created labels: {labels_file}")
    print(f"  Test set: {len(test_videos)} pairs = {len(test_videos)*2} real + {len(test_videos)*2} fake videos")
    return True


def run_benchmark():
    """Run benchmark on downloaded FaceForensics data."""
    print("\n" + "="*60)
    print("Running Benchmark")
    print("="*60)
    
    # Import and run universal benchmark
    sys.path.insert(0, str(Path(__file__).parent))
    from universal_benchmark import UniversalBenchmark
    
    benchmark = UniversalBenchmark(data_dir=str(FF_DATASET_DIR))
    
    # Override to use FaceForensics structure
    # This would need custom handling for the specific structure
    
    print("To run benchmark manually:")
    print("  python3 run_benchmark.py")
    print("  # or")
    print("  python3 universal_benchmark.py")


def main():
    print("="*60)
    print("FaceForensics++ Dataset Setup")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # Check for download script
    download_script = check_download_script()
    
    if not download_script:
        print("\n" + "="*60)
        print("DOWNLOAD SCRIPT NOT FOUND")
        print("="*60)
        print("""
The FaceForensics++ download script is not included in the GitHub repo.
You need to:

1. Fill out the Google Form:
   https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform

2. Wait for approval email (usually 1-2 business days)

3. Download the script from the link provided in the email

4. Place download-FaceForensics.py in one of these locations:
   - ./faceforensics_tools/dataset/
   - ~/Downloads/
   - ./ (project root)

5. Run this script again

ALTERNATIVE (Quick Test):
Use the synthetic data already generated:
   python3 universal_benchmark.py --synthetic
""")
        return 1
    
    # Ask how many videos to download
    print("\n" + "="*60)
    print("Download Options")
    print("="*60)
    print("1. Full test set (140 pairs = ~1GB)")
    print("2. Sample (10 pairs = ~100MB)")
    print("3. Cancel")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        success = download_dataset(download_script)
    elif choice == "2":
        success = download_dataset(download_script, num_videos=10)
    else:
        print("Cancelled.")
        return 0
    
    if success:
        print("\n✓ Download complete!")
        create_benchmark_labels()
        run_benchmark()
    else:
        print("\n✗ Download failed. Check error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
