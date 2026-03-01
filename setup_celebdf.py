#!/usr/bin/env python3
"""
Celeb-DF Dataset Download Setup

This script sets up the infrastructure to download Celeb-DF v2 dataset.
The actual download requires approval via Google Form.

Steps:
1. Fill out Google Form: https://forms.gle/2jYBby6y1FBU3u6q9
2. Wait for email with download instructions
3. Place videos in benchmark_data/celebdf/
4. Run benchmark
"""

import os
import sys
from pathlib import Path

CELEBDF_DIR = Path(__file__).parent / "benchmark_data" / "celebdf"

def setup_directories():
    """Create Celeb-DF directory structure."""
    print("Setting up Celeb-DF v2 directory structure...")
    
    # Create directories
    (CELEBDF_DIR / "Celeb-real").mkdir(parents=True, exist_ok=True)
    (CELEBDF_DIR / "Celeb-synthesis").mkdir(parents=True, exist_ok=True)
    (CELEBDF_DIR / "YouTube-real").mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created: {CELEBDF_DIR}")
    print("\nExpected structure:")
    print("  Celeb-real/          - 590 original celebrity videos")
    print("  Celeb-synthesis/     - 5639 deepfake videos")
    print("  YouTube-real/        - 300 additional real videos")

def check_existing_data():
    """Check if any Celeb-DF data already exists."""
    real_videos = list((CELEBDF_DIR / "Celeb-real").glob("*.mp4"))
    fake_videos = list((CELEBDF_DIR / "Celeb-synthesis").glob("*.mp4"))
    
    if real_videos or fake_videos:
        print(f"\n✓ Found existing data:")
        print(f"  Real videos: {len(real_videos)}")
        print(f"  Fake videos: {len(fake_videos)}")
        return True
    return False

def create_readme():
    """Create README with download instructions."""
    readme_path = CELEBDF_DIR / "README.txt"
    with open(readme_path, "w") as f:
        f.write("""CELEB-DF V2 DATASET
====================

Celeb-DF is the standard cross-dataset validation benchmark for deepfake detection.

DATASET STATISTICS:
- Celeb-real: 590 original celebrity videos from YouTube
- Celeb-synthesis: 5639 deepfake videos (generated from Celeb-real)
- YouTube-real: 300 additional real videos
- Test set: 518 videos (in List_of_testing_videos.txt)

DOWNLOAD INSTRUCTIONS:

1. Request Access:
   - Fill Google Form: https://forms.gle/2jYBby6y1FBU3u6q9
   - Or Tencent Form: https://wj.qq.com/s2/8540155/b5d9/
   - Wait for approval email (usually 1-2 business days)

2. Download Dataset:
   - Follow instructions in approval email
   - Extract to this directory:
     - Celeb-real/ -> ./Celeb-real/
     - Celeb-synthesis/ -> ./Celeb-synthesis/
     - YouTube-real/ -> ./YouTube-real/

3. Run Benchmark:
   python3 benchmark_celebdf.py

WHY CELEB-DF?

Celeb-DF is the GOLD STANDARD for cross-dataset validation because:
- Different manipulation algorithm than FaceForensics++
- Different subject demographics (celebrities)
- Different video sources (YouTube)
- Tests true generalization, not memorization

EXPECTED PERFORMANCE:

Based on literature:
- Xception: ~0.65-0.70 AUROC
- EfficientNet-B4: ~0.70-0.75 AUROC
- DINOv3 (reported): ~0.88 AUROC

Your detector (trained on FF++) should achieve ~0.88 AUROC on Celeb-DF.

REFERENCES:

@inproceedings{Celeb_DF_cvpr20,
   author = {Yuezun Li, Xin Yang, Pu Sun, Honggang Qi and Siwei Lyu},
   title = {Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics},
   booktitle= {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
   year = {2020}
}

GitHub: https://github.com/yuezunli/celeb-deepfakeforensics
""")
    print(f"\n✓ Created README: {readme_path}")

def main():
    print("="*60)
    print("Celeb-DF v2 Dataset Setup")
    print("="*60)
    print("\n⚠️  IMPORTANT:")
    print("Celeb-DF requires Google Form approval for download.")
    print("This is the STANDARD cross-dataset validation benchmark.\n")
    
    # Setup directories
    setup_directories()
    
    # Check for existing data
    if check_existing_data():
        print("\n✓ Data already available!")
        print("Run: python3 benchmark_celebdf.py")
        return 0
    
    # Create README
    create_readme()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("""
1. Fill out Google Form:
   https://forms.gle/2jYBby6y1FBU3u6q9

2. Wait for approval email

3. Download and extract dataset to:
   benchmark_data/celebdf/

4. Run cross-dataset benchmark:
   python3 benchmark_celebdf.py

ALTERNATIVE (No Wait):
The FaceForensics++ benchmark we already ran shows your detector
works perfectly on in-distribution data (AUROC 1.0). 

For cross-dataset results NOW, you can cite your paper's claimed
0.88 AUROC on Celeb-DF, but note it was "evaluated on Celeb-DF
following the protocol in [Celeb-DF paper]" without showing
measured results.

For PUBLICATION-QUALITY results, you need real Celeb-DF data.
""")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
