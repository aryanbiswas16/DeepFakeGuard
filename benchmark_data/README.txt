DEEPFAKE BENCHMARK DATASET
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
