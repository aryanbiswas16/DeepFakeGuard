# DeepFakeGuard Real Benchmark Guide

## ⚠️ The Problem

To get **actual benchmark results** (AUROC, Accuracy, etc.), you need **real test videos** with known ground truth labels (real vs fake).

**Synthetic videos produce meaningless results** because:
- They don't have real/fake characteristics
- Detectors will output random scores
- AUROC will be ~0.50 (no better than random)

---

## ✅ Solution: Use Real Test Data

### Option 1: Download FaceForensics++ (Recommended)

```bash
# 1. Request access at: https://github.com/ondyari/FaceForensics
# 2. Download FaceForensics++ c23 videos
# 3. Organize as:
benchmark_data/
├── real/
│   ├── 000_0000.mp4
│   └── ...
└── fake/
    ├── 000_0000.mp4
    └── ...
```

### Option 2: Use Celeb-DF (Cross-dataset validation)

```bash
# 1. Download from: https://github.com/yuezunli/celeb-deepfakeforensics
# 2. Organize videos with labels
```

### Option 3: Create Small Test Set

Collect ~10-20 videos yourself:
- 5-10 real videos (from your phone, public domain)
- 5-10 fake videos (from online deepfake datasets)

---

## 🚀 Running the Benchmark

I've created `benchmark_real.py` for you. Here's how to use it:

### Step 1: Prepare Your Data

```bash
# Create the data directory
mkdir -p benchmark_data

# Copy your test videos
cp /path/to/real_videos/*.mp4 benchmark_data/
cp /path/to/fake_videos/*.mp4 benchmark_data/

# Create labels.csv
cat > benchmark_data/labels.csv << EOF
video_path,label
real_video_1.mp4,REAL
real_video_2.mp4,REAL
fake_video_1.mp4,FAKE
fake_video_2.mp4,FAKE
EOF
```

### Step 2: Run Benchmark

```bash
cd DeepFakeGuard
python3 benchmark_real.py --data_dir ./benchmark_data --labels ./benchmark_data/labels.csv
```

### Step 3: View Results

Results are saved to `benchmark_results.json`:

```json
{
  "dinov3": {
    "auroc": 0.88,
    "accuracy": 0.86,
    "precision": 0.85,
    "recall": 0.87,
    "f1_score": 0.86,
    "avg_inference_time": 0.008
  },
  "resnet18": {
    "auroc": 0.65,
    ...
  },
  ...
}
```

---

## 📊 What Metrics You'll Get

| Metric | Description |
|--------|-------------|
| **AUROC** | Area Under ROC Curve - best overall metric (0.5 = random, 1.0 = perfect) |
| **Accuracy** | % correctly classified |
| **Precision** | Of predicted fakes, how many are actually fake |
| **Recall** | Of actual fakes, how many were detected |
| **F1 Score** | Harmonic mean of precision and recall |
| **Inference Time** | Seconds per video |

---

## 📝 Example Results Structure

For your research paper, you'll want results like:

```
Dataset: Celeb-DF (cross-dataset validation)
Num videos: 563 (346 real, 217 fake)

DINOv3 (Ours):
  AUROC: 0.88
  Accuracy: 0.86
  Inference: 8 ms/frame

Xception (baseline):
  AUROC: 0.81
  Accuracy: 0.72
  Inference: 12 ms/frame

EfficientNet-B4:
  AUROC: 0.84
  Accuracy: 0.75
  Inference: 15 ms/frame
```

---

## ⚡ Quick Test (Without Real Data)

If you just want to verify detectors run:

```bash
python3 test_detectors.py
```

This creates synthetic videos and checks all detectors initialize correctly. **But the scores will be meaningless (around 0.5).**

---

## 🎯 Next Steps

1. **Download real test data** (FaceForensics++ or Celeb-DF)
2. **Organize with labels.csv**
3. **Run benchmark_real.py**
4. **Use actual metrics** in your paper

---

## 📞 Need Help?

- FaceForensics++ download: https://github.com/ondyari/FaceForensics
- Celeb-DF download: https://github.com/yuezunli/celeb-deepfakeforensics
- Email: aryanbiswas16@gmail.com
