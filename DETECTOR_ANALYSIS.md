# DeepFakeGuard Detector Analysis & Expected Performance

## 🎯 Detector Overview

### 1. 🧠 DINOv3 (Vision Transformer)
**Architecture:** DINOv3 ViT-B/16 with custom classification head  
**Training:** Fine-tuned specifically on deepfake detection dataset  
**Input:** Face-cropped frames (requires MTCNN face detection)  
**Embeddings:** 768-dimensional  

**Expected Performance:**
- ✅ **Accuracy:** 0.88+ AUROC (your trained model)
- ✅ **Best at:** Detecting facial deepfakes, GAN artifacts
- ✅ **Strengths:** 
  - Trained specifically for deepfakes
  - Excellent on FaceSwap, DeepFaceLab outputs
  - Good generalization across datasets
- ⚠️ **Limitations:**
  - Requires face detection (fails on no-face videos)
  - Slower (face cropping overhead)
  - Needs trained weights file

**Best Use Cases:**
- High-stakes detection where accuracy matters most
- Videos with clear faces
- When you have the weights file available

---

### 2. 🎯 ResNet18 (CNN)
**Architecture:** ResNet18 with custom classifier head  
**Training:** Pretrained on ImageNet only (NOT fine-tuned on deepfakes)  
**Input:** Full frames (no face cropping)  
**Embeddings:** 512-dimensional  

**Expected Performance:**
- 📊 **Accuracy:** Baseline (~60-70% on deepfakes - estimated)
- 📊 **Best at:** Quick screening, fallback when faces not detected
- ✅ **Strengths:**
  - Very fast (no face detection overhead)
  - Works on any video (no face requirement)
  - No weights needed (uses pretrained ImageNet)
  - Lightweight model
- ⚠️ **Limitations:**
  - NOT fine-tuned on deepfakes (lower accuracy)
  - May miss subtle deepfake artifacts
  - General computer vision features, not specialized

**Best Use Cases:**
- Quick preliminary screening
- Videos without faces (landscapes, objects)
- When speed is more important than accuracy
- Fallback when DINOv3 fails

---

### 3. 🌿 IvyFake (CLIP-based)
**Architecture:** CLIP ViT-B/32 + Temporal & Spatial Artifact Analyzers  
**Training:** Pretrained CLIP (vision-language) + custom artifact detectors  
**Input:** Full frames (no face cropping required)  
**Embeddings:** 768-dimensional  
**Special Features:**
- Temporal Artifact Analysis (inconsistencies across frames)
- Spatial Artifact Analysis (artifacts in individual frames)
- Explainable outputs

**Expected Performance:**
- 📊 **Accuracy:** Unknown on deepfakes (designed for AIGC detection)
- 📊 **Best at:** Detecting AI-generated content artifacts
- ✅ **Strengths:**
  - Explainable (tells you what artifacts it found)
  - Temporal analysis (good for videos)
  - No face requirement
  - Novel architecture (different approach than others)
- ⚠️ **Limitations:**
  - First run downloads CLIP (~500MB)
  - Slower inference (CLIP is large)
  - May need tuning for deepfakes specifically
  - Designed for AIGC, not specifically deepfakes

**Best Use Cases:**
- When you need explanations
- Analyzing temporal inconsistencies
- Videos with complex artifacts
- Research/experimentation

---

## 📊 Expected Comparison Matrix

| Metric | DINOv3 | ResNet18 | IvyFake |
|--------|--------|----------|---------|
| **Accuracy** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ (unknown) |
| **Speed** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Face Required** | Yes | No | No |
| **Weights Needed** | Yes | No | No |
| **Explainable** | No | No | Yes |
| **Memory** | Medium | Low | High (CLIP) |
| **Setup** | Hard | Easy | Medium |

---

## 🧪 Testing Recommendations

### Test Set Composition:
1. **Real videos** (50%)
   - Various resolutions
   - With and without faces
   - Different content types

2. **Deepfake videos** (50%)
   - FaceSwap outputs
   - DeepFaceLab outputs
   - Different GAN architectures
   - Various compression levels

### Metrics to Track:
- **Accuracy:** Correct predictions / Total
- **False Positive Rate:** Real classified as Fake
- **False Negative Rate:** Fake classified as Real
- **Inference Time:** Seconds per video
- **Failure Rate:** Errors / Total

### Expected Behavior Patterns:

**DINOv3:**
- High accuracy on face-heavy videos
- May fail/error on videos without detectable faces
- Consistent predictions across similar videos
- Best confidence scores

**ResNet18:**
- Always runs (no face dependency)
- May show random-like performance (not trained on deepfakes)
- Fastest inference
- Good for baseline comparison

**IvyFake:**
- Novel predictions (different from other two)
- Provides feature explanations
- Slower on first run (model download)
- May excel on certain artifact types

---

## 🔬 Running the Benchmarks

### Quick Test:
```bash
python test_integration_quick.py
```

### Full Comparison:
```bash
# Single video
python benchmark_detectors.py -v video.mp4 -g FAKE

# Directory of videos
python benchmark_detectors.py -d /path/to/test/videos/

# With ground truth labels
python benchmark_detectors.py -d ./test_videos/
```

### Generate Report:
```bash
python benchmark_detectors.py -v test.mp4 -o my_report.json
cat my_report.json | python -m json.tool
```

---

## 💡 Analysis Tips

1. **Look for agreement:** When all 3 detectors agree, confidence is higher
2. **Check edge cases:** Test videos with no faces, heavy compression, etc.
3. **Analyze failures:** Which detector fails and why?
4. **Speed vs Accuracy tradeoff:** Is DINOv3's accuracy worth the speed cost?
5. **IvyFake explanations:** Do the artifact descriptions match reality?

---

## 📝 Recording Results

Use this template to document your findings:

```markdown
## Test Results - [Date]

### Test Set:
- Total videos: X
- Real videos: X
- Fake videos: X

### DINOv3:
- Accuracy: XX%
- Avg Time: X.Xs
- Failures: X
- Notes: 

### ResNet18:
- Accuracy: XX%
- Avg Time: X.Xs
- Failures: X
- Notes:

### IvyFake:
- Accuracy: XX%
- Avg Time: X.Xs
- Failures: X
- Notes:

### Key Findings:
- Best overall: 
- Most reliable:
- Biggest surprise:
- Recommended use cases:
```

---

## 🚀 Next Steps

After testing, consider:
1. **Ensemble methods:** Combine all 3 detectors for better accuracy
2. **Threshold tuning:** Adjust 0.5 threshold based on results
3. **Preprocessing:** Try different compression levels
4. **Fine-tuning:** Train IvyFake on deepfake dataset
5. **Feature analysis:** Use IvyFake explanations to improve DINOv3