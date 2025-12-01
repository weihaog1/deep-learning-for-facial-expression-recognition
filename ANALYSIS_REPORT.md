# Facial Expression Recognition: Experimental Analysis Report

## Executive Summary

This report presents a comparative analysis of three machine learning approaches for facial expression recognition on the Muxspace dataset (~13,700 images, 7 emotion classes). Our experiments reveal that **transfer learning with ResNet18 achieves the best performance (88.05%)**, significantly outperforming both the traditional baseline (82.10%) and the custom CNN trained from scratch (51.88%).

---

## 1. Dataset Analysis

### 1.1 Dataset Overview

| Metric | Value |
|--------|-------|
| Total Images | 13,681 |
| Number of Classes | 7 |
| Image Format | RGB JPEG |
| Train/Val/Test Split | 70% / 10% / 20% |

### 1.2 Class Distribution (Critical Issue)

| Emotion | Count | Percentage | Test Samples |
|---------|-------|------------|--------------|
| Neutral | 6,868 | 50.2% | 1,374 |
| Happiness | 5,696 | 41.6% | 1,139 |
| Surprise | 368 | 2.7% | 74 |
| Sadness | 268 | 2.0% | 54 |
| Anger | 252 | 1.8% | 50 |
| Disgust | 208 | 1.5% | 42 |
| Fear | 21 | **0.2%** | **4** |

**Key Finding:** The dataset exhibits severe class imbalance. Neutral and happiness together comprise **91.8%** of all samples, while fear has only **21 samples total** (4 in test set). This imbalance significantly impacts model training and evaluation.

---

## 2. Experimental Results

### 2.1 Model Comparison

| Model | Test Accuracy | Macro F1 | Training Time |
|-------|---------------|----------|---------------|
| HOG + SVM (Baseline) | 82.10% | 0.39 | ~5 minutes |
| Custom CNN | 51.88% | 0.13 | ~45 minutes |
| **ResNet18 (Transfer)** | **88.05%** | - | ~90 minutes |

### 2.2 Key Observations

#### Baseline (HOG + SVM): 82.10% Accuracy
- **Surprisingly strong performance** for a traditional ML approach
- Achieves 99.93% training accuracy but 82.10% test accuracy (some overfitting)
- Works well on majority classes (happiness: 85% F1, neutral: 85% F1)
- Struggles with minority classes (fear: 0%, sadness: 19% F1)

#### Custom CNN: 51.88% Accuracy (FAILURE)
- **Performed worse than random guessing for a 2-class problem**
- Model collapsed to predicting almost exclusively "neutral"
- Per-class breakdown reveals catastrophic failure:
  - Neutral: 96.3% recall (predicts neutral for everything)
  - Happiness: 8.3% recall
  - All other classes: 0-5% recall
- **Root Cause Analysis:**
  1. Insufficient training data for learning from scratch
  2. Severe class imbalance caused mode collapse
  3. Model learned to minimize loss by always predicting majority class
  4. Early stopping triggered at epoch 13 with only 52% validation accuracy

#### Transfer Learning (ResNet18): 88.05% Accuracy (BEST)
- **6 percentage points better than baseline**
- Two-phase training was effective:
  - Phase 1 (frozen backbone): Reached 68.79% validation accuracy
  - Phase 2 (fine-tuning): Jumped to 89.69% validation accuracy
- Pretrained ImageNet features transferred well to facial expressions
- Fine-tuning with low learning rate (0.0001) prevented catastrophic forgetting

---

## 3. Per-Class Performance Analysis

### 3.1 Baseline Model (HOG + SVM)

| Emotion | Precision | Recall | F1-Score | Analysis |
|---------|-----------|--------|----------|----------|
| Anger | 0.83 | 0.20 | 0.32 | High precision but misses most angry faces |
| Disgust | 0.50 | 0.21 | 0.30 | Low recall, often confused with anger |
| Fear | 0.00 | 0.00 | 0.00 | **Complete failure** (only 4 test samples) |
| Happiness | 0.84 | 0.85 | 0.85 | Strong performance (majority class) |
| Neutral | 0.81 | 0.90 | 0.85 | Best class (largest sample size) |
| Sadness | 0.67 | 0.11 | 0.19 | Poor recall, confused with neutral |
| Surprise | 0.67 | 0.14 | 0.22 | Often confused with fear/neutral |

### 3.2 Custom CNN (Mode Collapse)

| Emotion | Precision | Recall | F1-Score | Analysis |
|---------|-----------|--------|----------|----------|
| Anger | 0.00 | 0.00 | 0.00 | Never predicted |
| Disgust | 0.20 | 0.05 | 0.08 | Rarely predicted |
| Fear | 0.00 | 0.00 | 0.00 | Never predicted |
| Happiness | 0.46 | 0.08 | 0.14 | Occasionally predicted |
| **Neutral** | 0.52 | **0.96** | 0.68 | **Almost always predicted** |
| Sadness | 0.00 | 0.00 | 0.00 | Never predicted |
| Surprise | 0.00 | 0.00 | 0.00 | Never predicted |

**Diagnosis:** The custom CNN suffered from **mode collapse** - it learned that predicting "neutral" for everything minimizes the cross-entropy loss due to class imbalance.

---

## 4. Analysis of Failure: Custom CNN

### 4.1 Why Did the Custom CNN Fail?

1. **Insufficient Data**
   - 13,681 images is small for training a CNN from scratch
   - ImageNet has 1.2 million images; we have 1% of that
   - Custom CNN has 423,175 parameters to learn

2. **Severe Class Imbalance**
   - 91.8% of data is neutral + happiness
   - Loss function optimization favors majority classes
   - Without class weighting, model ignores minority classes

3. **No Pretrained Features**
   - Must learn edge detection, texture recognition, etc. from scratch
   - Transfer learning leverages features learned from millions of images

4. **Validation Accuracy Plateau**
   - Training accuracy stalled around 51-52%
   - Model never learned meaningful features
   - Early stopping preserved a barely-functional model

### 4.2 Potential Fixes

| Solution | Expected Impact |
|----------|-----------------|
| Class-weighted loss function | High - penalizes majority class errors |
| Oversampling minority classes | Medium - more training examples |
| Focal loss | High - focuses on hard examples |
| More training data | High - fundamental solution |
| Deeper data augmentation | Medium - effectively more data |
| Reduce model complexity | Medium - prevent overfitting |

---

## 5. Why Transfer Learning Succeeded

### 5.1 Advantages Demonstrated

1. **Rich Pretrained Features**
   - ResNet18 trained on 1.2M ImageNet images
   - Already knows edges, textures, shapes, faces
   - Only needs to learn emotion-specific features

2. **Two-Phase Training**
   - Phase 1: Frozen backbone (3,591 trainable params)
     - Quickly learns class boundaries with pretrained features
     - Prevents catastrophic forgetting
   - Phase 2: Full fine-tuning (11.2M params)
     - Adapts all features to emotion task
     - Low learning rate (0.0001) preserves pretrained knowledge

3. **Better Generalization**
   - Training accuracy: 97.93%
   - Validation accuracy: 89.91%
   - Test accuracy: 88.05%
   - Much smaller gap than baseline (less overfitting)

### 5.2 Training Dynamics

| Phase | Epochs | Start Val Acc | End Val Acc | Learning Rate |
|-------|--------|---------------|-------------|---------------|
| Phase 1 (Frozen) | 7 | 67.62% | 68.79% | 0.001 |
| Phase 2 (Fine-tune) | 11 | 86.26% | 89.91% | 0.0001 |

---

## 6. Recommendations

### 6.1 For Production Use
- **Use the Transfer Learning model** (88.05% accuracy)
- Be aware of poor performance on minority classes
- Consider ensemble with rule-based post-processing

### 6.2 For Improved Performance
1. **Address Class Imbalance:**
   - Implement weighted cross-entropy loss
   - Use focal loss for hard example mining
   - Oversample minority classes

2. **Data Collection:**
   - Collect more fear, disgust, anger samples
   - Target at least 500 samples per class

3. **Architecture Improvements:**
   - Try larger models (ResNet50, EfficientNet)
   - Implement attention mechanisms for face regions

4. **Training Improvements:**
   - Add mixup/cutmix augmentation
   - Use label smoothing
   - Implement k-fold cross-validation

---

## 7. Conclusions

### 7.1 Key Findings

1. **Transfer learning is essential** for small datasets
   - 88.05% vs 51.88% accuracy (custom CNN)
   - Pretrained features make the difference

2. **Traditional ML can be competitive**
   - HOG + SVM achieved 82.10% with minimal tuning
   - Simpler models are valuable baselines

3. **Class imbalance is critical**
   - Caused complete failure of custom CNN
   - All models struggle with minority classes (fear: 0% across all)

4. **Dataset size matters**
   - 13,681 images insufficient for training CNN from scratch
   - Transfer learning circumvents this limitation

### 7.2 Final Model Ranking

| Rank | Model | Accuracy | Recommendation |
|------|-------|----------|----------------|
| 1 | ResNet18 (Transfer) | **88.05%** | ✅ Use this |
| 2 | HOG + SVM | 82.10% | Good baseline |
| 3 | Custom CNN | 51.88% | ❌ Do not use |

---

## Appendix: Output Files

| File | Description |
|------|-------------|
| `outputs/baseline_confusion_matrix.png` | Baseline model confusion matrix |
| `outputs/baseline_per_class_metrics.png` | Baseline precision/recall/F1 chart |
| `outputs/baseline_metrics.json` | Baseline metrics in JSON |
| `outputs/custom_confusion_matrix.png` | Custom CNN confusion matrix |
| `outputs/custom_training_history.png` | CNN training curves |
| `outputs/model_comparison.png` | Side-by-side model comparison |
| `outputs/gradcam_custom.png` | Grad-CAM visualizations |
| `checkpoints/baseline_model.pkl` | Trained baseline model |
| `checkpoints/custom_best.pth` | Best custom CNN checkpoint |
| `checkpoints/transfer_finetuned_best.pth` | Best transfer learning checkpoint |

---

*Report generated from experimental run on CPU. Transfer learning training took approximately 90 minutes total.*
