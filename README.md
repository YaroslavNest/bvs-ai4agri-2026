# Beyond Visible Spectrum: AI for Agriculture 2026  
## Automated Multimodal Crop Disease Diagnosis

This repository contains experiments for the Kaggle competition  
**Beyond Visible Spectrum: AI for Agriculture 2026**.

The task is to classify wheat disease conditions (**Health / Rust / Other**) using multimodal UAV imagery: **RGB**, **Multispectral (MS)**, and **Hyperspectral (HS)**.

---

## 📌 Competition Overview

The competition focuses on detecting rust-affected areas by leveraging joint **spectral and spatial information** from multimodal remote sensing data.

Participants are encouraged to develop multimodal models capable of adapting to data with varying spectral characteristics and spatial resolutions.

---

## 📊 Dataset Description

### Data Acquisition

- **UAV:** DJI M600 Pro  
- **Sensor:** S185 snapshot hyperspectral sensor  
- **Flight altitude:** 60 meters (~4 cm/pixel)  
- **Spectral range:** 450–950 nm  
- **Spectral resolution:** 4 nm  
- **Acquisition dates:** May 3 and May 8, 2019  

---

### Modalities

Each sample includes three aligned modalities:

#### 🟢 RGB
- Format: `.png`  
- True-color images generated from hyperspectral bands  

#### 🔵 Multispectral (MS)
- Format: `.tif`  
- 5 vegetation-related bands:
  - Blue (~480 nm)
  - Green (~550 nm)
  - Red (~650 nm)
  - Red Edge (740 nm)
  - NIR (833 nm)

#### 🟣 Hyperspectral (HS)
- Format: `.tif`  
- 125 spectral bands (450–950 nm)  
- The first ~10 and last ~14 bands may contain sensor noise  

---

## 📁 Repository Structure
- notebooks/
  - bvs-ai-agri-2026-dataoverview-and-eda.ipynb
  - bvs-ai4agri-ms-rgb-hs.ipynb


### 1️⃣ Data Overview & EDA

Exploratory analysis includes:
- Class distribution
- Spectral inspection
- Data format analysis
- Noise investigation

### 2️⃣ Training & Inference

The main training notebook contains:
- Data preprocessing
- Cross-validation
- Separate training for MS, RGB, and HS
- OOF prediction generation
- Weighted blending
- Final submission pipeline

---

## 🧠 Modeling Approach

Each modality is trained independently.

### MS Model
- **Backbone:** ConvNeXt-Tiny (pretrained)

### HS Model
- **Backbone:** ConvNeXt-Tiny (pretrained)

### RGB Model
- **Backbone:** ResNet18 (pretrained)

Training is performed using **5-fold Stratified Cross-Validation**.

---

## 🔄 Preprocessing & Normalization

### Removal of Empty Samples

Training samples where all pixels were constant were removed from the training set.

---

### MS Normalization

Per-sample MinMax normalization:

  ```(x - x_min) / (x_max - x_min)```

### HS Normalization

1. Spectral trimming (125 → 101 channels)
2. Scaling by 65535
3. Per-channel mean/std computation on training folds
4. Standardization:
  ```(x - mean_channel) / std_channel```

## 🏋️ Training Setup

- 5-fold Stratified Cross-Validation  
- Early stopping  
- Learning rate scheduling  
- Geometric augmentations:
  - Horizontal flips  
  - Vertical flips  
  - 90° rotations  
  - Light multiplicative noise (train only)

---

## 🔀 Ensemble Strategy

Models are trained independently and combined using **weighted blending**.

Weights are selected via grid search on OOF logits:

```logits_mix = w_ms * logits_ms \```
           ```+ w_rgb * logits_rgb \```
           ```+ w_hs * logits_hs```

Final predictions are obtained by applying softmax over blended logits.

---

📈 Results
Public Leaderboard Accuracy: 0.726

---

📌 Observations

- RGB provides the weakest standalone signal.
- MS delivers stable performance.
- HS provides stronger signal than MS but:
  - is more prone to overfitting
  - shows higher fold-to-fold variance

🧪 What Was Tested

Extensive experimentation included:

- EfficientNetV2
- DINO ViT
- Multiple backbone sizes
- Alternative normalization strategies
- Backbone freezing
- Stronger regularization
- Alternative loss functions
- Architectural modifications

Most alternatives either:
- did not improve local CV
- improved local CV but failed on the public leaderboard

🎯 Conclusion

Despite extensive experimentation, the most stable and performant solution remained:
Independent modality training + OOF-based weighted blending
Careful normalization and stable cross-validation design were the key contributors to final performance.

