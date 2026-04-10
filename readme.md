# 🧠 Colon Cancer Detection using CLAM (Attention-based MIL)

## 📌 Overview

This project implements a **weakly supervised deep learning pipeline** for detecting colon cancer from histopathology Whole Slide Images (WSIs).

Instead of using raw images directly, the system operates on **pre-extracted patch-level features** and applies:

* CLAM
* Multiple Instance Learning

The model aggregates patch-level information to produce **slide-level predictions** and **interpretable attention heatmaps**.

---

## 🎯 Objective

* Classify WSIs into:

  * **Tumor (cancerous)**
  * **Normal (non-cancerous)**
* Use **slide-level labels only (weak supervision)**
* Provide **spatial interpretability using attention**

---

## 🧬 Dataset

The dataset consists of histopathology WSIs (from TCGA-like sources), processed into:

### 🔹 Raw Data (Not used directly)

* `.svs` files (gigapixel slides)

### 🔹 Processed Data (Used in project)

* `.pt` files (PyTorch tensors)

Each `.pt` file represents one slide and contains:

```text
{
  'features': [N, 2048],
  'coords':   [N, 2]
}
```

Where:

* `N` = number of patches (~10k–30k per slide)
* `2048` = feature dimension (CNN embedding)
* `coords` = spatial position (x, y) of each patch

---

## 🏗️ System Pipeline

### 🔹 Preprocessing (Completed Before This Stage)

```text
WSI (.svs)
   ↓
Grid Construction (Patch Extraction)
   ↓
CNN Feature Extraction (e.g., ResNet)
   ↓
Saved as .pt files
```

⚠️ This stage is already completed and not part of current implementation.

---

### 🔹 Active Pipeline

```text
Feature Bags (.pt)
        ↓
CLAM (Attention-based MIL)
        ↓
Patch-level Attention Scores
        ↓
Weighted Aggregation
        ↓
Slide-level Classification
        ↓
Grid Reconstruction (8×8 / 16×16)
        ↓
Attention Heatmap Visualization
```

---

## 🧠 Model Architecture

### Input

* Feature tensor: `[N, 2048]`

### Components

1. **Attention Module**

   * Assigns importance score to each patch:

   ```text
   α_i = attention weight of patch i
   ```

2. **Aggregation**

   ```text
   Slide representation = Σ (α_i × feature_i)
   ```

3. **Classifier**

   * Fully connected layer
   * Outputs:

     * Tumor / Normal probability

---

## 🔁 Training Strategy

* **5-Fold Cross-Validation**

```text
Dataset → Split into 5 folds
Train on 4 → Test on 1 (repeat 5 times)
```

### Model Outputs:

* `fold_X.pt` → trained weights
* `best_fold_X.pt` → best checkpoint
* `metrics_report.txt` → performance metrics
* `ensemble_predictions.csv` → combined results

---

## 📁 Project Structure

```text
MIL/
│
├── features/                # Input feature files (.pt)
├── models/
│   └── five_fold/          # Trained models
│
├── clam_model.py           # CLAM architecture
├── train_clam_5fold.py     # Training script
├── gen_labels.py           # Label generation
├── check_features.py       # Feature validation
├── inspect_data.py         # Debugging script
│
└── README.md
```

---

## 🔍 Grid Reconstruction (Your Task)

After model prediction, spatial structure is reconstructed using patch coordinates.

### Step-by-step:

1. Extract attention scores:

```text
A = [N]
```

2. Use coordinates:

```text
coords = [N, 2]
```

3. Divide slide into grid:

```text
8×8 or 16×16 regions
```

4. Aggregate attention:

```text
grid[i][j] = mean attention of patches in region
```

5. Generate heatmap

---

## 🎨 Output

For each slide:

* **Prediction**: Tumor / Normal
* **Confidence score**
* **Attention heatmap** showing important regions

---

## 📊 Evaluation Metrics

* Accuracy
* Precision / Recall
* F1 Score
* ROC-AUC (primary metric for medical tasks)

---

## ❗ Important Notes

### ✔ What is Used

* Pre-extracted features (`.pt`)
* Attention-based MIL (CLAM)

### ❌ What is NOT Used

* Raw `.svs` files (already processed)
* Patch extraction pipeline (precomputed)

---

## 🧠 Key Insight

> The system operates on feature-level representations instead of raw images, making it computationally efficient while retaining spatial interpretability through attention and coordinate mapping.

---

## 🧾 One-Line Summary

> An attention-based MIL system (CLAM) that aggregates patch-level features into slide-level cancer predictions and reconstructs spatial heatmaps using patch coordinates.

---

## 👨‍💻 Author

* Buddham Rajbhandari
* Kaviya Darshini
* Dakshini

---
