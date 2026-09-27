# PneumoScan — Chest X-Ray Pneumonia Classifier

A deep learning pipeline that classifies pediatric chest X-rays into **NORMAL**, **BACTERIAL pneumonia**, or **VIRAL pneumonia**, using both a custom CNN and a fine-tuned MobileNetV2 transfer-learning model.

> ⚠️ **Disclaimer:** This project is for educational / portfolio purposes only. It is **not** a medical device and must not be used for real clinical diagnosis.

---

## 🧠 Overview

Chest X-rays are one of the most common tools for diagnosing pneumonia, but distinguishing *bacterial* from *viral* pneumonia (and both from normal lungs) by eye is hard even for experienced radiologists. This project builds an end-to-end image classification pipeline that:

1. Parses and labels a raw chest X-ray dataset into 3 classes
2. Explores class distribution, image sizes, and visual samples
3. Trains a baseline CNN from scratch
4. Trains a MobileNetV2 transfer-learning model (fine-tuned)
5. Evaluates both with accuracy, precision/recall/F1, and confusion matrices

## 📊 Dataset

- **Source:** [Labeled Chest X-Ray Images](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images) (Kaggle)
- **Classes:** `NORMAL`, `BACTERIA`, `VIRUS` (both pneumonia types are derived from a single `PNEUMONIA` folder via filename keywords)
- **Split:** 4,447 train / 785 validation (stratified 85/15 split) / 624 test images
- **Class balance (train):** BACTERIA ~48.5%, NORMAL ~25.8%, VIRUS ~25.7% → handled with computed class weights during training

## 🏗️ Approach

| Step | Description |
|------|-------------|
| Data labeling | Custom parser assigns NORMAL / BACTERIA / VIRUS from folder + filename |
| EDA | Class distribution plots, sample image grids, image dimension/format checks |
| Preprocessing | Resize to 224×224, rescale to [0,1], augmentation (rotation, flip, zoom, shifts) for training only |
| Class imbalance | `sklearn.compute_class_weight` used during training |
| Model 1 — Baseline CNN | 4 conv blocks (32→64→128→128 filters) + BatchNorm + MaxPooling, ~276K params |
| Model 2 — Transfer Learning | MobileNetV2 (ImageNet weights), first 100 layers frozen, fine-tuned on top |
| Callbacks | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint (both models) |

The **MobileNetV2 transfer-learning model** was selected as the final model.

## ✅ Results (Test Set, 624 images)

| Class    | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| NORMAL   | 1.0000    | 0.6197 | 0.7652   | 234     |
| BACTERIA | 0.7649    | 0.9545 | 0.8493   | 242     |
| VIRUS    | 0.6158    | 0.7365 | 0.6708   | 148     |

- **Overall accuracy:** 77.72%
- **Macro F1:** 0.7617
- **Weighted F1:** 0.7754

### Key insights
- **NORMAL** is predicted with perfect precision but the model misses ~38% of real normal cases (recall 0.62) — likely mistaking them for mild pneumonia.
- **BACTERIA** is the easiest class to detect overall (highest F1).
- **VIRUS** is the hardest class to separate from bacterial pneumonia, dragging down macro performance.
- Class imbalance (BACTERIA ≈ 2× the other classes) was mitigated with class weighting but viral/bacterial confusion remains the main bottleneck.

## 🗂️ Project Structure

```
.
├── computer-vision-X_ray.ipynb   # Full notebook: EDA → training → evaluation
├── README.md
└── (model checkpoints saved during training: best_basic_cnn.keras, best_transfer_model.keras)
```

## ⚙️ Tech Stack

- Python, TensorFlow / Keras
- MobileNetV2 (transfer learning)
- NumPy, Pandas, OpenCV
- Matplotlib, Seaborn
- scikit-learn (metrics, class weighting, train/val split)

## 🚀 Running the Notebook

1. Download the dataset from Kaggle: `tolgadincer/labeled-chest-xray-images`
2. Place it so the notebook's `find_dataset_root()` can locate a `train/` folder containing `NORMAL/` and `PNEUMONIA/` subfolders (defaults to a Kaggle-style `/kaggle/input/...` path — adjust if running locally)
3. Install dependencies:
   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn opencv-python scikit-learn
   ```
4. Run all cells in `computer-vision-X_ray.ipynb`

## 🔭 Possible Improvements

- Address bacterial/viral confusion with finer-grained augmentation or a dedicated two-stage classifier (normal vs. pneumonia, then bacterial vs. viral)
- Try other backbones (EfficientNet, ResNet) or ensembling the CNN + MobileNetV2
- Add Grad-CAM visualizations for model interpretability
- Cross-validate rather than a single train/val split

---

*Built as a learning project exploring CNNs and transfer learning for medical image classification.*
