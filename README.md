# SVM + ResNet50 Features

SVM implementation that classifies objects from COCO dataset images into 5 categories using ResNet50 as a feature extractor and a Linear Support Vector Machine (SVM) as the classifier.

## Overview

This notebook implements a classical ML approach to image classification:
1. Load and preprocess metadata from a filtered COCO dataset
2. Download and crop object images using bounding box annotations
3. Extract deep features using a pretrained ResNet50 backbone
4. Train and evaluate a LinearSVC classifier on those features

## Dataset

**Source:** [COCO Data (filtered) on Kaggle](https://www.kaggle.com/datasets/atharvadhupkar/coco-data2) — `filtered_coco_metadata.csv`

**Size:** Total 329,487 annotated objects across 20 columns including bounding boxes, image URLs, and category labels. 7000 total subsamples for this implementation.

**Classes (5):**
| Index | Label     |
|-------|-----------|
| 0     | Bicycle   |
| 1     | Bus       |
| 2     | Car       |
| 3     | Person    |
| 4     | Truck     |

**Splits:**
| Set  | Samples             |
|------|---------------------|
| Train| 5,000 (1,000/class) |
| Val  | 1,000 (200/class)   |
| Test | 1,000 (200/class)   |

Splits are done at the **image level** to prevent data leakage (no image appears in more than one split).

## Pipeline

### 1. Data Preparation
- One-hot labels are converted to integer class indices
- Bounding boxes are parsed from string format into `x, y, w, h` columns
- Images are downloaded from COCO URLs and saved locally
- Balanced sampling ensures equal class representation per split

### 2. Feature Extraction
- Images are cropped to their bounding boxes, then resized to **224×224**
- Standard ImageNet normalization is applied
- A pretrained **ResNet50** (ImageNet weights) with the final classification head removed is used to extract **2048-dim** average pooled features
- Features are cached to disk (`resnet50_avgpool_features.pkl`) to avoid redundant computation

### 3. Hyperparameter Tuning
- A `GridSearchCV` over `C = [0.01, 0.1, 1, 10]` was run (code included but commented out)
- Best parameter found: **C = 1**

### 4. Model Training & Evaluation
- A `LinearSVC(C=1, class_weight='balanced', max_iter=2000)` is trained on the extracted features
- Evaluated on both the validation and test sets
- Confusion matrices are plotted for both splits

## Results

### Validation Set
| Class     | Precision | Recall    | F1-Score  |
|-----------|-----------|-----------|-----------|
| Bicycle   | 0.87      | 0.84      | 0.86      |
| Bus       | 0.79      | 0.82      | 0.81      |
| Car       | 0.54      | 0.60      | 0.57      |
| Person    | 0.81      | 0.83      | 0.82      |
| Truck     | 0.62      | 0.52      | 0.57      |
| **Accuracy** | | | **0.72** |

### Test Set
| Class     | Precision | Recall    | F1-Score  |
|-----------|-----------|-----------|-----------|
| Bicycle   | 0.87      | 0.91      | 0.89      |
| Bus       | 0.81      | 0.77      | 0.79      |
| Car       | 0.64      | 0.62      | 0.63      |
| Person    | 0.84      | 0.82      | 0.83      |
| Truck     | 0.61      | 0.64      | 0.62      |
| **Accuracy** | | | **0.75** |

Car and truck are the hardest classes to distinguish, likely due to visual similarity.

## Requirements

```
pandas
numpy
Pillow
requests
torch
torchvision
scikit-learn
matplotlib
joblib
```

## Usage

1. Download the [filtered COCO metadata](https://www.kaggle.com/datasets/atharvadhupkar/coco-data2) and place it at the expected path.
2. Run the notebook cells in order.
3. On first run, images will be downloaded and features extracted (~14 min on a T4 GPU). Subsequent runs load from cache.

## Notes

- The notebook was run on **Kaggle** with a **NVIDIA Tesla T4** GPU
- Feature extraction takes ~14 minutes; full notebook runtime is ~23 minutes
