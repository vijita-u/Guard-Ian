# YOLOv8m + Weighted Dataloader

YOLOv8m implementation for object detection on a filtered COCO dataset using a custom weighted dataloader to reduce class imbalance.

## Overview

This notebook implements an object detection pipeline for pedestrian safety:

1. Load and filter COCO annotations for selected traffic-related classes
2. Convert COCO bounding boxes into YOLO format
3. Create a custom weighted dataset to oversample rare classes
4. Train a pretrained YOLOv8m model with data augmentation
5. Validate and test the model using standard object detection metrics

## Dataset

**Source:** COCO 2017 Dataset

**Classes (5):**

| Index | Label   |
| ----- | ------- |
| 0     | Person  |
| 1     | Bicycle |
| 2     | Car     |
| 3     | Bus     |
| 4     | Truck   |

**Splits:**

| Set   | Images            |
| ----- | ----------------- |
| Train | 69,655            |
| Val   | 2,948             |
| Test  | 100 sample images |

The dataset is filtered so that only images containing at least one of the five target classes are included.

## Pipeline

### 1. Data Preparation

- COCO annotations are loaded using `pycocotools`
- Only person, bicycle, car, bus, and truck annotations are retained
- Bounding boxes are converted from COCO format to YOLO format
- Labels are saved as `.txt` files and images are organized into YOLO directory structure
- A `data.yaml` file is generated for training

### 2. Class Imbalance Handling

- A custom `YOLOWeightedDataset` class extends the default YOLO dataset
- Class frequencies are counted across the training dataset
- Inverse-frequency weights are assigned so rare classes receive larger weights
- Each image is assigned a weight based on the rarest class it contains
- Images containing rare classes are sampled more frequently during training

### 3. Model Training

- A pretrained `YOLOv8m` model is loaded
- The weighted dataset replaces the default YOLO dataset
- Training is performed with the following settings:
  - Epochs: 20
  - Image Size: 512×512
  - Batch Size: 16
  - Mosaic: 0.7
  - MixUp: 0.05
  - Copy-Paste: 0.1
  - Cache: Enabled

### 4. Validation & Testing

- Validation is performed using `model.val()`
- Predictions are generated on 100 sample test images using `model.predict()`
- Performance is evaluated using:
  - Precision
  - Recall
  - mAP\@50
  - mAP\@50-95
  - Confusion Matrix

## Results

### Model Summary

| Metric     | Value      |
| ---------- | ---------- |
| Layers     | 170        |
| Parameters | 25,859,215 |
| Gradients  | 25,859,199 |
| GFLOPs     | 79.1       |

### Training Results

| Class       | Precision | Recall   | mAP\@50  | mAP\@50-95 |
| ----------- | --------- | -------- | -------- | ---------- |
| Person      | 0.86      | 0.66     | 0.79     | 0.55       |
| Bicycle     | 0.73      | 0.64     | 0.70     | 0.50       |
| Car         | 0.83      | 0.61     | 0.73     | 0.49       |
| Bus         | 0.86      | 0.90     | 0.93     | 0.82       |
| Truck       | 0.76      | 0.67     | 0.79     | 0.55       |
| **Overall** | **0.81**  | **0.69** | **0.78** | **0.59**   |

### Validation Results

| Class       | Precision | Recall   | mAP\@50  | mAP\@50-95 |
| ----------- | --------- | -------- | -------- | ---------- |
| Person      | 0.83      | 0.67     | 0.77     | 0.53       |
| Bicycle     | 0.63      | 0.51     | 0.56     | 0.34       |
| Car         | 0.75      | 0.59     | 0.67     | 0.45       |
| Bus         | 0.76      | 0.80     | 0.84     | 0.71       |
| Truck       | 0.58      | 0.51     | 0.57     | 0.42       |
| **Overall** | **0.71**  | **0.62** | **0.68** | **0.49**   |

Bicycle and truck remain the hardest classes due to lower representation and visual similarity with other vehicles.

## Requirements

```
ultralytics
numpy
matplotlib
opencv-python
pycocotools
```

## Usage

1. Download the COCO 2017 dataset and annotation files.
2. Run the notebook cells in order.
3. Generate YOLO labels and folder structure.
4. Train the model using the weighted dataloader.
5. Validate the trained model and visualize predictions.

## Notes

- The notebook was run on \*\*Kaggle\*\* with a \*\*NVIDIA Tesla T4\*\* GPU
- Full notebook runtime is \~11 hours for 20 epochs

