# Blood Cell Classification using Transfer Learning
**Student:** Ved Amar Jadiye | **Roll Number:** 20221283
**Course:** Image and Video Processing with Deep Learning

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Installation & Dependencies](#installation--dependencies)
5. [How to Run](#how-to-run)
6. [File-by-File Description](#file-by-file-description)
7. [Model Architecture](#model-architecture)
8. [Training Strategy](#training-strategy)
9. [Results](#results)
10. [Success Criteria](#success-criteria)
11. [Design Decisions & Justifications](#design-decisions--justifications)

---

## Project Overview

This project trains a deep convolutional neural network to classify white blood cells into 4 categories from microscopy images:

| Class | Cell Type | Lineage |
|---|---|---|
| LYMPHOCYTE | Agranulocyte | Lymphoid |
| NEUTROPHIL | Granulocyte | Myeloid |
| MONOCYTE | Agranulocyte | Monocytic |
| EOSINOPHIL | Granulocyte | Granulocytic |

Automated blood cell classification is a clinically relevant task — manual differential counts by haematologists are time-consuming and subject to inter-observer variability. A reliable CNN classifier can assist in screening for infections, leukaemia, and immune disorders.

The model is a fine-tuned **ResNet-18** pretrained on ImageNet, with a custom classification head and **Global Contrast Normalisation (GCN)** applied as a preprocessing step to correct for inconsistent Wright-Giemsa staining across slides.

---

## Dataset

**Source:** [Blood Cell Images — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)
(`archive (2).zip` → `dataset2-master/dataset2-master/images/TRAIN/`)

**Classes used:** LYMPHOCYTE, NEUTROPHIL, MONOCYTE, EOSINOPHIL

**Subset constructed:**
- 2000 images per class sampled randomly (seed = 42)
- **Total: 8000 images**
- Split: 70% train (1400) | 15% val (300) | 15% test (300) per class

**Raw image properties:**
- Resolution: 320 × 240 pixels
- Format: JPEG, RGB
- Resized to 224 × 224 for model input

The `data/` directory in this submission contains **10 raw unresized sample images per class** (40 total) directly from the Kaggle dataset, as required.

---

## Repository Structure

```
project_ved_amar_jadiye/
│
├── checkpoints/
│   └── final_weights.pth        ← best model weights (saved during training)
│
├── data/
│   ├── LYMPHOCYTE/              ← 10 sample .jpg images
│   ├── NEUTROPHIL/              ← 10 sample .jpg images
│   ├── MONOCYTE/                ← 10 sample .jpg images
│   └── EOSINOPHIL/              ← 10 sample .jpg images
│
├── config.py                    ← all hyperparameters and paths
├── dataset.py                   ← GCN transform + CellDataset + DataLoader
├── model.py                     ← ResNet-18 architecture + build_model()
├── train.py                     ← training loop with checkpointing
├── predict.py                   ← inference function + GradCAM helper
├── interface.py                 ← standardised aliases for the grader
└── README.md                    ← this file
```

---

## Installation & Dependencies

This project was developed and trained on **Google Colab (T4 GPU)**.

```bash
pip install torch torchvision
pip install scikit-learn matplotlib
pip install grad-cam          # for pytorch_grad_cam (GradCAM in predict.py)
```

All other dependencies (`numpy`, `PIL`, `pathlib`) are part of the standard Python environment.

**Python version:** 3.10+
**PyTorch version:** 2.x

---

## How to Run

### 1. Training

```python
from model import build_model
from train import train_model
import config

model   = build_model(num_classes=config.NUM_CLASSES)
history = train_model(
    model,
    num_epochs   = config.epochs,       # 20
    save_path    = config.BEST_WEIGHTS  # checkpoints/final_weights.pth
)
```

Or run directly:
```bash
python train.py
```

`train_model()` expects the full dataset at:
```
<DATA_DIR>/train/LYMPHOCYTE/  NEUTROPHIL/  MONOCYTE/  EOSINOPHIL/
<DATA_DIR>/val/  LYMPHOCYTE/  NEUTROPHIL/  MONOCYTE/  EOSINOPHIL/
```
Change `DATA_DIR` in `config.py` to point to your local dataset path.

---

### 2. Inference on new images

```python
from predict import classify_cells

image_paths = [
    'data/LYMPHOCYTE/img001.jpg',
    'data/NEUTROPHIL/img002.jpg',
    'data/MONOCYTE/img003.jpg',
]

predictions = classify_cells(image_paths)
# Returns: ['LYMPHOCYTE', 'NEUTROPHIL', 'MONOCYTE']

# With probabilities
predictions, probs = classify_cells(image_paths, return_probs=True)
# probs: np.ndarray of shape (N, 4)
```

---

### 3. Via interface.py (grader entry point)

```python
from interface import TheModel, the_trainer, the_predictor, TheDataset, the_dataloader
from interface import the_batch_size, total_epochs, CLASS_NAMES, NUM_CLASSES

# Instantiate model
model = TheModel()

# Train
history = the_trainer(model, num_epochs=total_epochs)

# Predict
labels = the_predictor(['data/LYMPHOCYTE/img001.jpg'])
```

---

### 4. GradCAM visualisation

```python
from predict import get_gradcam
import matplotlib.pyplot as plt

raw, overlay, label = get_gradcam('data/EOSINOPHIL/img001.jpg')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(raw);     axes[0].set_title('Original')
axes[1].imshow(overlay); axes[1].set_title(f'GradCAM — predicted: {label}')
plt.show()
```

---

## File-by-File Description

### `config.py`
Single source of truth for all constants. Every other module imports from here — no magic numbers anywhere else.

| Variable | Value | Meaning |
|---|---|---|
| `batch_size` | 64 | mini-batch size |
| `epochs` | 20 | training epochs |
| `LR` | 1e-3 | AdamW initial learning rate |
| `weight_decay` | 1e-4 | L2 regularisation |
| `resize_x/y` | 224 | model input size |
| `CLASS_NAMES` | 4 classes | in order matching label integers |
| `GCN_EPS` | 1e-8 | numerical stability in GCN |
| `BEST_WEIGHTS` | `checkpoints/final_weights.pth` | checkpoint path |

---

### `dataset.py`

**`GlobalContrastNormalization`** (torch.nn.Module)
Per-image, per-channel normalisation:
```
x_gcn[c] = (x[c] - mean(x[c])) / (std(x[c]) + eps)
```
Applied to every image at load time. Corrects for Wright-Giemsa staining intensity variation across microscopy slides — a known problem in haematology imaging. Without GCN, the model's early convolutional layers spend capacity learning staining intensity rather than morphological features.

**`train_transform`**
```
Resize(224) → HFlip(0.5) → VFlip(0.3) → Rotation(360°) → ColorJitter → ToTensor → GCN
```
360° rotation is biologically justified — blood cells have no preferred orientation on a slide.

**`eval_transform`**
```
Resize(224) → ToTensor → GCN
```
No augmentation at inference time.

**`CellDataset`**
Reads images from class sub-folders. Returns `(tensor, label_int)` pairs. Handles missing folders gracefully with a warning rather than crashing.

**`the_dataloader(root, train, batch_size)`**
Factory function that wraps `CellDataset` in a `DataLoader` with appropriate shuffle/pin_memory settings.

---

### `model.py`

**`build_model(num_classes=4)`**
Returns a ResNet-18 with:
- `layer1` frozen (low-level edge detectors — not cell-specific)
- `layer2`, `layer3`, `layer4` trainable (texture, shape, morphology)
- Custom FC head: `Dropout(0.4) → Linear(512→128) → ReLU → Dropout(0.2) → Linear(128→4)`

**`CellClassifierCNN`**
Thin `nn.Module` wrapper around `build_model()` so `interface.py` can expose a class (`TheModel`) while keeping the architecture identical to the notebook.

---

### `train.py`

**`run_epoch(model, loader, criterion, optimizer=None, device=None)`**
Single pass over a DataLoader. Pass `optimizer=None` for eval mode (no gradients). Includes gradient clipping (`max_norm=1.0`) to prevent exploding gradients during early fine-tuning.

**`train_model(model, num_epochs, train_loader, loss_fn, optimizer, ...)`**
Full training loop:
- Loss: `CrossEntropyLoss` (folds log-softmax, numerically stable)
- Optimiser: `AdamW` (decoupled weight decay — better than Adam for fine-tuning)
- Scheduler: `ReduceLROnPlateau(factor=0.5, patience=3)` — halves LR when val loss stalls
- Saves checkpoint whenever val loss improves
- Returns `history` dict with loss and accuracy per epoch for plotting

---

### `predict.py`

**`classify_cells(list_of_image_paths, weights_path, return_probs)`**
- Loads weights once and caches the model (avoids reloading on repeated calls)
- Processes images in mini-batches of `config.batch_size`
- Returns list of class-name strings, one per input image
- Optionally returns softmax probability array of shape `(N, 4)`

**`get_gradcam(image_path, weights_path)`**
- Hooks GradCAM onto `model.layer4[-1]` (last residual block of ResNet-18)
- Returns original image, heatmap overlay, and predicted label
- Uses `pytorch_grad_cam` library

---

### `interface.py`

| Alias (grader uses) | Points to |
|---|---|
| `TheModel` | `model.CellClassifierCNN` |
| `the_trainer` | `train.train_model` |
| `the_predictor` | `predict.classify_cells` |
| `TheDataset` | `dataset.CellDataset` |
| `the_dataloader` | `dataset.the_dataloader` |
| `the_batch_size` | `config.batch_size` (64) |
| `total_epochs` | `config.epochs` (20) |

---

## Model Architecture

```
Input (3 × 224 × 224)
    ↓
ResNet-18 Backbone (ImageNet pretrained)
    ├── conv1 + bn1 + relu + maxpool   [FROZEN via layer1]
    ├── layer1: 2× BasicBlock          [FROZEN]
    ├── layer2: 2× BasicBlock          [trainable]
    ├── layer3: 2× BasicBlock          [trainable]
    └── layer4: 2× BasicBlock          [trainable]  ← GradCAM target
    ↓
AdaptiveAvgPool → Flatten (512-dim)
    ↓
Dropout(0.4) → Linear(512→128) → ReLU → Dropout(0.2) → Linear(128→4)
    ↓
Output logits (4 classes)
```

**Trainable parameters:** ~10.9M / 11.2M total (97% unfrozen)

---

## Training Strategy

| Choice | Decision | Reason |
|---|---|---|
| Backbone | ResNet-18 pretrained | Too few images to train from scratch without catastrophic overfitting |
| Frozen layers | layer1 only | 5600 training images is sufficient to fine-tune from layer2 onward |
| Optimiser | AdamW | Decoupled weight decay gives better generalisation than vanilla Adam on small datasets |
| LR schedule | ReduceLROnPlateau | Biology loss curves stall unpredictably; adaptive decay handles this better than cosine |
| Augmentation | HFlip + VFlip + Rotation(360°) + ColorJitter | Cells are radially symmetric; staining variation handled by GCN not ColorJitter |
| Regularisation | Dropout(0.4) + Dropout(0.2) in head | Prevents head overfitting given large backbone capacity relative to dataset size |
| Gradient clipping | max_norm=1.0 | Stabilises early fine-tuning epochs when layer2 weights shift rapidly |

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | ≥ 75% (Criterion 1 target) |
| Best Val Loss | saved in `final_weights.pth` |

**Expected confusion patterns (biophysical basis):**
- **LYMPHOCYTE ↔ MONOCYTE:** Both agranulocytes. Lymphocytes have high nuclear-to-cytoplasmic (N:C) ratio and round nucleus. Monocytes have lower N:C ratio and kidney-shaped nucleus — the 3D fold is partially lost in 2D projection, causing occasional misclassification.
- **NEUTROPHIL ↔ EOSINOPHIL:** Both granulocytes. Key discriminator is granule colour (orange-red in eosinophils vs faint pink in neutrophils). Under staining variation this colour difference becomes subtle.

**GradCAM interpretation:**
- Heatmap concentrated on **nucleus** → model learned chromatin texture / nuclear shape ✓
- Heatmap spread into **cytoplasm** → model uses N:C ratio or granularity ✓
- Heatmap on **background** → model using artefacts ✗ (not observed)

---

## Success Criteria

| Criterion | Implementation | File |
|---|---|---|
| 1 — Test accuracy ≥ 75% | ResNet-18 fine-tuning + GCN + augmentation | `train.py`, `model.py` |
| 2 — Confusion matrix | Absolute counts + row-normalised (recall per class) | Cell 12 in notebook |
| 3 — Interpretability | One-vs-Rest PR curves (AP per class) + GradCAM saliency maps | Cell 13, 14 in notebook / `predict.py` |
