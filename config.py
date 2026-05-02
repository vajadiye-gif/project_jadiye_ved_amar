# ==============================================================================
# config.py
# Central configuration — all hyperparameters and paths in one place.
# Every other module imports from here; change values only in this file.
# ==============================================================================

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR       = os.path.join(os.path.dirname(__file__), 'data')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
BEST_WEIGHTS   = os.path.join(CHECKPOINT_DIR, 'final_weights.pth')

# ---------------------------------------------------------------------------
# Image dimensions  (ResNet standard input)
# ---------------------------------------------------------------------------
resize_x       = 224    # width  (pixels)
resize_y       = 224    # height (pixels)
input_channels = 3      # RGB

# ---------------------------------------------------------------------------
# Training hyperparameters  (from Cell 4)
# ---------------------------------------------------------------------------
SEED        = 42
batch_size  = 64
epochs      = 20
LR          = 1e-3
weight_decay = 1e-4

# ---------------------------------------------------------------------------
# Classes  (from Cell 4 — Kaggle Blood Cell dataset)
# ---------------------------------------------------------------------------
CLASS_NAMES = ['LYMPHOCYTE', 'NEUTROPHIL', 'MONOCYTE', 'EOSINOPHIL']
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# GCN parameters  (from Cell 5)
# ---------------------------------------------------------------------------
GCN_EPS = 1e-8

# ---------------------------------------------------------------------------
# Dataset split ratios  (from Cell 3)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70   # 1400 / 2000
VAL_RATIO   = 0.15   # 300  / 2000
# test = remaining 0.15

N_PER_CLASS = 2000   # images sampled per class
