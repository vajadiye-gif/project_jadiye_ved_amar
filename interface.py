# ==============================================================================
# interface.py
# Standardised entry-point for the grading programme.
# All names on the RIGHT side are fixed — the grader calls these.
# Names on the LEFT side match this project's actual implementations.
# ==============================================================================

# The model class
from model import CellClassifierCNN as TheModel

# The function inside train.py that runs the training loop
from train import train_model as the_trainer

# The function inside predict.py that generates inference on a batch of images
from predict import classify_cells as the_predictor

# The custom Dataset class
from dataset import CellDataset as TheDataset

# The DataLoader factory
from dataset import the_dataloader as the_dataloader

# Hyperparameters from config
from config import batch_size as the_batch_size
from config import epochs     as total_epochs

# ---------------------------------------------------------------------------
# Additional exports the grader may access for metrics / evaluation
# ---------------------------------------------------------------------------
from config import CLASS_NAMES, NUM_CLASSES
