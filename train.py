# ==============================================================================
# train.py
# Loss, optimizer, scheduler, and training loop.
# Taken directly from Cells 8 & 9 of the project notebook.
# ==============================================================================

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from model   import build_model
from dataset import CellDataset, train_transform, eval_transform


# ==============================================================================
# CELL 8 — Loss, optimizer & LR scheduler
# ==============================================================================
"""
CrossEntropyLoss   : Standard for multi-class; numerically stable (folds log-softmax).
AdamW              : Adam + decoupled weight decay. Better regularisation than Adam
                     for fine-tuning pretrained networks on small datasets.
ReduceLROnPlateau  : Halves LR when val_loss plateaus for 3 epochs.
                     More adaptive than step/cosine decay for biology tasks
                     where loss curves can stall unpredictably.
"""


# ==============================================================================
# CELL 9 — Training loop
# ==============================================================================

def run_epoch(model, loader, criterion, optimizer=None, device=None):
    """
    One training or evaluation pass.
    Pass optimizer=None to run in eval mode (no gradient computation).
    Returns: (avg_loss, accuracy)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping: prevents exploding gradients in early fine-tuning
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


def train_model(
    model,
    num_epochs:   int        = config.epochs,
    train_loader: DataLoader = None,
    loss_fn                  = None,
    optimizer                = None,
    val_loader:   DataLoader = None,
    device                   = None,
    save_path:    str        = config.BEST_WEIGHTS,
) -> dict:
    """
    Run the full training loop with checkpointing.

    Parameters
    ----------
    model        : nn.Module  — model to train (build_model() recommended)
    num_epochs   : int        — total epochs
    train_loader : DataLoader — if None, built from config.DATA_DIR/train
    loss_fn      : callable   — if None, CrossEntropyLoss
    optimizer    : optimizer  — if None, AdamW is constructed
    val_loader   : DataLoader — if None, built from config.DATA_DIR/val
    device       : torch.device or str — auto-detected if None
    save_path    : str        — where to write final_weights.pth

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """

    # --- Device ---
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # --- DataLoaders (mirror Cell 6 construction) ---
    if train_loader is None:
        train_ds     = CellDataset(os.path.join(config.DATA_DIR, 'train'), transform=train_transform)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)
    if val_loader is None:
        val_ds     = CellDataset(os.path.join(config.DATA_DIR, 'val'), transform=eval_transform)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                                shuffle=False, num_workers=2, pin_memory=True)

    # --- Loss & Optimiser (from Cell 8) ---
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LR,
            weight_decay=config.weight_decay,
        )

    # ReduceLROnPlateau: halves LR when val_loss plateaus for 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # --- Checkpoint dir ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    history       = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')

    print(f'Training for {num_epochs} epochs on {device}')
    print('=' * 65)

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, optimizer, device)
        vl_loss, vl_acc = run_epoch(model, val_loader,   loss_fn, None,      device)

        scheduler.step(vl_loss)

        tag = ''
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), save_path)
            tag = '  ✓'

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        print(f'[{epoch:02d}/{num_epochs}]  '
              f'Train  loss={tr_loss:.4f}  acc={tr_acc:.3f}  |  '
              f'Val  loss={vl_loss:.4f}  acc={vl_acc:.3f}{tag}')

    print('=' * 65)
    print(f'Training complete. Best val loss: {best_val_loss:.4f}')
    print(f'Weights saved → {save_path}')

    return history


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import random
    import numpy as np

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    model   = build_model(num_classes=config.NUM_CLASSES)
    history = train_model(model, num_epochs=config.epochs)
