# ==============================================================================
# model.py
# ResNet-18 with custom classification head.
# Taken directly from Cell 7 of the project notebook.
# ==============================================================================

import torch.nn as nn
from torchvision import models

import config


# ==============================================================================
# CELL 7 — CNN architecture: ResNet-18 with custom head
# ==============================================================================
"""
Architecture rationale:

  ResNet-18 pretrained on ImageNet is the best choice here because:
  - With only 240 training images per class, a scratch CNN would overfit
    catastrophically within the first few epochs.
  - ImageNet features (edges, textures, shapes) transfer well to cytology.
  - ResNet-18 is small enough to train in <5 min/epoch on Colab T4.

Transfer learning strategy:
  - FREEZE  layer1: lowest-level edge/color detectors (already good)
  - FINETUNE layer2, layer3, layer4: texture / shape patterns
  - REPLACE fc: custom head with dropout for regularisation

  With 5600 training images we have enough data to fine-tune layer2
  in addition to layer3 and layer4, giving the model more flexibility
  to learn cell-specific morphological features.
"""


def build_model(num_classes: int = config.NUM_CLASSES) -> nn.Module:
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Freeze only layer1 — sufficient with 5600 training images to fine-tune layer2 onward
    for layer in [model.layer1]:
        for param in layer.parameters():
            param.requires_grad = False

    # Custom classification head
    in_feats = model.fc.in_features   # 512 for ResNet-18
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_feats, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),   # raw logits; CrossEntropyLoss handles log-softmax
    )
    return model


# Alias used by interface.py
class CellClassifierCNN(nn.Module):
    """
    Thin wrapper around build_model() so interface.py can import a class
    (TheModel) while the underlying architecture stays identical to the
    notebook's build_model() function.
    """
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self._model = build_model(num_classes=num_classes)

    def forward(self, x):
        return self._model(x)

    # Expose ResNet sub-modules so GradCAM / fine-tuning code can address them
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import torch
    model = build_model()
    dummy = torch.randn(2, config.input_channels, config.resize_y, config.resize_x)
    out   = model(dummy)
    n_tr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f'Output shape     : {tuple(out.shape)}')
    print(f'Parameters       : {n_tr:,} trainable / {n_tot:,} total  ({100*n_tr/n_tot:.1f}% unfrozen)')
