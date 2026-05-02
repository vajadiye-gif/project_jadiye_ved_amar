# ==============================================================================
# predict.py
# Inference and Grad-CAM utilities.
# Taken directly from Cells 11 & 14 of the project notebook.
# ==============================================================================

import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from model   import build_model
from dataset import eval_transform


# ---------------------------------------------------------------------------
# Model loader (cached so weights aren't reloaded on every call)
# ---------------------------------------------------------------------------
_cached_model  = None
_cached_device = None


def _load_model(weights_path: str = config.BEST_WEIGHTS, device=None):
    global _cached_model, _cached_device

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if _cached_model is not None and _cached_device == device:
        return _cached_model

    model = build_model(num_classes=config.NUM_CLASSES)

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Weights not found at '{weights_path}'. "
            "Run train.py first to generate final_weights.pth."
        )

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    _cached_model  = model
    _cached_device = device
    return model


# ==============================================================================
# CELL 11 — Inference function
# ==============================================================================

def classify_cells(
    list_of_image_paths: List[str],
    weights_path: str = config.BEST_WEIGHTS,
    device=None,
    return_probs: bool = False,
):
    """
    Run inference on a list of image file paths.

    Parameters
    ----------
    list_of_image_paths : list of path strings pointing to images in data/
    weights_path        : path to final_weights.pth
    device              : torch.device or str — auto-detected if None
    return_probs        : if True, also return softmax probability array

    Returns
    -------
    labels      : List[str]   — predicted class name for each image
    probs (opt) : np.ndarray of shape (N, NUM_CLASSES) if return_probs=True
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = _load_model(weights_path=weights_path, device=device)

    all_preds = []
    all_probs = []

    # Process in mini-batches (mirrors Cell 11 test loop)
    for start in range(0, len(list_of_image_paths), config.batch_size):
        batch_paths = list_of_image_paths[start : start + config.batch_size]

        imgs = torch.stack([
            eval_transform(Image.open(p).convert('RGB'))
            for p in batch_paths
        ]).to(device)

        with torch.no_grad():
            logits = model(imgs)                              # (B, NUM_CLASSES)
            probs  = F.softmax(logits, dim=1).cpu().numpy()  # (B, NUM_CLASSES)
            preds  = logits.argmax(1).cpu().numpy()           # (B,)

        all_preds.extend(preds.tolist())
        all_probs.append(probs)

    labels = [config.CLASS_NAMES[p] for p in all_preds]

    if return_probs:
        return labels, np.concatenate(all_probs, axis=0)
    return labels


# ==============================================================================
# CELL 14 — Grad-CAM saliency (single image helper)
# ==============================================================================

def get_gradcam(
    image_path:   str,
    weights_path: str = config.BEST_WEIGHTS,
    device=None,
):
    """
    Compute a Grad-CAM heatmap for a single image using pytorch_grad_cam.

    Returns
    -------
    raw_np  : np.ndarray (H, W, 3) float32 in [0,1]  — original image
    overlay : np.ndarray (H, W, 3) uint8              — Grad-CAM overlay
    label   : str  — predicted class name
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = _load_model(weights_path=weights_path, device=device)

    raw_pil = Image.open(image_path).convert('RGB').resize(
        (config.resize_x, config.resize_y)
    )
    raw_np  = np.array(raw_pil, dtype=np.float32) / 255.0
    input_t = eval_transform(raw_pil).unsqueeze(0).to(device)

    cam     = GradCAM(model=model, target_layers=[model.layer4[-1]])
    heatmap = cam(input_tensor=input_t, targets=None)[0]   # (H, W)
    overlay = show_cam_on_image(raw_np, heatmap, use_rgb=True)

    with torch.no_grad():
        pred = model(input_t).argmax(1).item()
    label = config.CLASS_NAMES[pred]

    return raw_np, overlay, label


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import glob
    samples = sorted(glob.glob(
        os.path.join(config.DATA_DIR, '**', '*.jpg'), recursive=True
    ))[:5]

    if not samples:
        print('No images found in data/ — add .jpg files first.')
    else:
        preds = classify_cells(samples)
        for path, pred in zip(samples, preds):
            print(f'{os.path.basename(path):40s}  →  {pred}')
