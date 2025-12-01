

import functools
from pathlib import Path
import numpy as np
import tifffile as tiff
import torch
from omegaconf import OmegaConf

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor



#  Paths 
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "raw_oct.tif"
DATA_MASK = ROOT / "data" / "masks" / "mask_oct.tif"
RESULTS_DIR = ROOT / "results"


#  OmegaConf resolvers 
def _times_resolver(*args):
    vals = [float(a) for a in args]
    return functools.reduce(lambda x, y: x * y, vals, 1.0)

def _divide_resolver(*args):
    vals = [float(a) for a in args]
    if not vals:
        return 1.0
    result = vals[0]
    for v in vals[1:]:
        result /= v
    return result

for name, fn in {"times": _times_resolver, "divide": _divide_resolver}.items():
    try:
        OmegaConf.register_new_resolver(name, fn)
    except Exception:
        pass


#  Data loading 
def load_full_volumes():
    print(f"Loading OCT from {DATA_RAW}")
    vol = tiff.imread(DATA_RAW)  # (D, H, W)
    print("OCT volume shape:", vol.shape, vol.dtype)

    print(f"Loading GT mask from {DATA_MASK}")
    mask_vol = tiff.imread(DATA_MASK)
    print("Mask volume shape:", mask_vol.shape, mask_vol.dtype)

    assert vol.shape == mask_vol.shape, "OCT and mask shapes must match"

    D = vol.shape[0]
    hrf_slices = [i for i in range(D) if np.any(mask_vol[i] > 0)]
    print("HRF present in slices:", hrf_slices[:10], "..." if len(hrf_slices) > 10 else "")

    return vol, mask_vol, hrf_slices


#  Build MedSAM2 
def build_medsam2_model():
    cfg_name = "configs/sam2.1_hiera_t512.yaml"
    ckpt = ROOT / "MedSAM2" / "checkpoints" / "MedSAM2_latest.pt"

    print(f"Using config name: {cfg_name}")
    print(f"Using checkpoint: {ckpt}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Building model on device:", device)

    sam_model = build_sam2(
        config_file=cfg_name,
        ckpt_path=str(ckpt),
        device=device,
        mode="eval",
    )
    predictor = SAM2ImagePredictor(sam_model)
    return predictor


#  Single-slice processing 
def process_single_slice(slice_idx, vol, mask_vol, predictor, results_dir, save_fig=True):
    """
    Run MedSAM2 on one slice, compute metrics, and (optionally) save mask + overlay.
    Returns a dict of metrics.
    """
    import matplotlib.pyplot as plt

    oct_slice = vol[slice_idx]                    # (H, W)
    gt_slice = (mask_vol[slice_idx] > 0).astype(np.uint8)

    ys, xs = np.where(gt_slice > 0)
    if len(xs) == 0:
        print(f"[Slice {slice_idx}] No HRF in GT mask, skipping.")
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    print(f"[Slice {slice_idx}] Bbox:", bbox.tolist())

    # 🔁 This block should match what you already had working for one slice:
    rgb_slice = np.stack([oct_slice, oct_slice, oct_slice], axis=-1).astype(np.float32)
    rgb_slice = (rgb_slice - rgb_slice.min()) / (np.ptp(rgb_slice) + 1e-6)

    predictor.set_image(rgb_slice)
    masks, scores, logits = predictor.predict(
        box=bbox[None, :],
        multimask_output=False,
    )
    pred_mask = masks[0].astype(np.uint8)

    print("Pred mask shape:", pred_mask.shape, pred_mask.dtype)
    print("Unique values in prediction:", np.unique(pred_mask))

    # metrics (this is exactly what you just ran)
    gt = gt_slice.astype(bool)
    pred = pred_mask.astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    gt_pixels = gt.sum()
    pred_pixels = pred.sum()

    if gt_pixels + pred_pixels == 0:
        dice = 1.0
    else:
        dice = 2 * intersection / (gt_pixels + pred_pixels + 1e-6)

    false_positive = np.logical_and(~gt, pred).sum()
    false_negative = np.logical_and(gt, ~pred).sum()
    overlap_ratio = intersection / (gt_pixels + 1e-6)

    print(
        f"[Slice {slice_idx}] Dice: {dice:.4f}, "
        f"GT: {gt_pixels}, Pred: {pred_pixels}, "
        f"FP: {false_positive}, FN: {false_negative}"
    )

    masks_dir = results_dir / "masks"
    figs_dir = results_dir / "figures"
    masks_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    mask_path = masks_dir / f"oct_slice{slice_idx}_medsam2.tif"
    tiff.imwrite(mask_path, (pred_mask * 255).astype(np.uint8))
    print("Saved predicted mask to:", mask_path)

    if save_fig:
        fig_path = figs_dir / f"oct_slice{slice_idx}_medsam2_overlay.png"
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(oct_slice, cmap="gray")
        plt.title(f"OCT slice {slice_idx}")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(oct_slice, cmap="gray")
        plt.imshow(gt_slice, alpha=0.4)
        plt.title("GT HRF")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(oct_slice, cmap="gray")
        plt.imshow(pred_mask, alpha=0.4)
        plt.title("MedSAM2 HRF")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print("Saved overlay to:", fig_path)

    return {
        "slice_idx": slice_idx,
        "dice": float(dice),
        "gt_pixels": int(gt_pixels),
        "pred_pixels": int(pred_pixels),
        "intersection": int(intersection),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "overlap_ratio": float(overlap_ratio),
    }