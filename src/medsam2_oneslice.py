from pathlib import Path
import sys

import numpy as np
import tifffile as tiff

import torch
from omegaconf import OmegaConf
import functools


# Register missing OmegaConf resolvers used in SAM2 configs
def _times_resolver(*args):
    # Multiply all arguments together
    vals = [float(a) for a in args]
    return functools.reduce(lambda x, y: x * y, vals, 1.0)

def _divide_resolver(*args):
    # Divide first argument by the rest: a / b / c / ...
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
        # It's fine if it's already registered
        pass

root = Path(__file__).resolve().parents[1]
medsam2_root = root / "MedSAM2"
sys.path.append(str(medsam2_root))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_oct_and_mask():
    oct_path = root / "data" / "raw" / "raw_oct.tif"
    mask_path = root / "data" / "masks" / "mask_oct.tif"

    print(f"Loading OCT from {oct_path}")
    vol = tiff.imread(oct_path)    # (D, H, W)
    print("OCT volume shape:", vol.shape, vol.dtype)

    print(f"Loading GT mask from {mask_path}")
    mask_vol = tiff.imread(mask_path)
    print("Mask volume shape:", mask_vol.shape, mask_vol.dtype)

    assert vol.shape == mask_vol.shape, "OCT and mask shapes must match"

    D = vol.shape[0]

    # find all slice indices that actually have HRF (non-zero mask)
    hrfs_slices = [i for i in range(D) if np.any(mask_vol[i] > 0)]
    if not hrfs_slices:
        raise ValueError("No HRF found in any slice of the GT mask!")

    # pick the "middle" slice among the HRF slices
    mid_idx = hrfs_slices[len(hrfs_slices) // 2]
    print(f"HRF present in slices: {hrfs_slices[:10]}{'...' if len(hrfs_slices) > 10 else ''}")
    print(f"Using slice index with HRF: {mid_idx}")

    oct_slice = vol[mid_idx]                     # (H, W)
    mask_slice = (mask_vol[mid_idx] > 0).astype(np.uint8)

    return oct_slice, mask_slice, mid_idx

def to_rgb(oct_slice):
    """Convert a single grayscale slice to 3-channel uint8 [0,255]."""
    sl = oct_slice.astype(np.float32)
    sl -= sl.min()
    if sl.max() > 0:
        sl /= sl.max()
    sl = (sl * 255).astype(np.uint8)        # (H, W) 0–255

    rgb = np.stack([sl, sl, sl], axis=-1)   # (H, W, 3)
    return rgb


def mask2D_to_bbox(mask2d, max_shift=0):
    """
    Compute a bounding box [x_min, y_min, x_max, y_max] from a 2D binary mask.
    This mirrors the function in medsam2_infer_3D_CT.py but without random shift.
    """
    ys, xs = np.where(mask2d > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None  # no HRF on this slice

    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    H, W = mask2d.shape

    if max_shift > 0:
        shift = np.random.randint(0, max_shift + 1)
        x_min = max(0, x_min - shift)
        x_max = min(W - 1, x_max + shift)
        y_min = max(0, y_min - shift)
        y_max = min(H - 1, y_max + shift)

    box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    return box

def build_medsam2_model():
    """Build the MedSAM2 model using their config + checkpoint."""
    cfg_name = "configs/sam2.1_hiera_t512.yaml"
    ckpt = medsam2_root / "checkpoints" / "MedSAM2_latest.pt"

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


def run_one_slice():
    oct_slice, mask_slice, idx = load_oct_and_mask()
    rgb_slice = to_rgb(oct_slice)

    # get bbox from GT HRF on this slice
    box = mask2D_to_bbox(mask_slice, max_shift=0)
    if box is None:
        print("No HRF found on this slice (mask is empty). "
              "Try changing slice_idx in load_oct_and_mask().")
        return

    print("Bounding box (x_min, y_min, x_max, y_max):", box)

    predictor = build_medsam2_model()

    # set image
    predictor.set_image(rgb_slice)

    # predictor expects box with shape (N, 4)
    box_input = box[None, :]  # (1, 4)

    masks, ious, lowres = predictor.predict(
        box=box_input,
        multimask_output=False,   # one mask
        return_logits=False,
        normalize_coords=True,    # uses orig image size internally
    )

    # masks: (1, 1, H, W) or (1, H, W) depending on version; handle both
    pred_mask = masks[0]
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[0]

    pred_mask_bin = (pred_mask > 0).astype(np.uint8)
    print("Pred mask shape:", pred_mask_bin.shape, pred_mask_bin.dtype)

    print("Predicted mask shape:", pred_mask.shape)
    print("Unique values in prediction:", np.unique(pred_mask))

    # Compute Dice Score
    gt = mask_slice.astype(bool)
    pred = pred_mask.astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    dice = 2 * intersection / (gt.sum() + pred.sum() + 1e-6)

    print("Dice score:", dice)
    print("GT pixels:", gt.sum())
    print("Predicted pixels:", pred.sum())
    print("Intersection:", intersection)


    false_positive = np.logical_and(~gt, pred).sum()
    false_negative = np.logical_and(gt, ~pred).sum()

    print("False positive pixels:", false_positive)
    print("False negative pixels:", false_negative)

    overlap_ratio = intersection / gt.sum()
    print("Overlap ratio (what % of GT was correctly found):", overlap_ratio)

    out_tif = root / "results" / "masks" / f"oct_slice{idx}_medsam2.tif"
    out_overlay = root / "results" / "figures" / f"oct_slice{idx}_medsam2_overlay.png"

    # save predicted mask
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(out_tif, pred_mask_bin)
    print(f"Saved predicted mask to: {out_tif}")

    # quick overlay for sanity
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(oct_slice, cmap="gray")
    plt.title("OCT slice")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(oct_slice, cmap="gray")
    plt.imshow(mask_slice, alpha=0.4)
    plt.title("GT HRF mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(oct_slice, cmap="gray")
    plt.imshow(pred_mask_bin, alpha=0.4)
    plt.title("MedSAM2 predicted mask")
    plt.axis("off")

    out_overlay.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_overlay, dpi=200)
    print(f"Saved overlay to: {out_overlay}")
    plt.show()


if __name__ == "__main__":
    run_one_slice()