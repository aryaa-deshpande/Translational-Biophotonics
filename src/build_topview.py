# src/build_topview.py

from pathlib import Path
import re
import csv

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt


# ----------------- Paths -----------------
ROOT = Path(__file__).resolve().parents[1]
DATA_MASK = ROOT / "data" / "masks" / "mask_oct.tif"
DATA_RAW = ROOT / "data" / "raw" / "raw_oct.tif"
PRED_MASKS_DIR = ROOT / "results" / "masks"
OUT_DIR = ROOT / "results" / "topview"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------- Load GT volume -----------------
def load_gt_volume():
    print(f"Loading GT mask volume from {DATA_MASK}")
    gt_vol = tiff.imread(DATA_MASK)  # (D, H, W)
    print("GT volume shape:", gt_vol.shape, gt_vol.dtype)
    return (gt_vol > 0).astype(np.uint8)


# ----------------- Rebuild 3D predicted volume -----------------
def load_pred_volume(gt_shape):
    """
    Create a 3D prediction volume (D, H, W) from per-slice MedSAM2 masks.
    Any slices without a prediction remain zero.
    """
    D, H, W = gt_shape
    pred_vol = np.zeros((D, H, W), dtype=np.uint8)

    pattern = re.compile(r"oct_slice(\d+)_medsam2\.tif")

    print(f"Scanning predicted masks in {PRED_MASKS_DIR}")
    for mask_path in sorted(PRED_MASKS_DIR.glob("oct_slice*_medsam2.tif")):
        m = pattern.match(mask_path.name)
        if not m:
            continue
        slice_idx = int(m.group(1))

        print(f"  Loading slice {slice_idx} from {mask_path.name}")
        mask_2d = tiff.imread(mask_path)
        mask_2d = (mask_2d > 0).astype(np.uint8)

        if mask_2d.shape != (H, W):
            print(
                f"  WARNING: shape mismatch for slice {slice_idx}: "
                f"mask_2d.shape={mask_2d.shape}, expected {(H, W)}"
            )
            # we could resize or skip; for now, just skip if mismatched
            continue

        pred_vol[slice_idx] = mask_2d

    print("Predicted volume built with shape:", pred_vol.shape, pred_vol.dtype)
    # === Save full 3D predicted mask as multi-page TIFF ===
    full_tif_path = OUT_DIR / "medsam2_pred_volume.tif"
    tiff.imwrite(full_tif_path, pred_vol.astype(np.uint8))
    print("Saved full 3D MedSAM2 volume to:", full_tif_path)
    return pred_vol


# ----------------- 3D Dice + topview computation -----------------
def compute_3d_metrics(gt_vol, pred_vol):
    gt = gt_vol.astype(bool)
    pred = pred_vol.astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    gt_pixels = gt.sum()
    pred_pixels = pred.sum()

    if gt_pixels + pred_pixels == 0:
        dice = 1.0
    else:
        dice = 2 * intersection / (gt_pixels + pred_pixels + 1e-6)

    print("\n=== 3D Volume Metrics ===")
    print("GT pixels:      ", gt_pixels)
    print("Pred pixels:    ", pred_pixels)
    print("Intersection:   ", intersection)
    print(f"3D Dice score:  {dice:.4f}")

    return {
        "dice_3d": float(dice),
        "gt_pixels": int(gt_pixels),
        "pred_pixels": int(pred_pixels),
        "intersection": int(intersection),
    }


def build_topviews(gt_vol, pred_vol):
    """
    Collapse along depth (axis=0) to get top-view (en-face) maps.
    """
    # Binary presence maps: at least one HRF pixel along depth
    top_gt = gt_vol.max(axis=0).astype(np.uint8)        # (H, W)
    top_pred = pred_vol.max(axis=0).astype(np.uint8)    # (H, W)

    # Frequency map: in how many slices did MedSAM2 predict HRF at each (y, x)
    freq_pred = pred_vol.sum(axis=0).astype(np.int32)   # (H, W)

    return top_gt, top_pred, freq_pred


def load_intensity_topview():
    """
    Build a structural top-view (en-face) image from the raw OCT volume
    by max-projection along depth.
    """
    print(f"Loading raw OCT volume from {DATA_RAW}")
    vol = tiff.imread(DATA_RAW).astype(np.float32)  # (D, H, W)

    # Max intensity projection along depth
    mip = vol.max(axis=0)  # (H, W)

    # Normalize to [0, 1] for visualization
    mip = (mip - mip.min()) / (np.ptp(mip) + 1e-6)
    return mip


# ----------------- Visualization helpers -----------------
def save_topview_images(top_gt, top_pred, freq_pred):
    intensity_top = load_intensity_topview()
    # GT topview
    gt_path = OUT_DIR / "gt_topview.png"
    plt.figure(figsize=(5, 5))
    plt.imshow(top_gt, cmap="gray")
    plt.title("GT HRF Topview")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(gt_path, dpi=150)
    plt.close()
    print("Saved GT topview to:", gt_path)

    # Pred topview (binary presence)
    pred_path = OUT_DIR / "medsam2_topview.png"
    plt.figure(figsize=(5, 5))
    plt.imshow(top_pred, cmap="gray")
    plt.title("MedSAM2 HRF Topview")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(pred_path, dpi=150)
    plt.close()
    print("Saved MedSAM2 topview to:", pred_path)

    # Frequency map (how many slices)
    freq_path = OUT_DIR / "medsam2_frequency_topview.png"
    plt.figure(figsize=(5, 5))
    # add small epsilon to avoid log(0) if you later use log-scale; for now just show raw
    plt.imshow(freq_pred, cmap="viridis")
    plt.title("MedSAM2 HRF Frequency (number of slices)")
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(freq_path, dpi=150)
    plt.close()
    print("Saved MedSAM2 frequency map to:", freq_path)

    # Overlay GT vs prediction
    overlay_path = OUT_DIR / "gt_vs_medsam2_topview_overlay.png"
    plt.figure(figsize=(5, 5))
    plt.imshow(top_gt, cmap="gray")
    # show prediction as transparent overlay
    plt.imshow(np.ma.masked_where(top_pred == 0, top_pred), alpha=0.5)
    plt.title("GT (gray) + MedSAM2 (overlay)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=150)
    plt.close()
    print("Saved overlay topview to:", overlay_path)

        # Overlay MedSAM2 on structural top-view
    struct_overlay_path = OUT_DIR / "medsam2_on_structural_topview.png"
    plt.figure(figsize=(5, 5))
    plt.imshow(intensity_top, cmap="gray")
    plt.imshow(np.ma.masked_where(top_pred == 0, top_pred), alpha=0.5)
    plt.title("Structural topview + MedSAM2 HRF")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(struct_overlay_path, dpi=150)
    plt.close()
    print("Saved MedSAM2 on structural topview to:", struct_overlay_path)

    # Overlay GT on structural top-view
    struct_gt_overlay_path = OUT_DIR / "gt_on_structural_topview.png"
    plt.figure(figsize=(5, 5))
    plt.imshow(intensity_top, cmap="gray")
    plt.imshow(np.ma.masked_where(top_gt == 0, top_gt), alpha=0.5)
    plt.title("Structural topview + GT HRF")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(struct_gt_overlay_path, dpi=150)
    plt.close()
    print("Saved GT on structural topview to:", struct_gt_overlay_path)

    # Also save TIFFs for further analysis
    tiff.imwrite(OUT_DIR / "gt_topview.tif", (top_gt * 255).astype(np.uint8))
    tiff.imwrite(OUT_DIR / "medsam2_topview.tif", (top_pred * 255).astype(np.uint8))
    print("Saved GT and MedSAM2 topview TIFFs.")


def save_metrics_csv(metrics_3d):
    csv_path = OUT_DIR / "medsam2_topview_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dice_3d", "gt_pixels", "pred_pixels", "intersection"],
        )
        writer.writeheader()
        writer.writerow(metrics_3d)
    print("Saved 3D metrics CSV to:", csv_path)


# ----------------- Main -----------------
def main():
    gt_vol = load_gt_volume()
    pred_vol = load_pred_volume(gt_vol.shape)

    metrics_3d = compute_3d_metrics(gt_vol, pred_vol)

    top_gt, top_pred, freq_pred = build_topviews(gt_vol, pred_vol)
    save_topview_images(top_gt, top_pred, freq_pred)
    save_metrics_csv(metrics_3d)


if __name__ == "__main__":
    main()