# src/compare_sam2_medsam2.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import csv
import tifffile as tiff

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]

raw_path = ROOT / "data" / "raw" / "098os-2.tif"
gt_path = ROOT / "data" / "masks" / "mask_oct.tif"

med_dir = ROOT / "results" / "masks"
sam2_npy_dir = ROOT / "results" / "Output_final"

out_img_dir = ROOT / "results" / "comparison" / "full_panels"
out_img_dir.mkdir(parents=True, exist_ok=True)

metrics_csv = ROOT / "results" / "comparison" / "sam2_vs_medsam2_metrics.csv"

# slices we care about
SLICES = [150, 170, 175, 180]


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    """Binary Dice score."""
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum() + 1e-6
    return 2.0 * inter / denom


print("Loading volumes...")
raw_vol = tiff.imread(raw_path)   # (Z, H, W)
gt_vol = tiff.imread(gt_path)     # (Z, H, W)

print(f"raw_vol shape: {raw_vol.shape}, dtype: {raw_vol.dtype}")
print(f"gt_vol  shape: {gt_vol.shape}, dtype: {gt_vol.dtype}")

all_rows = []
header = [
    "slice",
    "dice_medsam2_vs_gt",
    "dice_sam2_vs_gt",
    "dice_medsam2_vs_sam2",
    "gt_pixels",
    "medsam2_pixels",
    "sam2_pixels",
]

for s in SLICES:
    print(f"\n=== Slice {s} ===")

    # ---------- 1) raw + GT ----------
    raw_slice = raw_vol[s, :, :].astype(np.float32)
    gt_slice = (gt_vol[s, :, :] > 0).astype(np.uint8)

    # normalize raw for display
    raw_norm = (raw_slice - raw_slice.min()) / (np.ptp(raw_slice) + 1e-6)

    # make RGB version of raw for nicer plotting
    raw_rgb = np.stack([raw_norm] * 3, axis=-1)

    # ---------- 2) MedSAM2 mask ----------
    med_path = med_dir / f"oct_slice{s}_medsam2.tif"
    if not med_path.exists():
        print(f"[WARN] Missing MedSAM2 mask: {med_path}")
        continue

    med_mask = tiff.imread(med_path)
    # ensure HxW
    if med_mask.ndim == 3:
        med_mask = med_mask[0]

    med_mask_bin = (med_mask > 0).astype(np.uint8)

    # ---------- 3) SAM2 mask from .npy ----------
    sam2_npy_path = sam2_npy_dir / f"{s}.npy"
    if not sam2_npy_path.exists():
        print(f"[WARN] Missing SAM2 npy: {sam2_npy_path}")
        continue

    sam2_arr = np.load(sam2_npy_path)

    # many SAM2 outputs are float probs in [0,1] or logits; binarize at 0.5
    if sam2_arr.dtype != np.uint8:
        sam2_arr = (sam2_arr > 0.5).astype(np.uint8)

    # shape fix: resize to match GT if needed
    if sam2_arr.shape != gt_slice.shape:
        print(f"  resizing SAM2 from {sam2_arr.shape} -> {gt_slice.shape}")
        pil = Image.fromarray((sam2_arr * 255).astype(np.uint8))
        H, W = gt_slice.shape
        pil = pil.resize((W, H), resample=Image.NEAREST)
        sam2_arr = (np.array(pil) > 127).astype(np.uint8)

    sam2_mask_bin = sam2_arr

    # ---------- 4) metrics ----------
    gt_pixels = int(gt_slice.sum())
    med_pixels = int(med_mask_bin.sum())
    sam2_pixels = int(sam2_mask_bin.sum())

    dice_med_gt = float(dice_score(med_mask_bin, gt_slice))
    dice_sam_gt = float(dice_score(sam2_mask_bin, gt_slice))
    dice_med_sam = float(dice_score(med_mask_bin, sam2_mask_bin))

    print(f"  GT pixels:        {gt_pixels}")
    print(f"  MedSAM2 pixels:   {med_pixels} (Dice vs GT:  {dice_med_gt:.4f})")
    print(f"  SAM2 pixels:      {sam2_pixels} (Dice vs GT:  {dice_sam_gt:.4f})")
    print(f"  Dice MedSAM2 vs SAM2: {dice_med_sam:.4f}")

    all_rows.append(
        [
            s,
            dice_med_gt,
            dice_sam_gt,
            dice_med_sam,
            gt_pixels,
            med_pixels,
            sam2_pixels,
        ]
    )

    # ---------- 5) build panel figure ----------
    # GT overlay
    gt_overlay = raw_rgb.copy()
    gt_overlay[gt_slice == 1] = [1.0, 0.0, 0.0]  # red

    # MedSAM2 overlay
    med_overlay = raw_rgb.copy()
    med_overlay[med_mask_bin == 1] = [0.0, 1.0, 0.0]  # green

    # SAM2 overlay
    sam_overlay = raw_rgb.copy()
    sam_overlay[sam2_mask_bin == 1] = [0.0, 0.7, 1.0]  # cyan-ish

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(raw_rgb, cmap="gray")
    axes[0].set_title(f"Slice {s} – OCT")
    axes[0].axis("off")

    axes[1].imshow(gt_overlay)
    axes[1].set_title("GT mask")
    axes[1].axis("off")

    axes[2].imshow(med_overlay)
    axes[2].set_title(f"MedSAM2 (Dice={dice_med_gt:.2f})")
    axes[2].axis("off")

    axes[3].imshow(sam_overlay)
    axes[3].set_title(f"SAM2 (Dice={dice_sam_gt:.2f})")
    axes[3].axis("off")

    fig.tight_layout()
    out_path = out_img_dir / f"slice{s}_oct_gt_medsam2_sam2.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved panel: {out_path}")

# ---------- 6) save metrics CSV ----------
if all_rows:
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)
    print(f"\nSaved metrics to: {metrics_csv}")
else:
    print("\nNo slices processed – check paths / filenames.")