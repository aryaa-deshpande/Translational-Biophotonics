import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]

raw_path = ROOT / "data" / "raw" / "raw_oct.tif"
gt_path = ROOT / "data" / "masks" / "mask_oct.tif"

med_overlay_dir = ROOT / "results" / "figures"
sam2_dir = ROOT / "SAM2_Results"
out_dir = ROOT / "results" / "comparison_full"
out_dir.mkdir(parents=True, exist_ok=True)

slices = [150, 170, 175, 180]

print(f"Loading volumes:\n  raw: {raw_path}\n  gt:  {gt_path}")
raw_vol = tiff.imread(raw_path)
gt_vol = tiff.imread(gt_path)

for s in slices:
    print(f"\n=== Slice {s} ===")

    # ---- 1) Load raw slice ----
    raw_slice = raw_vol[s, ...]  # shape (H, W)
    raw_norm = (raw_slice - raw_slice.min()) / (np.ptp(raw_slice) + 1e-6)

    # ---- 2) GT overlay ----
    gt_slice = gt_vol[s, ...] > 0

    rgb = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
    gt_rgb = rgb.copy()

    # Red GT overlay
    alpha = 0.6
    gt_rgb[gt_slice] = (
        alpha * np.array([1.0, 0.0, 0.0]) + (1 - alpha) * gt_rgb[gt_slice]
    )

    # ---- 3) MedSAM2 overlay ----
    med_overlay_path = med_overlay_dir / f"oct_slice{s}_medsam2_overlay.png"
    if not med_overlay_path.exists():
        print(f"[WARN] Missing MedSAM2 overlay: {med_overlay_path}")
        continue

    med_img = Image.open(med_overlay_path)

    # ---- 4) SAM2 ----
    sam2_frame_path = sam2_dir / f"Fram_{s}.png"
    if not sam2_frame_path.exists():
        print(f"[WARN] Missing SAM2 image: {sam2_frame_path}")
        continue

    fram_img = Image.open(sam2_frame_path)
    w, h = fram_img.size

    tile_w = w // 3  # layout = [Original | GT | SAM2]
    sam2_panel = fram_img.crop((2 * tile_w, 0, w, h))

    # Resize both to match OCT slice shape
    target_size = (raw_slice.shape[1], raw_slice.shape[0])  # (W, H)
    sam2_resized = sam2_panel.resize(target_size)
    med_img_resized = med_img.resize(target_size)

    # ---- 5) Build final 4-panel figure ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(raw_norm, cmap="gray")
    axes[0].set_title(f"Original – slice {s}")
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(med_img_resized)
    axes[2].set_title("MedSAM2")
    axes[2].axis("off")

    axes[3].imshow(sam2_resized)
    axes[3].set_title("SAM2")
    axes[3].axis("off")

    fig.tight_layout()
    out_path = out_dir / f"slice{s}_orig_gt_medsam2_sam2.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved {out_path}")