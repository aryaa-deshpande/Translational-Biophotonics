import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.morphology import remove_small_objects, binary_opening, disk

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]

# CHANGE THIS to your new OCT file:
raw_path = ROOT / "data" / "raw" / "Confidential_merged_098OD-2_16bit_log.tif"

out_dir = ROOT / "results" / "hrf_candidates_other_eye"
mask_dir = out_dir / "masks"
overlay_dir = out_dir / "overlays"

mask_dir.mkdir(parents=True, exist_ok=True)
overlay_dir.mkdir(parents=True, exist_ok=True)

print("Loading new OCT volume from:", raw_path)
raw_vol = tiff.imread(str(raw_path))
print("raw_vol shape:", raw_vol.shape, "dtype:", raw_vol.dtype)

# make sure it's (z, h, w)
if raw_vol.ndim == 3:
    z_dim, H, W = raw_vol.shape
else:
    raise ValueError(f"Unexpected raw_vol shape: {raw_vol.shape}")

# ---------- hyper-parameters (you can tune these) ----------
clip_low = 20      # percentile for intensity clipping
clip_high = 99.5   # percentile for intensity clipping
thresh_factor = 2  # > mean + 2*std
min_size = 30      # remove tiny specks

hits = 0
nonempty_slices = 0

for z in range(z_dim):
    sl = raw_vol[z].astype(np.float32)

    # 1) normalize / clip
    lo = np.percentile(sl, clip_low)
    hi = np.percentile(sl, clip_high)
    sl_clipped = np.clip(sl, lo, hi)
    sl_norm = (sl_clipped - lo) / (hi - lo + 1e-6)

    # 2) simple “bright spot” detector
    mu = sl_norm.mean()
    sigma = sl_norm.std()
    thresh = mu + thresh_factor * sigma
    cand = sl_norm > thresh

    # 3) clean up
    cand = remove_small_objects(cand, min_size=min_size)
    cand = binary_opening(cand, footprint=disk(1))

    cand_pixels = int(cand.sum())
    if cand_pixels == 0:
        # nothing interesting in this slice
        continue

    nonempty_slices += 1

    # save binary candidate mask
    mask_path = mask_dir / f"slice_{z:03d}_candidates.tif"
    tiff.imwrite(str(mask_path), cand.astype(np.uint8) * 255)

    # build overlay for visualization
    rgb = np.stack([sl_norm, sl_norm, sl_norm], axis=-1)
    rgb = np.clip(rgb, 0, 1)

    # red overlay for candidates
    rgb[cand, 0] = 1.0   # R
    rgb[cand, 1] *= 0.3  # darken G
    rgb[cand, 2] *= 0.3  # darken B

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb, cmap="gray")
    ax.set_title(f"Slice {z} – HRF candidates (red)")
    ax.axis("off")

    overlay_path = overlay_dir / f"slice_{z:03d}_overlay.png"
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=200)
    plt.close(fig)

    print(f"[Slice {z}] cand_pixels={cand_pixels}  -> saved mask+overlay")

print("\n========== Unlabeled HRF candidate run ==========")
print("Total slices:", z_dim)
print("Slices with any candidates:", nonempty_slices)
print("Masks saved to:", mask_dir)
print("Overlays saved to:", overlay_dir)