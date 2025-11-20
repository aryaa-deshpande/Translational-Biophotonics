from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


root = Path(__file__).resolve().parents[1]

oct_path = root / "data" / "raw" / "raw_oct.tif"
mask_path = root / "data" / "masks" / "mask_oct.tif"

print(f"OCT path : {oct_path}")
print(f"Mask path: {mask_path}")


oct_img = tiff.imread(oct_path)
mask_img = tiff.imread(mask_path)

print("OCT shape :", oct_img.shape)
print("Mask shape:", mask_img.shape)
print("OCT dtype :", oct_img.dtype)
print("Mask dtype:", mask_img.dtype)

# make mask binary just in case
mask_bin = (mask_img > 0).astype(np.uint8)

# handle 2D vs 3D
if oct_img.ndim == 2:
    print("Detected: 2D OCT")
    oct_slice = oct_img
    mask_slice = mask_bin

elif oct_img.ndim == 3:
    n_slices = oct_img.shape[0]
    print(f"Detected: 3D OCT volume with {n_slices} slices")

    mid = n_slices // 2
    print(f"Using middle slice {mid} for quick visualization")

    oct_slice = oct_img[mid]
    mask_slice = mask_bin[mid] if mask_bin.ndim == 3 else mask_bin

else:
    raise ValueError(f"Unexpected OCT dimensions: {oct_img.ndim}")


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(oct_slice, cmap="gray")
axes[0].set_title("OCT slice")
axes[0].axis("off")

axes[1].imshow(oct_slice, cmap="gray")
axes[1].imshow(mask_slice, alpha=0.4)
axes[1].set_title("OCT + HRF GT mask")
axes[1].axis("off")

out_dir = root / "results" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "oct_case1_overlay.png"

plt.tight_layout()
plt.savefig(out_path, dpi=200)
print(f"Saved overlay to: {out_path}")

plt.show()


# --- top view (en-face projection) ---
topview = mask_bin.max(axis=0)

plt.figure(figsize=(5,5))
plt.imshow(topview, cmap="hot")
plt.title("Top View (HRF density)")
plt.axis("off")

out_top = out_dir / "oct_case1_topview.png"
plt.savefig(out_top, dpi=200)
print(f"Saved top view to: {out_top}")
plt.show()