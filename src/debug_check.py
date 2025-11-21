from pathlib import Path
import numpy as np
import tifffile as tiff

root = Path(__file__).resolve().parents[1]  # if this is in src/
vol_path = root / "results" / "topview" / "medsam2_pred_volume.tif"

vol = tiff.imread(vol_path)
print("Volume shape:", vol.shape)
print("dtype:", vol.dtype)
print("min, max:", vol.min(), vol.max())

nonzero = np.sum(vol > 0)
print("Number of non-zero voxels:", nonzero)

# Check some slices we KNOW should have HRF
for idx in [148, 171, 185]:
    sl = vol[idx]
    print(f"Slice {idx} -> unique values:", np.unique(sl), " nonzero pixels:", np.sum(sl > 0))