import numpy as np
import tifffile as tiff
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "raw_oct.tif"
PRED_PATH = ROOT / "results" / "topview" / "medsam2_pred_volume.tif"
OUT_TIF = ROOT / "results" / "topview" / "medsam2_structural_overlay_volume.tif"

print("Loading raw OCT...")
raw = tiff.imread(RAW_PATH)
raw = raw.astype(np.float32)

print("Loading predicted mask volume...")
pred = tiff.imread(PRED_PATH)
pred = pred.astype(np.uint8)

assert raw.shape == pred.shape, "Raw and pred volumes mismatch!"

print("Building overlay slices...")
overlay_slices = []

for i in range(raw.shape[0]):
    sl = raw[i]
    mask = pred[i]

    # Normalize OCT slice to [0,255]
    sl_norm = (sl - sl.min()) / (np.ptp(sl) + 1e-6)
    sl_norm = (sl_norm * 255).astype(np.uint8)

    # Convert to RGB
    rgb = np.stack([sl_norm, sl_norm, sl_norm], axis=-1)

    # Paint mask in RED
    rgb[mask > 0] = [255, 0, 0]

    overlay_slices.append(rgb)

overlay_slices = np.stack(overlay_slices)

print("Saving overlay volume as TIFF...")
tiff.imwrite(
    OUT_TIF,
    overlay_slices,
    photometric="rgb",
)

print(f"Saved structural overlay volume to:\n{OUT_TIF}")