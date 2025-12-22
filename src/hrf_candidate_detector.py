from pathlib import Path
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy import ndimage as ndi  # pip install scipy if needed


# -------- Paths --------
ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "Confidential_merged_098OD-2_16bit_log.tif"
GT_PATH = ROOT / "data" / "masks" / "mask_oct.tif"

OUT_DIR = ROOT / "results" / "hrf_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------- Helpers --------
def normalize_slice(sl: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D slice to [0, 1].
    """
    sl = sl.astype(np.float32)
    min_v = float(sl.min())
    max_v = float(sl.max())
    return (sl - min_v) / (max_v - min_v + 1e-6)


def detect_candidates(
    slice_norm: np.ndarray,
    k: float = 1.5,
    min_area: int = 10,
    max_area: int = 1200,
    min_aspect: float = 1.2,
) -> np.ndarray:
    """
    Very simple HRF candidate detector based on intensity + shape.

    - slice_norm: 2D float array in [0, 1]
    Returns:
        binary 2D mask (True where we think there might be HRF).
    """
    mean = float(slice_norm.mean())
    std = float(slice_norm.std())
    thr = mean + k * std

    # Brightness threshold
    binary = slice_norm > thr

    # Connected components
    labels, num = ndi.label(binary)
    out = np.zeros_like(binary, dtype=bool)

    for lab in range(1, num + 1):
        comp = labels == lab
        area = int(comp.sum())
        if area < min_area or area > max_area:
            continue

        ys, xs = np.where(comp)
        h = int(ys.max() - ys.min() + 1)
        w = int(xs.max() - xs.min() + 1)
        aspect = h / (w + 1e-6)

        # HRFs tend to be taller than wide
        if aspect < min_aspect:
            continue

        out |= comp

    return out


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dice coefficient between two binary masks.
    ONLY meaningful if both masks refer to the same eye / registration!
    """
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def make_debug_panel(
    z: int,
    raw_slice: np.ndarray,
    gt_slice: np.ndarray,
    cand_slice: np.ndarray,
    save_path: Path,
) -> None:
    """
    Make a quick visualization:
        - background: OCT
        - GT HRF: green
        - candidate HRF: red
    """
    raw_norm = normalize_slice(raw_slice)
    gt = gt_slice > 0
    cand = cand_slice

    rgb = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)

    # GT in green
    rgb[gt, 1] = 1.0
    rgb[gt, 0] *= 0.3
    rgb[gt, 2] *= 0.3

    # candidates in red
    rgb[cand, 0] = 1.0
    rgb[cand, 1] *= 0.3
    rgb[cand, 2] *= 0.3

    plt.figure(figsize=(4, 6))
    plt.imshow(rgb)
    plt.title(f"Slice {z} – GT (green), candidates (red)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# -------- Main pipeline --------
def main():
    print("Loading volumes...")
    raw_vol = tiff.imread(RAW_PATH)
    gt_vol = tiff.imread(GT_PATH)

    print("raw_vol shape:", raw_vol.shape, "dtype:", raw_vol.dtype)
    print("gt_vol  shape:", gt_vol.shape, "dtype:", gt_vol.dtype)

    # slices where GT has some HRF
    nz_slices = [z for z in range(gt_vol.shape[0]) if np.any(gt_vol[z] > 0)]
    print(f"Found {len(nz_slices)} slices with GT HRF:", nz_slices[:10], "...")

    hits = 0
    total = len(nz_slices)
    dice_scores = []

    debug_dir = OUT_DIR / "debug_panels"
    debug_dir.mkdir(exist_ok=True)
    debug_saved = 0
    max_debug = 6  # only save a few example panels

    for z in nz_slices:
        raw_slice = raw_vol[z]
        gt_slice = gt_vol[z] > 0

        sl_norm = normalize_slice(raw_slice)
        cand_mask = detect_candidates(sl_norm)

        overlap = np.logical_and(cand_mask, gt_slice)
        hit = bool(overlap.any())
        if hit:
            hits += 1

        d = dice(gt_slice, cand_mask)
        dice_scores.append(d)

        if debug_saved < max_debug:
            panel_path = debug_dir / f"slice{z}_gt_vs_candidates.png"
            make_debug_panel(z, raw_slice, gt_slice, cand_mask, panel_path)
            debug_saved += 1

        print(
            f"[Slice {z:3d}] "
            f"GT pixels={gt_slice.sum():4d}  "
            f"cand_pixels={cand_mask.sum():4d}  "
            f"hit={hit}  dice={d:.3f}"
        )

    recall = hits / total if total > 0 else 0.0
    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0

    print("\n========== HRF candidate detector summary ==========")
    print(f"Slices with GT HRF:     {total}")
    print(f"Slices with any hit:    {hits}")
    print(f"Coverage (recall):      {recall:.3f}")
    print(f"Mean Dice (GT vs cand): {mean_dice:.3f}")

    metrics_path = OUT_DIR / "hrf_candidate_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"slices_with_gt,{total}\n")
        f.write(f"slices_with_hit,{hits}\n")
        f.write(f"coverage_recall,{recall:.4f}\n")
        f.write(f"mean_dice,{mean_dice:.4f}\n")

    print("Saved metrics to:", metrics_path)


if __name__ == "__main__":
    main()