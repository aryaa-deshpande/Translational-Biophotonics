# src/medsam2_all_slices.py

import csv
from medsam2_core import (
    RESULTS_DIR,
    load_full_volumes,
    build_medsam2_model,
    process_single_slice,
)

def run_all_hrf_slices():
    vol, mask_vol, hrf_slices = load_full_volumes()
    predictor = build_medsam2_model()

    all_metrics = []
    for idx in hrf_slices:
        print("\n==============================")
        print(f"Processing slice {idx}")
        print("==============================")
        m = process_single_slice(idx, vol, mask_vol, predictor, RESULTS_DIR, save_fig=True)
        if m is not None:
            all_metrics.append(m)

    if not all_metrics:
        print("No slices were processed (no HRF found?).")
        return

    dices = [m["dice"] for m in all_metrics]
    avg_dice = sum(dices) / len(dices)
    print("\n SUMMARY: ")
    print(f"Processed {len(all_metrics)} HRF slices")
    print(f"Mean Dice: {avg_dice:.4f}")
    print("Per-slice Dice:")
    for m in all_metrics:
        print(f" - slice {m['slice_idx']}: {m['dice']:.4f}")

    metrics_path = RESULTS_DIR / "medsam2_slice_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slice_idx", "dice", "gt_pixels", "pred_pixels",
                "intersection", "false_positive", "false_negative", "overlap_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(all_metrics)
    print("Saved metrics table to:", metrics_path)

if __name__ == "__main__":
    run_all_hrf_slices()