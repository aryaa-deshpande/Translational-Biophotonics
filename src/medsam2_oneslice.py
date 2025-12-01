

from medsam2_core import (
    ROOT,
    RESULTS_DIR,
    load_full_volumes,
    build_medsam2_model,
    process_single_slice,
)

def run_one_slice(slice_idx=None):
    vol, mask_vol, hrf_slices = load_full_volumes()
    if slice_idx is None:
        # pick the first HRF slice if not specified
        slice_idx = hrf_slices[0]
    print(f"Using slice index with HRF: {slice_idx}")

    predictor = build_medsam2_model()
    metrics = process_single_slice(slice_idx, vol, mask_vol, predictor, RESULTS_DIR, save_fig=True)

    print("\nFinal metrics for slice", slice_idx)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    run_one_slice()