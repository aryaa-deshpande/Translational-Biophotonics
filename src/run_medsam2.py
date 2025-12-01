from pathlib import Path

import numpy as np
import tifffile as tiff


def load_and_prepare_oct():
    # repo root = one level above src/
    root = Path(__file__).resolve().parents[1]

    oct_path = root / "data" / "raw" / "raw_oct.tif"
    save_path = root / "data" / "raw" / "oct_case1_for_medsam2.npy"

    print(f"Loading OCT from: {oct_path}")
    vol = tiff.imread(oct_path)  # shape ~ (500, 900, 500)
    print("Original volume shape:", vol.shape)
    print("Original dtype:", vol.dtype)

    #   normalize to [0, 255] uint8, as MedSAM2 expects  
    vol = vol.astype(np.float32)

    # shift so min = 0
    vol -= vol.min()

    max_val = vol.max()
    if max_val > 0:
        vol /= max_val  # now in [0, 1]
    else:
        print("Warning: OCT volume is constant; normalization may be weird.")

    vol = (vol * 255).astype(np.uint8)

    print("Normalized volume shape:", vol.shape)
    print("Normalized dtype:", vol.dtype)

    #   save for MedSAM2 inference  
    np.save(save_path, vol)
    print(f"Saved normalized volume to: {save_path}")

    return vol


if __name__ == "__main__":
    load_and_prepare_oct()