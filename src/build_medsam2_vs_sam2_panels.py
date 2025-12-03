import os
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
cmp_dir = ROOT / "results" / "comparison"
cmp_dir.mkdir(parents=True, exist_ok=True)

slices = [150, 170, 175, 180]

for s in slices:
    med_path = cmp_dir / f"slice{s}_medsam2.png"
    sam_path = cmp_dir / f"slice{s}_sam2.png"

    if not med_path.exists() or not sam_path.exists():
        print(f"[WARN] Missing files for slice {s}")
        continue

    med_img = Image.open(med_path)
    sam_img = Image.open(sam_path)

    # Make them roughly the same size
    sam_img = sam_img.resize(med_img.size)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(med_img)
    axes[0].set_title(f"MedSAM2 – slice {s}")
    axes[0].axis("off")

    axes[1].imshow(sam_img)
    axes[1].set_title(f"SAM2 – slice {s}")
    axes[1].axis("off")

    out_path = cmp_dir / f"slice{s}_medsam2_vs_sam2.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved {out_path}")