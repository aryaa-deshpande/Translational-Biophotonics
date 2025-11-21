

# Translational Biophotonics – MedSAM2 HRF Segmentation

This repo contains my work for **Dr. Won’s Translational Biophotonics Lab**  
on segmenting **hyper-reflective foci (HRF)** in retinal OCT volumes.

I use **MedSAM2** to segment HRF on a 3D OCT scan, compute slice-wise and
volume-wise metrics, and generate “top-view” visualizations that Dr. Won likes
for comparing different models (SAM2 vs MedSAM2).

---

## 1. Repo structure

High-level layout:

```text
Translational-Biophotonics/
├── data/
│   ├── raw/         # OCT volume(s): raw_oct.tif, etc.
│   ├── masks/       # GT mask volume(s): mask_oct.tif (HRF GT)
│   └── sam2/        # (future) SAM2 outputs from Dev
├── MedSAM2/         # upstream MedSAM2 repo (cloned here, gitignored)
├── src/             # all my code lives here
│   ├── inspect_data.py
│   ├── medsam2_core.py
│   ├── medsam2_oneslice.py
│   ├── medsam2_all_slices.py
│   ├── build_topview.py
│   ├── build_overlay_volume.py
│   └── (debug scripts)
├── results/
│   ├── masks/       # predicted 2D masks per slice
│   ├── figures/     # per-slice overlays, quick plots
│   └── topview/     # 3D projections, volumes, CSVs
├── requirements.txt
└── README.md

Data (data/), MedSAM2 (MedSAM2/), and results/ are not tracked in git.
```
---

## 2. Setup

### 2.1. Python & environment

This project uses Python 3.10 (MedSAM2 requires ≥3.10).

### In repo root
```
python3.10 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```
### 2.2. Install MedSAM2 (editable)

Assumes the MedSAM2 repo is cloned inside this project as MedSAM2/.
```
cd MedSAM2
pip install -e ".[dev]"
cd ..
```
Note: you also need a working PyTorch with MPS / CUDA installed.
On my MacBook this runs on mps device.

---

## 3. Data layout

Place the OCT volume and GT mask here:
```
data/
  raw/
    raw_oct.tif        # 3D OCT volume, shape ~ (Z, H, W)
  masks/
    mask_oct.tif       # 3D GT HRF mask volume (same shape as OCT)
```
The scripts assume these exact filenames; change paths in ```src/inspect_data.py```
and ```src/medsam2_core.py``` if needed.

---

## 4. Scripts and what they do

### 4.1. Quick sanity check – OCT + GT mask
```
source venv/bin/activate
python src/inspect_data.py
```
This script:
```
	•	loads raw_oct.tif and mask_oct.tif
	•	prints shapes and dtypes
	•	finds slices that contain HRF
	•	saves a quick overlay figure in results/figures/
```
Use this to confirm the data is correctly aligned and non-empty.

---

### 4.2. MedSAM2 on a single slice
```
python src/medsam2_oneslice.py
```
This script:
```
	•	loads the OCT and GT mask volumes
	•	finds slices that contain HRF
	•	runs MedSAM2 on one HRF slice
	•	computes Dice, FP, FN, overlap ratio
	•	saves:
	•	predicted mask      → results/masks/oct_sliceXXX_medsam2.tif
	•	structural overlay  → results/figures/oct_sliceXXX_medsam2_overlay.png
```

Good for debugging model + prompting on a single slice.

---

### 4.3. MedSAM2 on all HRF slices (multi-slice)
```
python src/medsam2_all_slices.py
```
This script:
```
	•	loops over all slices that have HRF in mask_oct.tif
	•	computes:
	•	per-slice Dice
	•	FP / FN counts
	•	overlap ratios
	•	saves:
	•	per-slice overlays in results/figures/
	•	per-slice predicted masks in results/masks/
	•	a summary CSV: results/medsam2_slice_metrics.csv
```
This is the main quantitative evaluation of MedSAM2.

---

### 4.4. Build 3D mask volume + topview projections
```
python src/build_topview.py
```
This script:
```
	•	stacks all 2D predicted masks into a 3D volume
	•	computes 3D Dice between MedSAM2 and GT
	•	generates several “top view” projections:
```
Saved outputs (under results/topview/):
```
	•	medsam2_pred_volume.tif          – full 3D predicted HRF mask
	•	gt_topview.png                    – topview of GT mask
	•	medsam2_topview.png               – topview of MedSAM2 mask
	•	medsam2_frequency_topview.png     – per-pixel frequency map along depth
	•	gt_vs_medsam2_topview_overlay.png – GT vs MedSAM2 overlap
	•	medsam2_topview_metrics.csv       – 3D metrics summary
```
These figures are what Dr. Won calls the “top view”.

---

### 4.5. Build structural overlay volume (for Fiji)
```
python src/build_overlay_volume.py
```
This script:
```
	•	loads the raw OCT volume and the 3D MedSAM2 mask
	•	for each slice:
	•	normalizes the OCT slice to [0, 255]
	•	converts to RGB
	•	paints MedSAM2 HRF pixels in red
	•	stacks all slices into an RGB volume
```
Output:
```
	•	results/topview/medsam2_structural_overlay_volume.tif
```
Open this in Fiji and scroll through slices to see MedSAM2’s HRF
segmentation on top of the actual retinal structure.

---

## 5. Typical workflow (end-to-end)
1.	Inspect data
```
python src/inspect_data.py
```

2.	Sanity-check MedSAM2 on one slice
```
python src/medsam2_oneslice.py
```

3.	Run MedSAM2 on all HRF slices
```
python src/medsam2_all_slices.py
```

4.	Build 3D volume + topviews
```
python src/build_topview.py
```

5.	Build structural overlay volume for Fiji
```
python src/build_overlay_volume.py
```


Then use the CSVs + figures in ```results/``` for analysis and slides.

---

## 6. Next steps (planned)
```
	•	Add support for SAM2 segmentations under data/sam2/.
	•	Build sam2_compare.py:
	•	load SAM2 masks
	•	compute same slice/3D metrics
	•	generate SAM2 topviews
	•	Final comparison:
	•	MedSAM2 vs SAM2 vs GT
	•	tables + visual examples for paper / presentation.
```