"""
usage:
  cd SITCOM/
  python preprocess_aapm_ndct.py \
      --fd_path  "/egr/research-slim/shared/AAPM_Low_Dose_Train/Training_Image_Data/1mm B30" \
      --case_num L067 \ # different case number for different dataset
      --device   cuda:0
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ── physical constants ───────────────────────────────────────────────────────
MU_WATER = 0.0192

# ── SITCOM project root ───────────────────────────────────────────────────────
SITCOM_ROOT = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════════════
# CLI arguments
# ═══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="AAPM DICOM → NDCT PNG (for SITCOM)")

parser.add_argument('--fd_path',
    type=str,
    default='/egr/research-slim/shared/AAPM_Low_Dose_Train/Training_Image_Data/1mm B30',
    help='Full-dose DICOM root directory (containing case_num subdirectories)')
parser.add_argument('--out_root',
    type=str,
    default=str(SITCOM_ROOT / 'data' / 'ldct_demo'),
    help='PNG output root directory, default SITCOM/data/ldct_demo/')
parser.add_argument('--case_num',     type=str,   default='L067')
parser.add_argument('--window_level', type=float, default=40.0,
    help='HU window level WL, 40 for abdominal soft tissues, -600 for lungs')
parser.add_argument('--window_width', type=float, default=400.0,
    help='HU window width WW, 400 for abdominal soft tissues, 1500 for lungs')
parser.add_argument('--axial_skip',   type=int,   default=1,
    help='Take every N axial slices (1=all, 8=fast test)')

args = parser.parse_args()

# ── output directory (only create ndct/) ───────────────────────────────────────
ndct_dir = Path(args.out_root) / 'ndct'
ndct_dir.mkdir(parents=True, exist_ok=True)
print(f"[Config] case={args.case_num}")
print(f"[Config] NDCT → {ndct_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1. DICOM → HU → μ volume → interpolate to 256×256×256
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/3] Loading DICOM ...")
files = sorted(glob.glob(os.path.join(args.fd_path, args.case_num, "*.IMA")))
if len(files) == 0:
    raise RuntimeError(f"No .IMA files in: {os.path.join(args.fd_path, args.case_num)}")
print(f"      Found {len(files)} slices.")

vol = []
for f in tqdm(files, desc="  Reading"):
    ds    = pydicom.dcmread(f)
    slope = float(ds.RescaleSlope)     if hasattr(ds, 'RescaleSlope')     else 1.0
    inter = float(ds.RescaleIntercept) if hasattr(ds, 'RescaleIntercept') else 0.0
    hu    = ds.pixel_array.astype(np.float32) * slope + inter
    vol.append(hu)

vol_hu = np.stack(vol, axis=0)                           # [Z, H, W]
vol_mu = np.clip((vol_hu / 1000.0 + 1.0) * MU_WATER, 0, None)

fd_mu_t = torch.from_numpy(vol_mu).float()
fd_mu_t = F.interpolate(
    fd_mu_t.unsqueeze(0).unsqueeze(0),                   # [1,1,Z,H,W]
    size=(256, 256, 256),
    mode='trilinear', align_corners=False
).squeeze(1)                                             # [1,256,256,256]

mu_min   = fd_mu_t.min().item()
mu_max   = fd_mu_t.max().item()
mu_scale = max(mu_max - mu_min, 1e-8)
print(f"      μ range: [{mu_min:.5f}, {mu_max:.5f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2. Normalization
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/3] Normalizing ...")
x_gt_norm = ((fd_mu_t - mu_min) / (mu_scale + 1e-8)).clamp(0, 1)  # [1,256,256,256]


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3. Slices → HU windowing → uint8 PNG
# ═══════════════════════════════════════════════════════════════════════════════
def mu_norm_to_uint8(norm_slice: np.ndarray,
                     mu_min: float, mu_scale: float,
                     wl: float, ww: float) -> np.ndarray:
    """Normalized μ slices [0,1] → HU → windowing → uint8"""
    mu  = norm_slice * mu_scale + mu_min
    hu  = (mu / MU_WATER - 1.0) * 1000.0
    lo  = wl - ww / 2.0
    hi  = wl + ww / 2.0
    return ((np.clip(hu, lo, hi) - lo) / (hi - lo) * 255.0).astype(np.uint8)


print(f"\n[3/3] Saving PNG slices (axial_skip={args.axial_skip}) ...")
n_slices = x_gt_norm.shape[1]
saved = 0

for z in tqdm(range(0, n_slices, args.axial_skip), desc="  Saving"):
    ndct_arr = mu_norm_to_uint8(
        x_gt_norm[0, z].cpu().numpy(),
        mu_min, mu_scale,
        args.window_level, args.window_width
    )
    Image.fromarray(ndct_arr, mode="L").save(ndct_dir / f"slice_{saved:04d}.png")
    saved += 1

print(f"\n✓  Saved {saved} NDCT PNG slices → {ndct_dir}")
print(f"\n>> Next: open ldct_sitcom_fixed.ipynb")
print(f"   LDCT will be simulated automatically (USE_EXISTING_LDCT=False)")
