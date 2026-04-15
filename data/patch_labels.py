"""
One-off script: patch the labs field in indoor360_torchfeed HDF5 files
to assign _empty=0, _full=1 based on filename, using the same deterministic
shuffle+split as prepare_sun360.py (seed=62346, fracs=0.73/0.13/0.14).

Run once:
  uv run python data/patch_labels.py
"""
import random
from pathlib import Path

import h5py
import numpy as np

PANO_DIR = Path("data/indoor360_raw/images")
OUT_DIR  = Path("data/indoor360_torchfeed")
SEED     = 62346
SPLIT_FRAC = {"train": 0.73, "val": 0.13}  # test is remainder

FILENAMES = {
    "train": "pixels_trn_torchfeed.h5",
    "val":   "pixels_val_torchfeed.h5",
    "test":  "pixels_tst_torchfeed.h5",
}

# --- Reproduce the same file scan as prepare_sun360.py ---
ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
files = sorted(p for p in PANO_DIR.rglob("*") if p.suffix.lower() in ext)
print(f"Found {len(files)} images")

def get_label(p: Path) -> int:
    stem = p.stem.lower()
    if stem.endswith("_empty"):
        return 0
    elif stem.endswith("_full"):
        return 1
    return 0

pairs = [(str(p), get_label(p)) for p in files]

# --- Reproduce the same shuffle+split ---
rng = random.Random(SEED)
shuffled = list(pairs)
rng.shuffle(shuffled)

N = len(shuffled)
n_train = int(N * SPLIT_FRAC["train"])
n_val   = int(N * SPLIT_FRAC["val"])

splits = {
    "train": shuffled[:n_train],
    "val":   shuffled[n_train:n_train + n_val],
    "test":  shuffled[n_train + n_val:],
}

# --- Patch each HDF5 ---
for split_name, items in splits.items():
    h5_path = OUT_DIR / FILENAMES[split_name]
    if not h5_path.exists():
        print(f"Skipping {h5_path} (not found)")
        continue

    with h5py.File(h5_path, "r") as f:
        N_h5 = f["target_viewgrid"].shape[0]

    if N_h5 != len(items):
        print(f"WARNING: {h5_path} has {N_h5} entries but expected {len(items)}")

    labels = np.array([[float(lbl)] for _, lbl in items[:N_h5]], dtype=np.float64)
    n_empty = int((labels == 0).sum())
    n_full  = int((labels == 1).sum())

    with h5py.File(h5_path, "a") as f:
        del f["labs"]
        f.create_dataset("labs", data=labels)

    print(f"{h5_path.name}: {N_h5} entries  empty={n_empty}  full={n_full}")

print("Done.")
