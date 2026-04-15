"""
Preprocess raw SUN360 equirectangular panoramas → torchfeed HDF5 format.

Usage
-----
# From a flat directory of panoramas (JPEG/PNG):
  uv run python data/prepare_sun360.py \\
    --pano-dir /path/to/SUN360/panoramas \\
    --out-dir   origin_code/SUN360/data/torchfeed \\
    --seed 62346

# From the Refer360 archive (tar.gz with sub-directories per category):
  uv run python data/prepare_sun360.py \\
    --pano-dir /path/to/refer360images \\
    --out-dir   origin_code/SUN360/data/torchfeed \\
    --seed 62346

# Preview mode (no files written):
  uv run python data/prepare_sun360.py --pano-dir /path --dry-run

Input format
------------
Any equirectangular panorama images (JPEG/PNG) at any resolution.
Sub-directories are interpreted as category names (used as class labels).

Output format
-------------
Split HDF5 files matching the torchfeed schema:
  pixels_trn_torchfeed.h5 / pixels_val_torchfeed.h5 / pixels_tst_torchfeed.h5

Each file contains:
  target_viewgrid        uint8  (N, 3, 256, 128)  — Torch7 (W before H) layout
  gridshape              float64 [4, 8]
  view_snapshape         float64 [32, 32]
  pano_dims              float64 [128, 256]
  average_target_viewgrid uint8  (3, 256, 128)
  labs                   float64 (N, 1)
  shuffle_ord            float64 (N_total,)

Data download
-------------
The SUN360 panoramas are no longer available from the original MIT CSAIL server
(sun360.csail.mit.edu — returns no response as of 2026).

Alternative sources:
  1. Refer360 project (SUN360 subset, ~67k panoramas):
     https://github.com/volkancirik/refer360 → fill Google form to receive link
     Archive: refer360images.tar.gz  (organized by category)

  2. Places365 panoramas (partially overlapping):
     http://places2.csail.mit.edu/

  3. If/when MIT CSAIL restores sun360.csail.mit.edu/Images/, panoramas are at:
     http://sun360.csail.mit.edu/Images/<category>/<filename>.jpg

Split sizes (from original paper):
  train: ~7186   val: ~1265   test: ~1355
"""
import argparse
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image

# Grid parameters matching the torchfeed format
N_ELEV = 4
N_AZIM = 8
VIEW_H = 32
VIEW_W = 32
PANO_H = N_ELEV * VIEW_H   # 128
PANO_W = N_AZIM * VIEW_W   # 256

# Train/val/test split fractions
SPLIT_FRAC = {"train": 0.73, "val": 0.13, "test": 0.14}

_SPLIT_FILENAMES = {
    "train": "pixels_trn_torchfeed.h5",
    "val":   "pixels_val_torchfeed.h5",
    "test":  "pixels_tst_torchfeed.h5",
}


def find_panoramas(pano_dir: str) -> List[Tuple[str, int]]:
    """
    Scan pano_dir for image files.
    Returns list of (filepath, label_int) pairs.
    Sub-directories are treated as categories; top-level images get label 0.
    """
    pano_dir = Path(pano_dir)
    ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Collect (path, category_name) pairs
    pairs: List[Tuple[Path, str]] = []
    categories: Dict[str, List[Path]] = defaultdict(list)

    for item in sorted(pano_dir.rglob("*")):
        if item.suffix.lower() in ext:
            cat = item.parent.name if item.parent != pano_dir else "_root"
            categories[cat].append(item)

    # Build label map
    cat_names = sorted(categories.keys())
    cat2int = {c: i for i, c in enumerate(cat_names)}
    print(f"Found {len(cat_names)} categories:")
    for cat in cat_names[:20]:
        print(f"  {cat}: {len(categories[cat])} panoramas")
    if len(cat_names) > 20:
        print(f"  ... and {len(cat_names) - 20} more")

    for cat, paths in categories.items():
        for p in paths:
            pairs.append((p, cat2int[cat]))

    print(f"Total panoramas: {len(pairs)}")
    return [(str(p), label) for p, label in pairs]


def process_panorama(img_path: str) -> Optional[np.ndarray]:
    """
    Load and resize a panorama to (PANO_H, PANO_W, 3).
    Returns uint8 array or None on error.
    """
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((PANO_W, PANO_H), Image.LANCZOS)
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"  Warning: skipping {img_path}: {e}")
        return None


def pano_to_viewgrid_torch7(pano: np.ndarray) -> np.ndarray:
    """
    Convert (PANO_H, PANO_W, 3) uint8 panorama to Torch7-style viewgrid.
    Output: (3, PANO_W, PANO_H) uint8 — channels-first, W before H.
    """
    # (H, W, C) → (C, H, W)
    chw = pano.transpose(2, 0, 1)
    # Torch7 stores (C, W, H) — swap last two dims
    return chw.transpose(0, 2, 1)   # (C, W, H) = (3, 256, 128)


def split_data(pairs: List[Tuple[str, int]],
               seed: int) -> Dict[str, List[Tuple[str, int]]]:
    """Split panoramas into train/val/test."""
    rng = random.Random(seed)
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
    for name, items in splits.items():
        print(f"  {name}: {len(items)} panoramas")
    return splits


def write_hdf5(out_path: str,
               items: List[Tuple[str, int]],
               all_indices: List[int]) -> None:
    """
    Write one split HDF5 file.

    all_indices: global indices of each item in the full (shuffled) list,
                 stored in shuffle_ord to match the original schema.
    """
    N = len(items)
    viewgrids = np.zeros((N, 3, PANO_W, PANO_H), dtype=np.uint8)
    labs = np.zeros((N, 1), dtype=np.float64)
    valid_mask = np.ones(N, dtype=bool)

    print(f"  Processing {N} panoramas → {out_path}")
    for i, (path, label) in enumerate(items):
        if i % 500 == 0:
            print(f"    [{i}/{N}]")
        pano = process_panorama(path)
        if pano is None:
            valid_mask[i] = False
            continue
        viewgrids[i] = pano_to_viewgrid_torch7(pano)
        labs[i, 0] = float(label)

    # Filter out failed panoramas
    viewgrids = viewgrids[valid_mask]
    labs = labs[valid_mask]

    # Compute average viewgrid
    avg_vg = viewgrids.mean(axis=0).astype(np.uint8)   # (3, PANO_W, PANO_H)

    shuffle_ord = np.array([all_indices[i] for i in range(N) if valid_mask[i]],
                           dtype=np.float64)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("target_viewgrid",          data=viewgrids)
        f.create_dataset("average_target_viewgrid",  data=avg_vg)
        f.create_dataset("labs",                     data=labs)
        f.create_dataset("shuffle_ord",              data=shuffle_ord)
        f.create_dataset("gridshape",    data=np.array([N_ELEV, N_AZIM], dtype=np.float64))
        f.create_dataset("view_snapshape", data=np.array([VIEW_H, VIEW_W], dtype=np.float64))
        f.create_dataset("pano_dims",    data=np.array([PANO_H, PANO_W], dtype=np.float64))

    print(f"    Saved {viewgrids.shape[0]} panoramas to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess SUN360 panoramas → HDF5")
    parser.add_argument("--pano-dir", required=True,
                        help="Directory containing panorama images (JPEGs/PNGs)")
    parser.add_argument("--out-dir",  default="origin_code/SUN360/data/torchfeed",
                        help="Output directory for HDF5 files")
    parser.add_argument("--seed",     type=int, default=62346,
                        help="Random seed for train/val/test split")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Scan and report without writing files")
    args = parser.parse_args()

    print(f"Scanning panoramas in: {args.pano_dir}")
    pairs = find_panoramas(args.pano_dir)

    if args.dry_run:
        print("Dry run — no files written.")
        return

    print("\nSplitting data...")
    splits = split_data(pairs, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # Compute global indices for shuffle_ord
    # (matches original: global index in full shuffled list)
    rng = random.Random(args.seed)
    all_shuffled = list(range(len(pairs)))
    rng.shuffle(all_shuffled)

    n_train = len(splits["train"])
    n_val   = len(splits["val"])

    split_indices = {
        "train": all_shuffled[:n_train],
        "val":   all_shuffled[n_train:n_train + n_val],
        "test":  all_shuffled[n_train + n_val:],
    }

    print("\nWriting HDF5 files...")
    for split_name in ("train", "val", "test"):
        out_path = os.path.join(args.out_dir, _SPLIT_FILENAMES[split_name])
        write_hdf5(out_path, splits[split_name], split_indices[split_name])

    print("\nDone! Files written to:", args.out_dir)
    print("To train with full data:")
    print(f"  uv run python train.py --data-dir {args.out_dir}")


if __name__ == "__main__":
    main()
