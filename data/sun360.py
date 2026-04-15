"""
SUN360 panoramic scene dataset loader.

HDF5 structure (from origin_code/SUN360/data/torchfeed/):
  Separate files per split:
    pixels_trn_torchfeed.h5
    pixels_val_torchfeed.h5
    pixels_tst_torchfeed.h5

  Each file contains:
    /target_viewgrid           uint8  (N, C, W_total, H_total)   — Torch7 layout (W before H)
                                      where H_total = n_elev * view_H, W_total = n_azim * view_W
                                      values in [0, 255]
    /gridshape                 float64 [n_elev, n_azim]  — e.g. [4, 8]
    /view_snapshape            float64 [view_H, view_W]  — e.g. [32, 32]
    /average_target_viewgrid   uint8  (C, W_total, H_total)      — per-dataset mean
    /labs                      float64 (N, 1)                     — category labels
    /pano_dims                 float64 [H, W]                     — full panorama dimensions

NOTE: Torch7 HDF5 writes tensors in column-major (C, W, H) order. When read by h5py
(row-major), the last two spatial dims are transposed. We correct this with a transpose.

After loading, viewgrids are returned as:
  (N, n_views, C, view_H, view_W)  float32 in [0, 1]
  where n_views = n_elev * n_azim
"""
import os
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


_SPLIT_FILENAMES = {
    "train": "pixels_trn_torchfeed.h5",
    "val":   "pixels_val_torchfeed.h5",
    "test":  "pixels_tst_torchfeed.h5",
}


class SUN360Dataset(Dataset):
    """
    Loads SUN360 panoramic viewgrids from the split HDF5 files.

    Each item is a viewgrid of shape (n_views, C, view_H, view_W)
    with float32 values in [0, 1].

    Default grid: 4 elevations × 8 azimuths = 32 views, each 3×32×32.
    """

    def __init__(self,
                 data_dir: str,
                 split: str = "train",
                 mean_subtract: bool = True):
        """
        Args:
            data_dir: directory containing the split .h5 files
                      e.g. 'origin_code/SUN360/data/minitorchfeed'
            split: 'train', 'val', or 'test'
            mean_subtract: subtract the per-dataset mean viewgrid
        """
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split
        self.mean_subtract = mean_subtract

        h5_path = os.path.join(data_dir, _SPLIT_FILENAMES[split])
        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f"SUN360 HDF5 not found: '{h5_path}'.\n"
                f"Expected files: {list(_SPLIT_FILENAMES.values())}\n"
                f"in directory: '{data_dir}'"
            )

        self._load(h5_path, mean_subtract)

    def _load(self, h5_path: str, mean_subtract: bool) -> None:
        with h5py.File(h5_path, "r") as f:
            # Read raw tiled panorama: (N, C, W_total, H_total) in Torch7 order
            raw = f["target_viewgrid"][:]                   # (N, C, W_total, H_total)
            gs  = f["gridshape"][:]                         # [n_elev, n_azim]
            vs  = f["view_snapshape"][:]                    # [view_H, view_W]

            n_elev   = int(gs[0])
            n_azim   = int(gs[1])
            view_H   = int(vs[0])
            view_W   = int(vs[1])
            self.n_elev   = n_elev
            self.n_azim   = n_azim
            self.n_views  = n_elev * n_azim
            self.view_H   = view_H
            self.view_W   = view_W

            # Correct Torch7 dimension order: (N, C, W_total, H_total) → (N, C, H_total, W_total)
            vg = raw.transpose(0, 1, 3, 2)                 # (N, C, H_total, W_total)

            N, C, H_total, W_total = vg.shape
            assert H_total == n_elev * view_H, \
                f"H_total={H_total} != n_elev({n_elev}) * view_H({view_H})"
            assert W_total == n_azim * view_W, \
                f"W_total={W_total} != n_azim({n_azim}) * view_W({view_W})"

            # Reshape tiled panorama → individual views
            # (N, C, n_elev, view_H, n_azim, view_W)
            vg = vg.reshape(N, C, n_elev, view_H, n_azim, view_W)
            # (N, n_elev, n_azim, C, view_H, view_W)
            vg = vg.transpose(0, 2, 4, 1, 3, 5)
            # (N, n_views, C, view_H, view_W)
            vg = vg.reshape(N, n_elev * n_azim, C, view_H, view_W)

            # Normalize to [0, 1]
            vg = vg.astype(np.float32) / 255.0

            # Optional mean subtraction
            if mean_subtract and "average_target_viewgrid" in f:
                avg_raw = f["average_target_viewgrid"][:]   # (C, W_total, H_total)
                avg = avg_raw.transpose(0, 2, 1)            # (C, H_total, W_total)
                avg = avg.reshape(C, n_elev, view_H, n_azim, view_W)
                avg = avg.transpose(1, 3, 0, 2, 4)          # (n_elev, n_azim, C, view_H, view_W)
                avg = avg.reshape(1, n_elev*n_azim, C, view_H, view_W).astype(np.float32) / 255.0
                self.mean_viewgrid = torch.from_numpy(avg.squeeze(0))
                vg -= avg
                vg = np.clip(vg, -1.0, 1.0)
            else:
                self.mean_viewgrid = None

        self.viewgrids = torch.from_numpy(vg)   # (N, n_views, C, view_H, view_W)

    def __len__(self) -> int:
        return len(self.viewgrids)

    def __getitem__(self, idx: int) -> Tensor:
        """Returns viewgrid of shape (n_views, C, view_H, view_W)."""
        return self.viewgrids[idx]
