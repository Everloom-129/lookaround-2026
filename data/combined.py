"""
Combined dataset loader for mixed indoor360 + SUN360 training.

indoor360 is re-split 4:3:3 from the concatenation of all three original splits.
SUN360 uses its existing train/val splits as-is.

Usage:
    train_loader, val_loader = make_combined_loaders(
        indoor_dir="data/indoor360_torchfeed",
        sun360_dir="data/sun360_torchfeed",
        batch_size=32,
        seed=42,
    )
"""
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset

from data.sun360 import SUN360Dataset


def resplit_indoor360(data_dir: str, ratio=(0.4, 0.3, 0.3), seed: int = 42):
    """
    Load all three indoor360 splits, concatenate, and re-split with given ratio.

    Returns:
        train_ds, val_ds, test_ds  — TensorDatasets of (N, n_views, C, H, W)
    """
    assert abs(sum(ratio) - 1.0) < 1e-6, "ratio must sum to 1"

    parts = []
    for split in ("train", "val", "test"):
        ds = SUN360Dataset(data_dir, split=split)
        parts.append(ds.viewgrids)  # (N, n_views, C, H, W)

    all_viewgrids = torch.cat(parts, dim=0)  # (N_total, ...)
    N = len(all_viewgrids)

    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(N, generator=rng)

    n_train = int(N * ratio[0])
    n_val   = int(N * ratio[1])
    # test gets remainder so counts add up exactly
    idx_train = perm[:n_train]
    idx_val   = perm[n_train:n_train + n_val]
    idx_test  = perm[n_train + n_val:]

    train_ds = TensorDataset(all_viewgrids[idx_train])
    val_ds   = TensorDataset(all_viewgrids[idx_val])
    test_ds  = TensorDataset(all_viewgrids[idx_test])

    print(f"[indoor360 resplit 4:3:3] total={N}  "
          f"train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")
    return train_ds, val_ds, test_ds


class _UnwrapTensor(torch.utils.data.Dataset):
    """TensorDataset wraps items in a tuple; this unwraps the first element."""
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx][0]


def make_combined_loaders(
    indoor_dir: str,
    sun360_dir: str,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 2,
):
    """
    Build combined train and val DataLoaders.

    Train = indoor360 train (4/10) + sun360 train
    Val   = indoor360 val   (3/10) + sun360 val

    Returns:
        train_loader, val_loader, n_elev, n_azim
    """
    indoor_train_ds, indoor_val_ds, _ = resplit_indoor360(
        indoor_dir, ratio=(0.4, 0.3, 0.3), seed=seed
    )
    sun360_train_ds = SUN360Dataset(sun360_dir, split="train")
    sun360_val_ds   = SUN360Dataset(sun360_dir, split="val")

    print(f"[sun360] train={len(sun360_train_ds)}  val={len(sun360_val_ds)}")

    # Unwrap TensorDataset tuples so all datasets return a plain tensor
    combined_train = ConcatDataset([_UnwrapTensor(indoor_train_ds), sun360_train_ds])
    combined_val   = ConcatDataset([_UnwrapTensor(indoor_val_ds),   sun360_val_ds])

    print(f"[combined] train={len(combined_train)}  val={len(combined_val)}")

    train_loader = DataLoader(
        combined_train, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        combined_val, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False,
    )

    # Grid shape — same for both datasets (4×8)
    n_elev = sun360_train_ds.n_elev
    n_azim = sun360_train_ds.n_azim

    return train_loader, val_loader, n_elev, n_azim
