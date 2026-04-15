"""
Logging utilities: optional wandb integration + console logging.
"""
import os
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor


def init_logging(config, run_name: Optional[str] = None) -> Any:
    """
    Initialize wandb run if config.use_wandb is True.

    Returns:
        wandb run object, or None if not using wandb.
    """
    if not config.use_wandb:
        return None
    try:
        import wandb
        run = wandb.init(
            project="lookaround-2026",
            name=run_name,
            config=vars(config),
        )
        return run
    except ImportError:
        print("[logging] wandb not installed, skipping.")
        return None


def log_metrics(metrics: Dict[str, float], step: int,
                run: Any = None) -> None:
    """
    Log metrics to wandb (if available) and print to stdout.
    """
    msg = f"[step {step:6d}] " + "  ".join(
        f"{k}={v:.4f}" for k, v in metrics.items()
    )
    print(msg)
    if run is not None:
        run.log(metrics, step=step)


def log_val_recon(
    run: Any,
    step: int,
    recon_list: List[Tensor],
    target: Tensor,
    n_elev: int,
    n_azim: int,
    mean_vg: Optional[Tensor] = None,
) -> None:
    """
    Log val reconstruction images to wandb for a single sample.

    Logs three panels every call:
      val/gt           — ground truth viewgrid (static reference)
      val/recon_final  — predicted viewgrid at step T (tracks improvement over epochs)
      val/recon_steps  — all T steps stacked vertically (shows within-episode progression)

    Args:
        recon_list: list of T tensors (B, N_views, C, H, W), uses first sample (index 0)
        target:     (B, N_views, C, H, W) — mean-subtracted
        mean_vg:    (N_views, C, H, W) — dataset mean; added back before display so
                    pixel values are in [0, 1] and colors look correct
    """
    if run is None:
        return
    try:
        import wandb
        import numpy as np
    except ImportError:
        return

    def _to_uint8_grid(vg: Tensor) -> "np.ndarray":
        """
        Convert a single viewgrid (N_views, C, H, W) to a tiled uint8 RGB image.
        Mean is added back and values clamped to [0, 1] before conversion.
        Layout: n_elev rows × n_azim cols of H×W patches.
        """
        if mean_vg is not None:
            vg = vg + mean_vg.to(vg.device)
        vg = vg.clamp(0.0, 1.0).cpu().float().numpy()   # (N_views, C, H, W)
        _, C, H, W = vg.shape
        vg = vg.reshape(n_elev, n_azim, C, H, W)
        # → (n_elev, H, n_azim, W, C)
        vg = vg.transpose(0, 3, 1, 4, 2)
        vg = vg.reshape(n_elev * H, n_azim * W, C)
        return (vg * 255).astype(np.uint8)

    # First sample only
    gt_vg   = target[0]                   # (N_views, C, H, W)
    gt_grid = _to_uint8_grid(gt_vg)       # (n_elev*H, n_azim*W, 3)

    final_grid = _to_uint8_grid(recon_list[-1][0])

    # All T steps stacked vertically with a 2-px white separator
    T = len(recon_list)
    H_grid = gt_grid.shape[0]
    W_grid = gt_grid.shape[1]
    sep = np.full((2, W_grid, 3), 255, dtype=np.uint8)

    rows = []
    for t in range(T):
        rows.append(_to_uint8_grid(recon_list[t][0]))
        if t < T - 1:
            rows.append(sep)
    steps_panel = np.concatenate(rows, axis=0)   # (T*H + (T-1)*2, W, 3)

    run.log({
        "val/gt":           wandb.Image(gt_grid,    caption="ground truth"),
        "val/recon_final":  wandb.Image(final_grid, caption=f"pred step T={T}"),
        "val/recon_steps":  wandb.Image(steps_panel, caption=f"steps 1–{T} (top→bottom)"),
    }, step=step)
