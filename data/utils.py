"""
Viewgrid utilities: view access, rotation compensation, observed-view pasting.
"""
from typing import Dict, Tuple

import torch
from torch import Tensor


def get_view(viewgrid: Tensor, elev_idx: int, azim_idx: int,
             n_azim: int = 8) -> Tensor:
    """
    Extract a single view from a batched viewgrid.

    Args:
        viewgrid: (B, N_views, C, H, W)
        elev_idx: elevation index [0, n_elev)
        azim_idx: azimuth index — will be wrapped modulo n_azim
        n_azim: number of azimuth positions

    Returns:
        (B, C, H, W)
    """
    azim_idx = azim_idx % n_azim
    flat_idx = elev_idx * n_azim + azim_idx
    return viewgrid[:, flat_idx]  # (B, C, H, W)


def circ_shift_viewgrid(recon: Tensor, delta_0: int,
                         n_elev: int = 5, n_azim: int = 8) -> Tensor:
    """
    Circularly shift the predicted viewgrid along the azimuth axis.

    The agent treats the first view's azimuth as origin. To compare against
    the ground-truth viewgrid, shift the prediction by delta_0 positions.

    Args:
        recon: (B, N_views, C, H, W)
        delta_0: first view's azimuth index (integer shift)
        n_elev, n_azim: grid dimensions

    Returns:
        (B, N_views, C, H, W) — shifted
    """
    B, N, C, H, W = recon.shape
    recon = recon.view(B, n_elev, n_azim, C, H, W)
    recon = torch.roll(recon, shifts=delta_0, dims=2)
    return recon.view(B, N, C, H, W)


def paste_observed(recon: Tensor,
                   observed: Dict[Tuple[int, int], Tensor],
                   n_azim: int = 8) -> Tensor:
    """
    Paste actually observed views into the predicted viewgrid (after circ_shift).

    After circ_shift_viewgrid(delta_0), position j in recon corresponds to
    absolute azimuth j.  The observed dict keys are absolute (elev, azim) indices,
    so each view pastes directly at its absolute index — no additional shift needed.

    Args:
        recon: (B, N_views, C, H, W) — already shifted by delta_0
        observed: dict from (elev_idx, azim_idx) → (B, C, H, W), absolute indices
        n_azim: number of azimuth positions

    Returns:
        (B, N_views, C, H, W) — with observed views pasted in
    """
    recon = recon.clone()
    for (elev_idx, azim_idx), view in observed.items():
        flat_idx = elev_idx * n_azim + (azim_idx % n_azim)
        recon[:, flat_idx] = view
    return recon


def step_position(elev: int, azim: int,
                  d_elev: int, d_azim: int,
                  n_elev: int = 5, n_azim: int = 8) -> Tuple[int, int]:
    """Apply an action (d_elev, d_azim) to current position with boundary handling."""
    new_elev = max(0, min(n_elev - 1, elev + d_elev))
    new_azim = (azim + d_azim) % n_azim
    return new_elev, new_azim
