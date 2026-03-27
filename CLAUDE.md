# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PyTorch 2026 reimplementation of "Learning to Look Around: Intelligently Exploring Unseen Environments for Unknown Tasks" (Jayaraman & Grauman, CVPR 2018 — arXiv:1709.00507). Reference Lua/Torch7 code is in `origin_code/`.

## Commands

```bash
# Install dependencies (Python 3.11, torch 2.3.1+cu121)
uv sync

# Train on indoor360 dataset (recommended, ~2 hours on cuda:2)
uv run python train.py --data-dir data/indoor360_torchfeed --wandb --device cuda:2

# Train on mini SUN360 (10 panoramas, fast smoke-test)
uv run python train.py --data-dir origin_code/SUN360/data/minitorchfeed --device cuda:2

# Evaluate a checkpoint
uv run python eval.py --checkpoint checkpoints/ckpt_epoch2000.pt \
    --data-dir data/indoor360_torchfeed --device cuda:2 --wandb

# Preprocess raw panoramas (JPEG/PNG dir) → torchfeed HDF5
uv run python data/prepare_sun360.py --pano-dir /path/to/images --out-dir data/mytorchfeed

# Policy transfer evaluation (requires categorized dataset with labels)
uv run python eval_transfer.py --checkpoint checkpoints/ckpt_epoch2000.pt \
    --data-dir data/indoor360_torchfeed
```

## Architecture

The agent runs T=6 step episodes. At each step:

1. **ViewEncoder** (`models/encoder.py`) — 3-layer CNN encodes the current 3×32×32 patch → 256D
2. **LocationSensor** (`models/location.py`) — MLP on `[abs_elev, d_elev_prev, d_azim_prev]` → 16D
3. **CombineModule** (`models/combine.py`) — concat(256D, 16D) → Linear → BN → Dropout → 256D
4. **AgentMemory** (`models/memory.py`) — single-layer LSTM (hidden=256) accumulates context
5. **CompletionHead** (`models/completion.py`) — transposed-conv decoder: 256D → full viewgrid (N_views×3×32×32)
6. **Actor** (`models/actor.py`) — MLP on `[h_t ‖ rel_pos ‖ t/T]` → 14 action logits

Training (`train.py`) is two-phase:
- **Phase 1** (50 epochs, T=1): reconstruction loss only, warms up encoder+decoder
- **Phase 2** (2000 epochs, T=6): reconstruction loss (sum over all T steps) + REINFORCE with EMA baseline (reward = −MSE at final step T)

Key implementation details:
- **Rotation compensation** (`data/utils.py:circ_shift_viewgrid`): since the agent has no absolute azimuth reference, the predicted viewgrid is circularly shifted by `delta_0` (first view's azimuth index) before computing MSE against ground truth.
- **Paste observed** (`data/utils.py:paste_observed`): after decoding, actually-observed views are pasted into the prediction — error only comes from unobserved views.
- **All batch elements share position** at each step (simplification over the paper): the single shared `(elev, azim)` position is driven by `batch[0]`'s action.

## Data

**HDF5 format** (`data/sun360.py`): Torch7 column-major quirk — raw data is `(N, C, W, H)`, must `.transpose(0,1,3,2)` on load to get `(N, C, H, W)`. The dataset returns `(N, N_views, C, 32, 32)` float32 in [0,1].

Grid: 4 elevations × 8 azimuths = 32 views, each 3×32×32. Action space: 14 = 3×5 neighborhood minus center (0,0).

**Available datasets:**
- `origin_code/SUN360/data/minitorchfeed/` — 10 panoramas per split (smoke test only)
- `data/indoor360_torchfeed/` — ShanghaiTech-Kujiale: 2591 train / 461 val / 498 test indoor panoramas
- Full SUN360: MIT CSAIL server unreachable as of 2026; use `data/prepare_sun360.py` if you obtain panoramas

## CUDA / Environment

System has 3 GPUs; use **cuda:2** (RTX 3090). GPUs 0/1 are GTX TITAN X (old, loaded).

torch==2.3.1+cu121 is locked in `pyproject.toml` — do not upgrade. Newer torch versions (2.5+) have NCCL/cuDNN library conflicts on this machine.

## Disk

Checkpoints are ~33 MB each. `config.py` has `save_every=100` (saves at epochs 100, 200, …, 2000 = 20 files ≈ 660 MB). Do not lower `save_every` — it previously filled the 2 TB disk.
