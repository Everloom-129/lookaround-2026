# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PyTorch 2026 reimplementation of "Learning to Look Around: Intelligently Exploring Unseen Environments for Unknown Tasks" (Jayaraman & Grauman, CVPR 2018 — arXiv:1709.00507). Reference Lua/Torch7 code is in `origin_code/`.

## Commands

```bash
# Install dependencies (Python 3.11, torch 2.3.1+cu121)
uv sync

# Train on SUN360 dataset — preferred, use the script (sets run name from hparams)
bash train_sun360_only.sh

# Or manually on SUN360 (wandb run name auto-generated):
uv run python train.py --data-dir data/sun360_torchfeed --wandb --device cuda:2

# Train on indoor360 (alternative dataset, same HDF5 format):
uv run python train.py --data-dir data/indoor360_torchfeed --wandb --device cuda:2

# Train on mini SUN360 (10 panoramas, fast smoke-test)
uv run python train.py --data-dir origin_code/SUN360/data/minitorchfeed --device cuda:2

# Evaluate a checkpoint
uv run python eval.py --checkpoint checkpoints/ckpt_epoch2000.pt \
    --data-dir data/indoor360_torchfeed --device cuda:2 --wandb

# Preprocess raw panoramas (JPEG/PNG dir) → torchfeed HDF5
uv run python data/prepare_sun360.py --pano-dir /path/to/images --out-dir data/mytorchfeed

# Policy transfer evaluation — produces results/transfer_panocontext.json
#                              and results/transfer_modelnet10.json
uv run python eval_transfer.py --checkpoint checkpoints/ckpt_epoch2000.pt \
    --data-dir data/panocontext_torchfeed
uv run python eval_transfer.py --checkpoint checkpoints/ckpt_epoch2000.pt \
    --data-dir data/modelnet10_torchfeed

# Plot Fig 5 (requires both JSON files above)
uv run python plot_fig5.py

# Interactive visualization app (Gradio, port 7860)
uv run python app.py --device cuda:2 --port 7860
```

## Architecture

The agent runs T=6 step episodes. At each step:

1. **ViewEncoder** (`models/encoder.py`) — 3-layer CNN encodes the current 3×32×32 patch → 256D
2. **LocationSensor** (`models/location.py`) — MLP on `[rel_elev, rel_azim, t/T, abs_elev]` → 16D (4 inputs)
3. **CombineModule** (`models/combine.py`) — concat(256D, 16D) → Linear → BN → Dropout → 256D
4. **AgentMemory** (`models/memory.py`) — single-layer LSTM (hidden=256) accumulates context
5. **CompletionHead** (`models/completion.py`) — transposed-conv decoder: 256D → full viewgrid (N_views×3×32×32), final activation LeakyReLU(0.8)
6. **Actor** (`models/actor.py`) — MLP on `[h_t(256) ‖ rel_pos(2) ‖ t/T(1) ‖ abs_elev(1)]` → 14 action logits (260D input)

Training (`train.py`) is two-phase:
- **Phase 1** (50 epochs, T=1): reconstruction loss only, warms up encoder+decoder. All modules trained with Adam + weight_decay=0.005.
- **Phase 2** (2000 epochs, T=6): encoder/decoder/memory/location/combine are **frozen**. Only Actor and a learned scalar baseline are updated. Actor uses REINFORCE with per-sample reward (B,); baseline is `nn.Parameter` trained at 150× actor lr via MSE against reward. LR scheduler: ReduceLROnPlateau(patience=200, factor=0.5).

Key implementation details:
- **Mean subtraction** (`data/sun360.py`): targets are mean-subtracted (`mean_subtract=True` by default); the dataset mean viewgrid is stored in `dataset.mean_viewgrid`. CompletionHead uses LeakyReLU(0.8) (not Sigmoid) so it can produce negative outputs to match centered targets.
- **Rotation compensation** (`data/utils.py:circ_shift_viewgrid`): since the agent has no absolute azimuth reference, the predicted viewgrid is circularly shifted by `delta_0` (first view's azimuth index) before computing MSE against ground truth.
- **Paste observed** (`data/utils.py:paste_observed`): after decoding, actually-observed views (mean-subtracted) are pasted into the prediction — error only comes from unobserved views.
- **All batch elements share position** at each step (simplification over the paper): the single shared `(elev, azim)` position is driven by `batch[0]`'s action.

## Wandb Logging

When `--wandb` is passed, the following are logged during phase 2:

- `train/policy_loss`, `train/reward`, `train/recon_loss`, `train/baseline` — per batch
- `train/epoch_*`, `val/recon_loss`, `val/reward` — every 10 epochs
- `val/gt`, `val/recon_final`, `val/recon_steps` — reconstruction images for a fixed val sample, every 10 epochs. Mean is added back before display so colors are in [0,1]. `val/recon_steps` shows all T=6 steps stacked vertically.

Run names encode hyperparameters (set via `--run-name` or auto-built in `train_sun360_only.sh`), e.g. `sun360only_ep2000_bs32_lr1e-3_wd5e-3_T6_lp1.0`.

## Visualization App (`app.py`)

Gradio 6 app with three tabs:

- **Offline Results** — load pre-computed `results/eval_metrics.json` / `results/transfer_metrics.json` as plots + tables; browse val panoramas by index (shows full 4×8 viewgrid).
- **Live Episode** — pick a val sample + policy (ours/random/large-action) + start position, then step through the T=6 episode one click at a time. Shows ground truth viewgrid (red=current, gold=visited), predicted viewgrid, trajectory map, and live MSE curve.
- **Run Full Eval** — runs `eval_transfer` on the full val set in a background thread with a live progress bar; saves a new `results/eval_transfer.png` when done. Dataset dropdown supports indoor360, modelnet10, panocontext.

App loads the model and val dataset once at startup. State per session is stored in `gr.State`.

## Data

**HDF5 format** (`data/sun360.py`): Torch7 column-major quirk — raw data is `(N, C, W, H)`, must `.transpose(0,1,3,2)` on load to get `(N, C, H, W)`. The dataset returns `(N, N_views, C, 32, 32)` float32, mean-subtracted (values roughly in [-0.5, 0.5]).

Grid: 4 elevations × 8 azimuths = 32 views, each 3×32×32. Action space: 14 = 3×5 neighborhood minus center (0,0).

**Available datasets:**
- `origin_code/SUN360/data/minitorchfeed/` — 10 panoramas per split (smoke test only)
- `data/indoor360_torchfeed/` — ShanghaiTech-Kujiale: 2591 train / 461 val / 498 test indoor panoramas
- `data/sun360_torchfeed/` — Full SUN360 dataset, downloaded from [Everloom/SUN360](https://huggingface.co/datasets/Everloom/SUN360) on HuggingFace and preprocessed via `data/prepare_sun360.py`

## CUDA / Environment

System has 3 GPUs; use **cuda:2** (RTX 3090). GPUs 0/1 are GTX TITAN X (old, loaded).

torch==2.3.1+cu121 is locked in `pyproject.toml` — do not upgrade. Newer torch versions (2.5+) have NCCL/cuDNN library conflicts on this machine.

## Disk

Checkpoints are ~11 MB each (phase 2 only saves actor + baseline; frozen modules are still stored for completeness). `config.py` has `save_every=100` (saves at epochs 100, 200, …, 2000 = 20 files). Do not lower `save_every` — it previously filled the 2 TB disk.

## Bug Fixes Applied (vs original pytorch code)

All bugs documented in `logs/buglog.md` have been fixed. Summary:

| Fix | Files changed |
|-----|--------------|
| Mean-subtract targets; LeakyReLU(0.8) decoder | `data/sun360.py`, `models/completion.py` |
| Phase 2 freezes encoder/decoder/memory; actor-only optimizer | `train.py` |
| Learned scalar baseline at 150× lr; per-sample (B,) reward | `utils/rewards.py`, `train.py` |
| LocationSensor 4D input; Actor 260D input (add abs_elev) | `models/location.py`, `models/actor.py`, `config.py` |
| weight_decay=0.005; ReduceLROnPlateau scheduler | `train.py` |

The same episode-runner fixes (p_t 4D, abs_elev to actor) were applied to `eval.py`, `eval_transfer.py`, and `app.py`.
