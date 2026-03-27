# Learning to Look Around — PyTorch 2026

Open source, Claude Code-based, 2026 PyTorch reimplementation of:

**"Learning to Look Around: Intelligently Exploring Unseen Environments for Unknown Tasks"**
Jayaraman & Grauman, CVPR 2018 — https://arxiv.org/abs/1709.00507

Reference Lua/Torch7 code: `code-2017/` (original, non-runnable)

---

## Paper Overview

![Teaser](origin_results/fig1_teaser.png)
*Fig. 1 — An agent observing limited portions of a scene must decide where to look next to most efficiently reduce uncertainty about unobserved regions.*

The paper proposes a reinforcement learning agent that learns to **actively complete panoramic scenes** by selecting informative viewpoints. The agent is rewarded for reducing its reconstruction error over unobserved views — no downstream task supervision required.

### Architecture

![Architecture](origin_results/fig2_architecture.png)
*Fig. 2 — Five modules: **sense** (CNN patch encoder + proprioception MLP), **fuse** (combine module), **aggregate** (LSTM memory), **decode** (transposed-conv viewgrid predictor), **act** (policy MLP).*

### Qualitative Results

![Episodes](origin_results/fig3_episodes.png)
*Fig. 3 — Active completion episodes. As the agent explores, its predicted viewgrid converges toward the ground truth. Top: outdoor scene. Bottom: 3D object (unseen category).*

### Quantitative Results

![MSE vs Time](origin_results/fig4_mse_vs_time.png)
*Fig. 4 — Per-pixel MSE × 1000 vs. timestep on SUN360 (left), ModelNet seen (center), ModelNet unseen (right). The learned policy consistently outperforms random and large-action baselines.*

### Policy Transfer

![Policy Transfer](origin_results/fig5_policy_transfer.png)
*Fig. 5 — Unsupervised policy transfer: the completion policy (trained without labels) drives an active categorization system and matches the fully supervised baseline on both SUN360 and ModelNet-10.*

---

## Quick Start

```bash
# Install dependencies
uv sync

# Train (uses mini SUN360 dataset included in code-2017/)
uv run python train.py

# Evaluate a checkpoint
uv run python eval.py --checkpoint checkpoints/ckpt_epoch2000.pt
```

Data directory: `code-2017/SUN360/data/minitorchfeed/` (mini, 10 panoramas per split)
Full data: `code-2017/SUN360/data/torchfeed/` — use `--data-dir` flag

---

## TODO

### Setup
- [x] `uv init` — initialize project
- [x] Write `pyproject.toml` with all dependencies
- [x] `uv sync` — install deps (torch, torchvision, h5py, numpy, tqdm, wandb, matplotlib)
- [x] SUN360 dataset — available at `code-2017/SUN360/data/` (mini + full)

### Data pipeline
- [x] `data/utils.py` — `get_view`, `circ_shift_viewgrid`, `paste_observed`, `step_position`
- [x] `data/sun360.py` — `SUN360Dataset` (HDF5 loader, train/val/test splits, handles Torch7 HDF5 transpose quirk)

### Models
- [x] `models/encoder.py` — `ViewEncoder` (3-layer CNN → 256D)
- [x] `models/location.py` — `LocationSensor` (MLP on elevation + relative motion → 16D)
- [x] `models/combine.py` — `CombineModule` (fuse patch+loc → 256D, BN+Dropout)
- [x] `models/memory.py` — `AgentMemory` (single-layer LSTM, hidden=256)
- [x] `models/completion.py` — `CompletionHead` (transposed-conv decoder → 32×3×32×32)
- [x] `models/actor.py` — `Actor` (3-layer MLP, input=hidden+rel_pos+time → 14 logits)
- [x] `models/baselines.py` — `RandomPolicy`, `LargeActionPolicy`

### Utilities
- [x] `utils/rewards.py` — `update_ema_baseline`, `compute_reinforce_loss`
- [x] `utils/logging.py` — wandb / console logging

### Training
- [x] `train.py` — Phase 1: pretrain T=1 (encoder + decoder only, 50 epochs)
- [x] `train.py` — Phase 2: full training T=6 (REINFORCE + reconstruction loss, 2000 epochs)
- [x] Gradient clipping (`max_norm=5.0`)
- [x] Checkpoint save/resume (`checkpoints/ckpt_epoch*.pt`)
- [x] Full training completed — recon loss 0.32 → 0.12 over 2000 epochs
- [ ] Training on full SUN360 dataset (currently using mini — 10 samples)
- [ ] GPU training (currently CPU only — old CUDA driver)

### Evaluation
- [x] `eval.py` — per-pixel MSE×1000 at final step T on val/test set
- [x] Compare vs. `RandomPolicy` and `LargeActionPolicy` baselines
- [x] Plot MSE vs. timestep curve
- [ ] Run evaluation after full training completes

### Done
- [x] Design doc (`designdoc.md`) — v2.0, corrected from paper + Lua reference code
- [x] README with paper figures (`readme.md`)
- [x] Full 2000-epoch training completed (`checkpoints/ckpt_epoch2000.pt`, `training.log`)
