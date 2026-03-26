# Design Doc: PyTorch Reimplementation of "Learning to Look Around" (CVPR 2018)

**Document version:** 2.0
**Target reader:** AI coding agent
**Paper:** Jayaraman & Grauman, CVPR 2018 — [arXiv:1709.00507](https://arxiv.org/abs/1709.00507)
**Original codebase:** https://github.com/dineshj1/lookaround (Torch7/Lua, reference only: `code-2017/`)

---

## 0. Goal

Produce a clean, runnable PyTorch reimplementation of the "Learning to Look Around" paper. The agent learns a policy for active panoramic scene completion — given a few observed views of a 360° scene, it selects which direction to look next to minimally reduce its uncertainty about the unobserved portions, without knowing any downstream task in advance.

The implementation must be self-contained, reproducible, and written in idiomatic PyTorch (2.x). No external RL libraries. No VLA components.

---

## 1. Problem Formulation

**Setup:**
- The world is a 360° panoramic scene discretized into a **viewgrid**: a 5 × 8 grid of views (5 elevations × 8 azimuths = **40 views** total).
- Each view is a `3 × 32 × 32` image crop (RGB, 32×32 pixels).
- The agent receives a budget of **T = 6** views to observe.
- The agent's goal is to complete the viewgrid — reconstruct all 40 views given only the T observed ones.
- No task supervision: reward comes purely from reduction in completion error.

**Observation Space:**
At each timestep t, the agent has:
- A memory state `a_t` (LSTM hidden state) summarizing all views seen so far.
- The current view `x_t`: a `3×32×32` crop at 2D position `θ_t = (elev, azim)`.
- Proprioceptive metadata `p_t = [absolute_elevation/n_elev, Δelev/n_elev, Δazim/n_azim]` (dim = 3).

**Action Space:**
- Discrete: **14 actions** representing 2D offsets `(Δelev, Δazim)` within a 3-elev × 5-azim neighborhood of the current position, excluding the center (no-stay).
- Actions wrap around on the azimuth axis (cylindrical world).
- Elevation is clamped to valid range `[0, n_elev-1]`.

**Reward Signal:**
```
R(X) = -MSE_final(recon_T, target_viewgrid)
```
Reward is computed **once**, at the final timestep T, as the negative reconstruction MSE. Used only for REINFORCE updates to the actor. The reconstruction loss for sense/fuse/aggregate/decode is summed across all timesteps (see Section 3).

**Episode Structure:**
```
for each episode (one panorama):
    θ_0 = random initial (elev, azim)
    delta_0 = θ_0.azim          # azimuth offset for rotation compensation
    init h_0, c_0 = zeros

    for t = 0 to T-1:
        x_t = viewgrid[θ_t]
        p_t = [θ_t.elev/n_elev, Δelev/n_elev, Δazim/n_azim]
        f_t = combine(encoder(x_t), loc_sensor(p_t))
        a_t, (h,c) = LSTM(f_t, (h,c))
        recon_t = decode(a_t)
        recon_t = circ_shift(recon_t, delta_0)   # align to GT azimuth frame
        recon_t = paste_observed(recon_t)         # paste seen views

        if t < T-1:
            δ_t ~ actor(a_t, rel_pos_t, t/T)
            θ_{t+1} = step(θ_t, δ_t)

    R = -MSE(recon_T, target)
    optimize actor via REINFORCE(R, ema_baseline)
    optimize all other modules via sum of reconstruction MSEs over t
```

---

## 2. Architecture

Five modules: **sense** (ViewEncoder + LocationSensor), **combine** (CombineModule), **aggregate** (AgentMemory/LSTM), **decode** (CompletionHead), **act** (Actor).

```
ViewEncoder ──┐
              ├─→  CombineModule  →  AgentMemory (LSTM)  ─→  Actor
LocationSensor┘                              │
                                             └─→  CompletionHead
```

### ViewEncoder (`models/encoder.py`)

3-layer CNN:
```
Conv(3→32,   5×5, stride=1, pad=2) → MaxPool(3×3, stride=2) → ReLU
Conv(32→32,  5×5, stride=1, pad=2) → AvgPool(3×3, stride=2) → ReLU
Conv(32→64,  5×5, stride=1, pad=2) → AvgPool(3×3, stride=2) → Flatten
Linear(576 → D_enc=256)
```
Output: `(B, 256)`

Note: After 3 conv+pool stages on a 32×32 input: 32→16→8→4, then 64 channels → `64×3×3=576` (accounting for 3×3 pool with stride 2 on 8px → 3px). Verify spatial size with a forward pass.

### LocationSensor (`models/location.py`)

Small MLP encoding proprioceptive metadata:
```
Linear(3 → 16) → ReLU
```
- Input `p_t = [abs_elev_norm, Δelev_norm, Δazim_norm]` → dim 3
- Output: `(B, 16)`
- At t=0, relative motion = (0, 0).

### CombineModule (`models/combine.py`)

Fuses ViewEncoder output + LocationSensor output:
```
Concat([patch_feat(256), loc_feat(16)]) → (B, 272)
→ Linear(272 → 256) → ReLU
→ Linear(256 → 256) → BatchNorm(256) → Dropout(0.3)
```
Output: `(B, 256)` = `f_t`, fed into LSTM each timestep.

### AgentMemory (`models/memory.py`)

```python
nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
```
- `D_hidden = 256`
- Full `(h, c)` state maintained across timesteps within an episode.
- State reinitialized between episodes, NOT between timesteps.
- Output `a_t = h_t[-1]`: `(B, 256)`.

### CompletionHead (`models/completion.py`)

Transposed-convolution decoder:
```
Linear(256 → 1024) → LeakyReLU(0.2)
→ Reshape(B, 64, 4, 4)
→ ConvTranspose2d(64→256,  5×5, stride=2, pad=2, output_pad=1)  → ReLU
→ ConvTranspose2d(256→128, 5×5, stride=2, pad=2, output_pad=1)  → ReLU
→ ConvTranspose2d(128→N_views*C, 5×5, stride=2, pad=2, output_pad=1)  → Sigmoid
→ Reshape(B, N_views=40, C=3, H=32, W=32)
```
Spatial progression: `4 → 8 → 16 → 32` (×2 each stage).
Output: `(B, 40, 3, 32, 32)` — predicted viewgrid at timestep t.

### Actor (`models/actor.py`)

3-layer MLP with position + time inputs:
```
Input: concat([a_t(256), rel_pos(2), t/T(1)]) → (B, 259)
→ Linear(259 → 128) → ReLU
→ Linear(128 → 128) → ReLU → BatchNorm(128)
→ Linear(128 → K=14)        # raw logits
```
- `rel_pos = (θ_t.elev - θ_0.elev, θ_t.azim - θ_0.azim)` (normalized)
- Logits → `Categorical` distribution → sample action (training) or argmax (eval)

---

## 3. Training Algorithm

### Phase 1 — Pretraining (T=1)

- Run single-step episodes: observe one view, predict full viewgrid, compute reconstruction loss.
- Train ViewEncoder, LocationSensor, CombineModule, AgentMemory, CompletionHead.
- No actor involved. No REINFORCE.
- Run for `pretrain_epochs = 50` epochs.

### Phase 2 — Full Training (T=6)

Unfreeze all modules. Train jointly with both losses.

**Reconstruction Loss (sum over all timesteps):**
```python
recon_loss = sum(
    F.mse_loss(circ_shift(recon_t, delta_0), target_viewgrid)
    for t in range(T)
)
```
Backprop flows through all modules via BPTT.

**Policy Loss (REINFORCE, final step):**
```python
R = -F.mse_loss(circ_shift(recon_T, delta_0), target_viewgrid).detach()
ema_baseline = alpha * ema_baseline + (1 - alpha) * R.item()
advantage = R - ema_baseline
policy_loss = -(sum(log_probs)) * advantage   # log_probs: list of T-1 tensors
```

**Total loss:**
```python
loss = recon_loss + lambda_policy * policy_loss
optimizer.zero_grad()
loss.backward()
clip_grad_norm_(all_params, max_norm=5.0)
optimizer.step()
```

### Rotation Compensation (CircShift)

The predicted viewgrid treats the first view's azimuth as origin. Apply circular shift along the azimuth dimension before computing any MSE:

```python
# data/utils.py
def circ_shift_viewgrid(recon: Tensor, delta_0: int,
                         n_elev: int = 5, n_azim: int = 8) -> Tensor:
    B, N, C, H, W = recon.shape
    recon = recon.view(B, n_elev, n_azim, C, H, W)
    recon = torch.roll(recon, shifts=delta_0, dims=2)  # shift azimuth dim
    return recon.view(B, N, C, H, W)
```

### Memorize Observed Views

After circ_shift, paste the actually observed patches into the predicted viewgrid:

```python
# data/utils.py
def paste_observed(recon: Tensor, observed: dict,
                   delta_0: int, n_azim: int = 8) -> Tensor:
    """
    recon: (B, N_views, C, H, W) — already shifted
    observed: dict mapping (elev_idx, azim_idx) → (B, C, H, W)
    """
    recon = recon.clone()
    for (elev_idx, azim_idx), view in observed.items():
        shifted_azim = (azim_idx + delta_0) % n_azim  # compensate shift
        flat_idx = elev_idx * n_azim + shifted_azim
        recon[:, flat_idx] = view
    return recon
```

---

## 4. Data Pipeline

### SUN360Dataset (`data/sun360.py`)

HDF5 structure (from `code-2017/SUN360/sun360.lua`):
```
/target_viewgrid          float32 (N, 40, 3, 32, 32), values in [0, 1]
/labs                     int     category labels
/average_target_viewgrid  float32 (40, 3, 32, 32), for mean subtraction
/gridshape                int     [5, 8]
/view_snapshape           int     [32, 32]
```

Split keys: check actual HDF5 — likely separate keys or separate files.

```python
class SUN360Dataset(Dataset):
    def __init__(self, h5_path: str, split: str = 'train',
                 mean_subtract: bool = False): ...
    def __getitem__(self, idx) -> Tensor:
        return self.viewgrids[idx]  # (40, 3, 32, 32)
    def __len__(self) -> int: ...
```

---

## 5. Evaluation

**Metric:** Per-pixel MSE × 1000 at final step T on the validation set.

**Paper Table 1 targets (SUN360):**
| Method | MSE×1000 |
|---|---|
| 1-view | 39.40 |
| random | 31.88 |
| large-action | 30.76 |
| **ours** | **23.16** |

**Baselines** (`models/baselines.py`):

| Class | Behavior |
|---|---|
| `RandomPolicy` | Uniform random action from 14-action set |
| `LargeActionPolicy` | Always select action with largest magnitude (perimeter of 3×5 grid) |

---

## 6. Configuration

```python
@dataclass
class Config:
    # Data
    h5_path: str = "data/sun360.h5"
    n_elev: int = 5
    n_azim: int = 8
    n_views: int = 40           # n_elev * n_azim
    view_height: int = 32
    view_width: int = 32

    # Model
    d_enc: int = 256
    d_loc_in: int = 3           # LocationSensor input dim
    d_loc: int = 16             # LocationSensor output dim
    d_hidden: int = 256         # LSTM hidden dim
    n_actions: int = 14         # 3-elev × 5-azim neighborhood minus center
    dropout: float = 0.3

    # 14 2D action offsets (Δelev, Δazim) — 3×5 minus center (0,0)
    action_deltas: list = field(default_factory=lambda: [
        (-1,-2),(-1,-1),(-1, 0),(-1, 1),(-1, 2),
        ( 0,-2),( 0,-1),        ( 0, 1),( 0, 2),
        ( 1,-2),( 1,-1),( 1, 0),( 1, 1),( 1, 2),
    ])

    # Episode
    T: int = 6

    # Training
    batch_size: int = 32
    lr: float = 1e-3            # Adam
    lambda_policy: float = 1.0
    baseline_decay: float = 0.9 # EMA alpha
    max_grad_norm: float = 5.0
    n_epochs: int = 2000
    pretrain_epochs: int = 50

    # Logging
    log_every: int = 100
    save_every: int = 5
    checkpoint_dir: str = "checkpoints/"
    use_wandb: bool = False
```

---

## 7. File Structure

```
lookaround_pytorch/          ← project root (this repo)
├── config.py
├── train.py                 # Two-phase: pretrain (T=1) then full (T=6)
├── eval.py
├── models/
│   ├── __init__.py
│   ├── encoder.py           # ViewEncoder
│   ├── location.py          # LocationSensor
│   ├── combine.py           # CombineModule
│   ├── memory.py            # AgentMemory (LSTM)
│   ├── actor.py             # Actor (3-layer MLP)
│   ├── completion.py        # CompletionHead (transposed-conv decoder)
│   └── baselines.py         # RandomPolicy, LargeActionPolicy
├── data/
│   ├── __init__.py
│   ├── sun360.py            # SUN360Dataset
│   └── utils.py             # get_view, circ_shift_viewgrid, paste_observed
├── utils/
│   ├── __init__.py
│   ├── rewards.py           # EMA baseline + REINFORCE loss
│   └── logging.py           # wandb/tensorboard
├── pyproject.toml
└── README.md
```

---

## 8. Requirements

```
torch>=2.0.0
torchvision>=0.15.0
h5py>=3.8.0
numpy>=1.24.0
tqdm>=4.65.0
wandb>=0.15.0
matplotlib>=3.7.0
```

Managed with `uv`. Install: `uv sync`

---

## 9. Implementation Notes & Gotchas

- **Azimuth wrap-around:** `azim % n_azim`. Elevation: `clamp(0, n_elev-1)`.
- **CircShift always before MSE:** Apply `circ_shift_viewgrid(recon, delta_0)` before every loss and before `paste_observed`. The shift aligns predicted azimuth-0 with true azimuth `delta_0`.
- **Gradient flow:** `advantage` must be `.detach()`ed. `log_probs` are actor-only. Reconstruction loss flows through all other modules.
- **HDF5 normalization:** Values may be stored as uint8 (0–255) or float32. Normalize to `[0, 1]` if needed.
- **Episode batching:** LSTM state shape: `(1, B, 256)`. Reset between episodes.
- **EMA baseline is a scalar:** `alpha * baseline + (1-alpha) * R.mean().item()`.
- **Pretrained checkpoint:** Save/load full model state between Phase 1 and Phase 2.
- **Conv output size:** Verify 576 feature dim from ViewEncoder with `assert encoder(torch.zeros(1,3,32,32)).shape == (1,256)`.

---

## 10. What NOT to Do

- Do not use a pretrained backbone (ViT, CLIP, ResNet)
- Do not use PPO/A2C — vanilla REINFORCE with EMA baseline only
- Do not use 1D azimuth-only actions — use the full 2D 14-action space
- Do not share weights between CompletionHead and ViewEncoder
- Do not apply reconstruction loss only at the final step — sum over all T timesteps

---

## 11. Suggested Implementation Order

1. `data/utils.py` — `get_view`, `circ_shift_viewgrid`, `paste_observed`
2. `data/sun360.py` — HDF5 loader
3. `models/encoder.py` + `models/location.py` + `models/combine.py`
4. `models/completion.py` — train standalone (reconstruction only)
5. `models/memory.py` — integrate LSTM
6. `utils/rewards.py` — EMA baseline + REINFORCE
7. `models/actor.py` + full `train.py`
8. `models/baselines.py` + `eval.py`
