# Bug Log — LookAround 2026 PyTorch Reimplementation

Discovered by comparing `origin_code/SUN360/lookaround.lua`, `VRRegressReward.lua`,
`actorMod.lua`, and `sun360.lua` against our PyTorch code.

---

## P0 — `mean_subtract_output` missing (root cause of policy_loss ≈ 0)

**Where:** `data/sun360.py` `SUN360Dataset._load()`, `models/completion.py`, `train.py`

**Original (`lookaround.lua:88`):**
```lua
cmd:option('--mean_subtract_output', 1, ...)  -- default True
-- sun360.lua: target = raw_pixels/255 - average_viewgrid/255
```
The reconstruction **target** is mean-subtracted. The decoder's final activation is
`LeakyReLU(0.8)` (allows negative outputs to match the centered target).

**Ours:**
- Target is raw pixels in [0, 1] — decoder can cheat by predicting the global mean
  (~0.4) everywhere and already achieve low MSE
- Final activation is `Sigmoid` — cannot produce negative values even if mean
  subtraction were added
- Consequence: reward `-MSE_final` has near-zero variance across policies, so
  `policy_loss ≈ 0.0000` throughout all 2000 epochs

**Fix:**
1. In `SUN360Dataset._load()`: subtract `average_target_viewgrid/255` from target viewgrids
   (use `mean_subtract=True` flag; already stored in HDF5 as `/average_target_viewgrid`)
2. In `CompletionHead.deconv`: replace final `nn.Sigmoid()` with `nn.LeakyReLU(0.8, inplace=True)`
3. Clip `paste_observed` values accordingly (observed patches should also be mean-subtracted
   before pasting)

---

## P0 — Phase 2 trains all modules together (policy gradient drowned out)

**Where:** `train.py` `train_full()`

**Original (`example.sh`):**
```bash
# Phase 2: only Actor trains; encoder/decoder/LSTM are frozen
th lookaround.lua --finetune_lrMult 0 --finetuneDecoderFlag 1 --rho 4 ...
```
`finetune_lrMult=0` sets the learning rate of all pretrained modules to zero.
Only the Actor's parameters receive gradients in Phase 2.

**Ours (`train.py:240`):**
```python
all_params = encoder + loc_sensor + combine + memory + completion + actor
optimizer = Adam(all_params, lr=1e-3)
```
`recon_loss ≈ 0.14` vs `policy_loss ≈ 0.0003` — recon loss is ~450× larger,
completely masking the actor gradient.

**Fix:**
In `train_full()`, create a separate optimizer for the actor only. Freeze all
other modules during Phase 2 (set `requires_grad=False` or use a separate
param group with `lr=0`).

---

## P1 — Baseline is EMA scalar, not a learned parameter

**Where:** `utils/rewards.py`, `train.py`

**Original (`lookaround.lua:607-611`):**
```lua
seq[step]:add(nn.Constant(0,1)):add(nn.Add(1))  -- trainable scalar, init=0
-- VRRegressReward.lua:
self.gradInput[2][t] = self.criterion:backward(step_baseline, self.reward[t])
self.gradInput[2][t]:mul(self.baseline_lr_factor)  -- baseline_lr_factor = 150
```
Baseline is a trainable `nn.Add(1)` (single scalar parameter, init=0) trained
via MSE against the actual reward, at **150× the actor's learning rate**.
This converges in ~10 steps and gives low-variance advantage estimates.

The reward is also **per-sample** `(B,)`, not a scalar batch mean.

**Ours:**
```python
baseline = alpha * baseline + (1 - alpha) * reward.item()  # EMA, non-differentiable
advantage = (reward - baseline)  # reward is already batch-mean scalar
```
EMA with α=0.9 converges slowly (~10/(1-0.9)=100 steps to track changes).
Using batch-mean reward throws away per-sample variance information.

**Fix:**
Replace EMA with a learned `nn.Linear(1, 1, bias=False)` (or `nn.Parameter`)
scalar per timestep, trained with a dedicated high-lr optimizer.
Compute per-sample reward `(B,)` and advantage `(B,)` instead of scalar mean.

---

## P2 — Actor and LocationSensor inputs differ from original

**Where:** `models/actor.py`, `models/location.py`, `train.py`

**Original:**
```lua
-- LocationSensor input (location_ipsz):
--   [rel_elev, rel_azim, t/T, abs_elev]  → 4 inputs
location_ipsz = 2 + 1 + 1  -- rel_pos + time + knownElev

-- Actor input (act_ipsz):
--   [h_t(256), rel_elev, rel_azim, t/T, abs_elev]  → 260 inputs
act_ipsz = hiddenSize + 2 + 1 + 1
```

**Ours:**
```python
# LocationSensor: [abs_elev_norm, d_elev_prev_norm, d_azim_prev_norm] → 3 inputs
p_t = [elev_cur/n_elev, d_elev_prev/n_elev, d_azim_prev/n_azim]

# Actor: [h_t(256), rel_elev, rel_azim, t/T] → 259 inputs (missing abs_elev)
```

Two differences:
1. LocationSensor uses **previous-step deltas** instead of **cumulative relative position**
2. Actor is missing **absolute elevation** as an explicit input

**Fix:**
- Change `p_t` to `[rel_elev_norm, rel_azim_norm, t/T, abs_elev_norm]`
- Add `abs_elev` to actor input: `cat([h_t, rel_pos, time_frac, abs_elev])` → 260D
- Update `d_loc_in=4` in config and `Actor.__init__` accordingly

---

## P2 — Optimizer mismatch (SGD vs Adam, no weight decay)

**Where:** `train.py`, `config.py`

**Original:**
```lua
cmd:option('--learningRate', 0.1)     -- but example.sh uses 40 (Phase1) / 15 (Phase2)
cmd:option('--momentum', 0.9)
cmd:option('--weightDecay', 0.005)
-- lr_style = 'step_if_stagnant': halve lr when best train loss hasn't improved
--   by 0.0002 over the last 200 epochs
```

**Ours:** Adam, `lr=1e-3`, no weight decay, constant LR.

**Fix:**
- Add `weight_decay=0.005` to optimizer
- Consider LR scheduler: `ReduceLROnPlateau(patience=200, factor=0.5, threshold=0.0002)`
- Adam is acceptable as a substitute for SGD+momentum but weight decay is important
  for regularization (val gap grew to 0.041 by epoch 2000)

---

## Summary Table

| ID | File(s) | Impact | Status |
|----|---------|--------|--------|
| P0-mean | `data/sun360.py`, `models/completion.py` | Policy never learns | **Fixed** |
| P0-freeze | `train.py` `train_full()` | Policy gradient drowned | **Fixed** |
| P1-baseline | `utils/rewards.py`, `train.py` | High variance, slow convergence | **Fixed** |
| P2-inputs | `models/actor.py`, `models/location.py`, `config.py` | Minor accuracy | **Fixed** |
| P2-optim | `train.py` | Generalization gap | **Fixed** |
