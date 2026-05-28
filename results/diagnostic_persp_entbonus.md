# Diagnostic Report — perspective + entropy bonus run (killed at epoch ~190)

Per `goal.md` decision tree, killed training early when health gates flagged 🔴 silent
failure pattern identical to the 3 prior runs.

## Trigger: 🔴 gate failure at epoch 190 / 2000

| Indicator | Threshold | Measured | Status |
|---|---|---|---|
| `actor/logit_std` | > 0.3 | **0.013** | ✗ 23× too small |
| `actor/entropy` | < 2.5 | **2.639** (= log(14)) | ✗ pinned at maxent |
| `train/grad_norm` | > 0.05 | **0.008** | ✗ 6× too small |

## Diagnostic findings

### (a) Phase 1 pretraining recon_loss

```
epoch 10: 0.0493
epoch 20: 0.0493
epoch 30: 0.0492
epoch 40: 0.0494
epoch 50: 0.0495
```

Slightly above the 0.05 threshold but acceptable — perspective views are inherently
harder to reconstruct than tile-resize (each cell is now a true 45° projection with
distinct content, no spatial smoothing tricks for the decoder). Phase 1 is not the
bottleneck.

### (b) Phase 2 `actor/logit_std` trajectory

```
step  2700 (start phase 2): 0.1107   ← initial Xavier init
step  3600 (~17 batches):   0.0141   ← collapsed 8× within first phase-2 epoch
step  5600:                 0.0125
step  7600:                 0.0109
step  8600:                 0.0089   ← minimum
step 12600:                 0.0180   ← slight bounce, still ~6× below init
```

**Logits collapse from 0.11 → 0.01 within ~17 batches of phase 2 and never recover.**
This is the entropy-bonus gradient actively pulling logits toward the maxent uniform
distribution. The α=0.01 term in `pg_loss = pg_loss - 0.01·H(π)` (added in this
iteration to "fix" flat logits) was the exact wrong sign — maxent direction reinforces
the failure mode rather than escaping it.

### (c) Reward variance across actions (NEW SIGNAL FROM PERSPECTIVE PROJECTION)

Measured on 6 random val panos × 14 first-action choices, deterministic argmax after:

```
 pano  a00  a01  a02  a03  a04  a05  a06  a07  a08  a09  a10  a11  a12  a13   min   max  range_pct
  197  22.17  21.22  19.84  20.02  20.20  21.76  21.35  21.32  21.37  22.61  22.60  22.61  22.60  22.61   19.84  22.61  12.81%
  215  44.95  44.77  44.78  44.83  44.54  45.71  45.05  47.08  47.34  46.71  46.82  46.71  46.82  46.71   44.54  47.34   6.11%
   20  15.93  16.04  15.95  15.83  15.67  16.44  16.44  16.49  16.50  16.44  16.44  16.50  16.49  16.50   15.67  16.50   5.13%
  132  24.01  24.26  24.38  24.39  24.43  26.32  26.25  26.27  26.34  26.32  26.25  26.34  26.27  26.34   24.01  26.34   9.09%
  261  79.20  79.26  78.66  77.68  77.31  83.74  83.40  83.61  82.77  81.77  81.54  80.42  79.21  78.56   77.31  83.74   7.99%
  248  11.82  11.81  11.95  11.95  11.95  11.82  11.81  11.95  11.95  10.92  10.84  10.85  11.06  11.34   10.84  11.95   9.64%

Mean reward range across panos: 8.46%
(Threshold for learnable REINFORCE signal: >5%; below ~2% noise dominates)
```

**Perspective projection DID add meaningful action-conditional reward variance** —
8.46% mean range vs the ~2% noise floor of tile-based runs. The signal is now in
principle extractable, but the entropy bonus is preventing the actor from extracting it.

## Two paper-deviations to remove (next iteration)

| Deviation | Where | Effect | Fix |
|---|---|---|---|
| `BatchNorm1d(128)` before actor output head | `models/actor.py:31` | Whitens pre-logits every batch — second restoring force toward uniform logits, in addition to entropy bonus | Remove (origin Lua actor has no BN — verified `grep -i BatchNorm origin_code/SUN360/SUN360ActiveMod.lua` is empty) |
| Entropy bonus α=0.01 | `train.py:entropy_coef` | Pulls logits to maxent uniform — proven by 8× logit_std collapse in 17 batches | Set α=0.0 (paper has no entropy term) |

## Next iteration: paper-aligned variant

Single batch of changes for next training run:
- Remove BN from `Actor.net`
- Set `entropy_coef = 0.0` in `train_full`
- Keep everything else (perspective projection, memory-unfrozen, baseline lr ×10) —
  those are paper-aligned and not the failure source

Realistic expected outcome per goal.md: **🟡 (partial pass)** more likely than 🟢.
The fix removes the two restoring forces pulling logits to uniform, but whether
the small REINFORCE gradient is strong enough to differentiate actions on its own
with 32-sample batches and a 1673-pano dataset is empirically untested.

Will commit eval results regardless of tier, per goal.md instructions.
