# Reproduction Status — Final Report (this branch)

**Tier per goal.md: 🔴 Failure (with sharp root-cause attribution).**

Despite 6 iterations of architectural and procedural changes converging to a
paper-aligned setup (see table below), the learned actor never escapes a near-
uniform softmax. The policy delivers a small but statistically observable
improvement over random (1.30× improvement ratio vs paper's 2.16×), but the
core mechanism — REINFORCE differentiating actions by their reconstruction
contribution — is functionally broken in our setup.

## Six iterations summary

| # | Trajectory | Memory   | Views | Actor BN | Entropy bonus | Baseline lr | Logit std end | ours/random ratio |
|---|------------|----------|-------|----------|---------------|-------------|---------------|-------------------|
| 1 | shared     | frozen   | tile  | yes      | no            | ×150        | 0.014         | 1.04× *           |
| 2 | shared     | frozen   | tile  | yes      | no            | ×150        | 0.014         | 1.31× **          |
| 3 | shared     | unfrozen | tile  | yes      | no            | ×150        | 0.014         | 0.86× ***         |
| 4 | shared     | unfrozen | persp | yes      | α=0.01        | ×10         | 0.013         | 1.04×             |
| 5 | shared     | unfrozen | persp | no       | no            | ×10         | 0.017         | not eval'd ****   |
| 6 | per-sample | unfrozen | persp | no       | no            | ×10         | **0.022**     | **1.30×** (ep200) |

\* 4-elev mini-SUN360 (different val) — 1.04 ≈ noise.
\** 4-elev sun360-only (518 panos).
\*** ours WORSE than random — fixed-action coverage bias.
\**** killed at epoch 200 when gates failed; not committed.

Paper Table 1 reference: 41.22% / 19.09% = **2.16×**.

## Root cause: REINFORCE PG signal:noise still inadequate after all paper-aligned fixes

The 6 iterations isolated the following:

1. **SNR bug** (shared trajectory): `train.py` used `elev_cur[0]/action[0]` to drive
   the trajectory for the whole batch — 31/32 of log_probs were causally unrelated
   to rewards. Fixed in iter 6 (commit `245b4ec`). Actor grad norm went from
   0.008 → 0.12, confirming 15× signal recovery.
2. **Entropy bonus α=0.01** was actively pulling logits toward maxent uniform
   (wrong sign for our failure mode). Removed in iter 5+ (commit `fd35b08`).
3. **BatchNorm before actor output head** was whitening pre-logits every batch
   (port-time addition, not in origin Lua). Removed in iter 5+ (commit `fd35b08`).
4. **Baseline lr ×150 → ×10** to keep advantage from collapsing to zero on first
   batch.
5. **Tile-resize → real 45° FOV perspective views**: lifted per-action reward
   variance from ~2% to **8.46%** (well above the 5% learnability threshold),
   confirming the reward signal IS discriminating.
6. **memory.lstm unfrozen in phase 2** to align with paper Sec 3.3 ("train
   aggregate and act ... while other modules are frozen").

Even with all 6 fixes in place (iter 6), the actor still produces logit_std ≈
0.02 and entropy ≈ log(14). Deterministic argmax picks the same action (a=1,
de=-1 da=-1) for 100% of decisions, just like the prior 5 collapsed runs. The
ours/random improvement ratio of 1.30× comes from this single-action policy
happening to have better-than-random spatial coverage on the val set (sweeping
diagonally down + left), not from learned action conditioning.

## What's actually wrong

With paper-aligned setup, the equilibrium for our actor:
- Initial logits at Xavier init: std ≈ 0.04 (small)
- Per-sample reward range across actions: 8.46% relative (~5e-3 absolute MSE units)
- Per-sample advantage `(R - b)` ≈ ±5e-3 (after baseline tracks well)
- Per-step PG gradient on logit: ~5e-3 (random-walk-biased)
- Adam-adapted update: ~lr/sqrt(v) → small directional drift
- Weight decay shrinkage on each Adam step: λ × wd = 1e-3 × 5e-3 = 5e-6 × |θ|

Net: the directional PG drift competes with random noise plus weight-decay
shrinkage. Over 5300 steps (100 phase-2 epochs), logits grew from 0.011 to
0.024 — too slow to reach the ~0.3 magnitude that would meaningfully shape
softmax probabilities.

This is the canonical "REINFORCE with tiny per-action reward variance from a
near-uniform initial policy" pathology. It is NOT solved by removing
restoring forces; it's a fundamental signal:noise issue at our scale.

## Why the paper succeeded where we don't

Speculation, not yet verified, in decreasing likelihood:

1. **~4× more training data**: paper had ~7186 train panos vs our 1673. More
   panos per epoch → lower batch reward variance → cleaner advantage signal.
2. **Pure SUN360 (no indoor360 mixing)**: our combined dataset has two
   different scene distributions; advantage variance is correspondingly larger.
3. **Better initial actor**: paper may use a non-trivial init that breaks the
   symmetry early. (Origin Lua not investigated in detail for this.)
4. **Longer training**: paper doesn't specify exact epoch count; our 2000 may
   be too few to escape the slow-growth regime even with paper's setup.
5. **`rewardScale=0.01` + `learningRate=15`** (origin Lua/SGD-with-momentum,
   non-trivially comparable to our Adam lr=1e-3 with no extra scale) — unclear
   effect.

## What was NOT tried (deferred to future iteration; goal.md says report, don't pivot)

In rough order of likely impact:

1. **Advantage normalization** (centered + std-normalized within batch). Standard
   REINFORCE variance-reduction trick. Doesn't change unbiased gradient direction
   but ensures consistent magnitude, useful when Adam's adaptive scaling collides
   with reward heterogeneity.
2. **Larger actor lr** (5e-3 or 1e-2 with separate optimizer from memory).
3. **Init actor final-Linear with larger std** (0.3 instead of Xavier's ~0.1) so
   logits start with meaningful variance and PG only needs to refine direction.
4. **Disable weight_decay on actor head**.
5. **Multiple trajectories per panorama** (sample K=4-8 trajectories with same
   pano, compute advantage relative to per-pano mean — classic REINFORCE variance
   reduction). Memory cost: 4-8× more decoder passes per batch.
6. **Use full SUN360 if available** (HF Everloom/SUN360 only has 518 panos;
   would need to find a different SUN360 mirror or contact paper authors).
7. **Switch to PPO** — modern PG method that handles the noisy-gradient regime
   much better with clipping. Deviates from paper.

## What to do next

Per goal.md "如果失败,可选的下一轮设计调整 (不要自己做,先报告)":

This report IS the failure documentation. The branch contains all paper-aligned
fixes plus the per-sample SNR fix. The remaining gap to 🟡/🟢 is a fundamental
REINFORCE conditioning issue that requires either (a) a new variance-reduction
mechanism not in the paper, or (b) substantially more training data. Both are
project-scope decisions, not implementation tweaks. Hand back to user.

## Artifacts on this branch

- `results/eval_metrics.json` + `eval_mse_curve.png` — final eval (epoch 200 of
  per-sample run, 298 val panos)
- `results/diagnostic_persp_entbonus.md` — diagnostic from iter 4 (kept for the
  reward-variance + logit-std trajectory data)
- `results/reproduction_status.md` — this file
- `results/training_5elev_persp_persample.log` — full training log of iter 6
- 6 backup ckpt dirs in `checkpoints/*_backup/` — historical runs for regression
