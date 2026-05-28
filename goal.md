# Reproduction Goal 


你好。继续 paper-align/5elev-perspective-entbonus 分支上的 SUN360 active observation
completion 复现工作。请先 git log + checkpoints/ 看现状,再按照下面的 success criteria
工作,不要重新设计训练流程。

## 论文复现的成功定义 (Jayaraman & Grauman, CVPR 2018)

复现目标不是匹配论文的绝对 MSE 数字 (我们只有 1673 train panos vs paper ~7186,
domain 也是 SUN360+indoor360 混合) — 目标是 **匹配论文的核心 claim: 学到的 policy
显著优于 random**。分级目标:

| 等级 | 量化标准 | 含义 |
|---|---|---|
| 🟢 通过 | ours improvement over random ≥ 1.5× | 学到的 policy 有意义,可声称复现 |
| 🟡 部分通过 | ours 严格优于 random,improvement gap ≥ 10%,且 actor/logit_std > 0.3 | 至少 actor 在动,但 gap 偏小 |
| 🔴 失败 | ours ≤ random,或 actor/logit_std < 0.05 | policy 没学到 — 需要诊断而非继续训 |

(论文 Table 1: improvement ratio 41.22% / 19.09% = 2.16×)

## 必须检查的训练健康指标

在 wandb run 或最新 checkpoint 上验证:
1. `actor/logit_std` 末段 > 0.3 (上一轮是 0.02 = 平的)
2. `actor/entropy` 末段 < 2.5 (上限 log(14)=2.639)
3. `train/grad_norm` 末段 > 0.05 (不是被 clip 死)
4. eval ours/random/large-action 三条 curve 单调下降

## 执行步骤

1. **看现状**:
   - `tmux list-sessions` — 若 lookaround_train 还在,先 `tmux attach -t lookaround_train`
     看是否还在跑;不在说明已结束
   - 看 `results/training_5elev_persp_entbonus.log` 最后 30 行确认正常退出
   - `ls checkpoints/ckpt_epoch*.pt` 应该看到 ep2000

2. **跑 eval + 诊断** (在 cuda:2 上):
   ```
   uv run python eval.py --checkpoint checkpoints/ckpt_epoch2000.pt \
       --data-dir data/sun360_indoor360_5elev --device cuda:2
   ```
   然后用之前的 action histogram + entropy 诊断脚本看 policy 是否 collapse。

3. **对照判定**: 把新 ours / random / large-action 和 paper Table 1 + 前三轮历史结果
   (4-elev sun360-only / 5-elev frozen / 5-elev memunfrozen-tile, 都在 git log 里有 commit)
   做一张四方表,套上面的分级标准给出结论。

4. **根据等级分支**:
   - 🟢: 把 eval 结果 commit 到本分支,update results/eval_metrics.json + curves,
        在 readme.md 加个 "Reproduction Results" 段。准备发起 PR 合并到 master。
   - 🟡: 同上提交结果,但也开一个新 issue 列下一步可改方向 (见 5)。
   - 🔴: **不要继续训**。先做根因诊断:
        (a) Phase 1 末态 recon_loss 是否正常 (< 0.05)?
        (b) Phase 2 的 actor/logit_std 整条曲线是单调上涨还是一直 ≈ 0?
        (c) reward 在不同 action 上的 variance 是多少 (用 1 个 batch 强制不同 action
            跑 episode 比较)?
        把诊断报告 commit 后再决定下一步。

## 如果失败,可选的下一轮设计调整 (不要自己做,先报告)

- Entropy bonus α=0.01 可能是错方向 (logits 本来就 maxed,bonus 在强化均匀)。
  需要的是更强的 REINFORCE 信号,不是更多探索熵
- 改用 per-step reward (违反 paper 但 paper 也明说他们试过)
- 在 actor head 加 layer-norm 让初始化时 logits 尺度更大
- Gumbel-softmax 替代 categorical sampling
- 增加 phase 1 epoch 数,让 encoder/decoder 学得更好

切记: 用 advisor() 在做实质决策前征询一次。诚实报告失败,不要掩饰指标。

---

## 设计依据 

1. **绝对 MSE 不可比** (数据规模差 4.3×,domain 混合) → 用 "ours vs random improvement
   比值" 作为主指标,这正是 paper Sec 4.2 的核心 claim
2. **加 logit_std / entropy 健康检查** — 前三轮训练 val_loss 看起来 OK 但 actor 实际
   没学,这种 silent failure 必须显式 gate
3. **分级 + 失败时禁止继续训** — 避免下一个 Claude 浪费 4 小时再训一次得到同样结果
4. **指明可选下一轮方向但禁止自作主张** — 决策权保留在人这边

## 训练监控速查

| 操作 | 命令 |
|---|---|
| 看 tmux session 是否还在 | `tmux list-sessions` |
| 进入 tmux 看 stdout (Ctrl+B d 退出不杀) | `tmux attach -t lookaround_train` |
| 不进 tmux 看日志 | `tail -f results/training_5elev_persp_entbonus.log` |
| Wandb 项目 | https://wandb.ai/jie20-zju/lookaround-2026 |
| 预期完成时间 | 启动后约 3.7 小时,产出 `checkpoints/ckpt_epoch2000.pt` |
| 完成自动标记 | 日志末尾会附加 `=== tmux session: training finished, exit code=...` |

## 当前已运行配置 (本分支 commit 2ac25ba 之上的 tmux 训练)

- Data: `data/sun360_indoor360_5elev` (1673 train / 298 val / 322 test panos)
- Grid: 5 elev × 8 azim = 40 views (真透视投影,45° FOV)
- Phase 2 trainable: memory (LSTM) + actor (对齐 paper §3.3)
- REINFORCE entropy bonus α=0.01 (注: 见上面"如果失败"段的怀疑)
- Baseline lr: actor lr × 10 (从原 ×150 降下来)
- Wandb 已加: `actor/entropy`, `actor/logit_std` per-batch
- Run name: `sun360_indoor360_5elev_persp_entbonus_ep2000_bs32_lr1e-3_wd5e-3_T6_lp1.0`

## 历史 checkpoints (供回归对比)

```
checkpoints/4elev_backup/                       — 4×8 grid, frozen mem, sun360-only (~518 panos)
checkpoints/5elev_sun360only_backup/            — 5×8, frozen mem, sun360-only
checkpoints/5elev_combined_frozen_backup/       — 5×8, frozen mem, combined 2293 (ep ~1130, 提前中止)
checkpoints/5elev_memunfrozen_tile_backup/      — 5×8, unfrozen mem, tile views (ep2000)
checkpoints/5elev_persp_entbonus_killed_backup/ — 5×8, unfrozen mem, persp, entbonus α=0.01 (ep300, killed)
checkpoints/5elev_persp_nobn_noent_killed_backup/—5×8, unfrozen mem, persp, no BN, no entbonus (ep200, killed)
checkpoints/ckpt_epoch*.pt                      — per-sample trajectory (ep200, killed; final state)
```

---

## STATUS (2026-05-28): 🔴 Failure — see `results/reproduction_status.md`

6 iterations attempted, full paper alignment achieved (per-sample trajectories,
unfrozen memory, real perspective projection, no BN, no entropy bonus, baseline
lr ×10). Final eval: ours=41.95, random=43.50 (1.30× improvement ratio vs paper's
2.16×). Gate per goal.md: 🔴 (actor/logit_std=0.022 < 0.05 threshold; gap 3.56% <
10% threshold).

**Root cause**: even with all paper-aligned fixes, the actor's REINFORCE PG gradient
is too noisy to overcome the small per-action reward variance from a near-uniform
initial policy. Logit_std grows from 0.011 → 0.022 over 100 phase-2 epochs — too
slow to clear the 0.3 threshold by 2000 epochs.

**Next iteration options** documented in `results/reproduction_status.md` §"What
was NOT tried". Decision deferred to user — these involve choosing between paper
deviations (advantage normalization, weight-decay-off-on-actor, larger lr) and
project-scope changes (more data, multi-trajectory sampling, PPO).
