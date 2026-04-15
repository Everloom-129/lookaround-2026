好问题。站在 2026 年回看 2017 年的设计，以下是按模块梳理的「如果重新做，可以怎么选」：

---

## 一、Encoder：3 层浅 CNN → 预训练 backbone

原来是 3 层 5×5 卷积，输入 32×32，输出 256D。2017 年没有好的小图预训练模型，现在有很多选择：

- **用 ViT-Tiny / DINOv2-Small 做 frozen feature extractor**：即使 32×32 很小，resize 到 224 后用 DINOv2 的 patch token 做 pooling，encoder 质量会远好于 3 层 CNN。预训练权重免费，Phase 1 可能都不需要了。
- **如果坚持轻量**：MobileNetV3-Small 或 EfficientNet-B0 的前几个 block，加 AdaptiveAvgPool，也比手写 3 层 CNN 强很多。

## 二、Memory：LSTM → Transformer / SSM

单层 LSTM hidden=256，6 步序列。2017 年 LSTM 是标配，现在：

- **Causal Transformer（6 token 序列）**：T=6 非常短，一个 2-layer、4-head 的小 Transformer 就够。好处是可以 attend 到任意历史步而非只靠 hidden state 压缩。
- **Mamba / S4 (state space model)**：如果你想保持 RNN 式的 O(1) per-step inference，Mamba 是 LSTM 的现代替代，gating 机制更灵活。
- **不过说实话**，T=6 太短了，LSTM 和 Transformer 差距不会很大。更大的收益在于加长 T。

## 三、Decoder：转置卷积 → 现代生成模型

CompletionHead 用 3 层 ConvTranspose 从 256D 直接生成 32 个 32×32 的 view。这是 2017 年的做法，现在：

- **换 loss**：MSE 在像素空间会产生模糊重建。可以加 **perceptual loss**（VGG/LPIPS）或 **SSIM loss**，让重建更清晰，reward 信号也更有意义。
- **用 diffusion decoder**：latent diffusion 做条件生成（condition on LSTM hidden），质量远超转置卷积。但训练成本高，需要权衡。
- **VQ-VAE + Transformer decoder**：先把 viewgrid 编码到离散 token，再用 Transformer 自回归预测。和 DALL-E 1 的思路类似。

## 四、RL 算法：REINFORCE → PPO / A2C

REINFORCE + scalar baseline 是最基础的 policy gradient，方差大、sample efficiency 低：

- **PPO（Proximal Policy Optimization）**：clip 住 importance ratio，多次 mini-batch 复用同一 rollout，收敛更稳。实现上也就多几十行代码。
- **带 critic 的 A2C**：用一个小 MLP 做 state-dependent baseline V(s)，而不是一个全局标量。对于「不同 panorama 难度不同」的场景，per-state baseline 方差会低很多。
- **Entropy bonus**：加 `- β H(π)` 鼓励探索，防止 actor 过早坍缩到确定性策略。原论文没有这个。

## 五、训练超参和 schedule

| 2017 做法 | 2026 建议 |
|-----------|-----------|
| Adam + weight_decay=0.005 | **AdamW**（解耦 weight decay），或 **LAMB/LARS** |
| ReduceLROnPlateau | **Cosine annealing with warmup**，更平滑 |
| Phase 1 frozen → Phase 2 frozen | **渐进解冻（gradual unfreezing）**：Phase 2 先只训 actor，再慢慢 unfreeze decoder/encoder 做 end-to-end fine-tune |
| 固定 T=6 | **课程学习（curriculum）**：从 T=2 开始，逐步增加到 T=6 甚至更长，让 actor 先学简单的再学难的 |
| scalar baseline (150× lr) | **learned critic V(s)**，用 GAE（Generalized Advantage Estimation）计算 advantage |

## 六、数据和增强

- **数据增强**：原论文几乎没有 augmentation。现在可以对每个 view 做 color jitter、random erasing、mixup。对 viewgrid 做随机 azimuth rotation（已经有 `circ_shift`，可以在训练时随机 shift 做 augmentation）。
- **更大的 view 分辨率**：32×32 太低了。如果用预训练 encoder，可以提升到 64×64 或 128×128，decoder 也相应加深。
- **对比学习预训练**：用 SimCLR/BYOL 在全部 view patch 上做自监督预训练 encoder，比 Phase 1 的 reconstruction warmup 效果可能更好。

## 七、评估指标

- **除了 MSE 还可以看**：LPIPS（感知相似度）、FID/KID（生成质量）、SSIM。MSE 低不代表视觉质量好。
- **下游 task 性能**：原论文有 recognition transfer，现在可以用 CLIP 做 zero-shot 评估——看 agent 探索后的表示是否能支持 zero-shot 分类。

---

## 总结：优先级排序

如果只改 3 件事，按性价比排：

1. **PPO + learned critic V(s) 替代 REINFORCE + scalar baseline** — 实现简单、收益大
2. **Perceptual loss (LPIPS) 替代纯 MSE** — reward 信号质量直接提升
3. **预训练 encoder (DINOv2 frozen)** — 免费的表示质量提升，Phase 1 可以缩短甚至去掉