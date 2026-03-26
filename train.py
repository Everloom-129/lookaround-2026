"""
Training script for the "Learning to Look Around" PyTorch reimplementation.

Two-phase training:
  Phase 1 (pretrain): T=1 episode, reconstruction loss only, 50 epochs.
                      Warms up encoder + decoder before RL.
  Phase 2 (full):     T=6 episodes, reconstruction loss + REINFORCE, 2000 epochs.

Usage:
  uv run python train.py
  uv run python train.py --h5 /path/to/sun360.h5 --epochs 2000 --wandb
"""
import argparse
import os
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from config import Config
from data.sun360 import SUN360Dataset
from data.utils import circ_shift_viewgrid, get_view, paste_observed, step_position
from models.actor import Actor
from models.combine import CombineModule
from models.completion import CompletionHead
from models.encoder import ViewEncoder
from models.location import LocationSensor
from models.memory import AgentMemory
from utils.logging import init_logging, log_metrics
from utils.rewards import compute_reinforce_loss, update_ema_baseline


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    batch: torch.Tensor,             # (B, N_views, C, H, W)
    encoder: ViewEncoder,
    loc_sensor: LocationSensor,
    combine: CombineModule,
    memory: AgentMemory,
    completion: CompletionHead,
    actor: Actor,
    config: Config,
    device: torch.device,
    T: int,
    deterministic: bool = False,
):
    """
    Run one episode for a batch of panoramas.

    Returns:
        recon_list:   list of T tensors, each (B, N_views, C, H, W) — predicted viewgrids
        log_probs:    list of T-1 tensors, each (B,) — log probs of taken actions
        delta_0_batch:(B,) int tensor — first-view azimuth index per batch element
        elev_0_batch: (B,) int tensor — first-view elevation index
    """
    B = batch.shape[0]
    n_elev, n_azim = config.n_elev, config.n_azim

    # Random starting position per batch element
    elev_0 = torch.randint(0, n_elev, (B,))
    azim_0 = torch.randint(0, n_azim, (B,))
    delta_0 = azim_0.clone()  # azimuth offset for rotation compensation

    # Current position (same for all batch elements at same step, but
    # we track per-element to allow independent trajectories)
    elev_cur = elev_0.clone()
    azim_cur = azim_0.clone()

    # Initialize LSTM state
    h, c = memory.init_hidden(B, device)

    # Tracking
    observed: List[dict] = [{}] * B  # per-element observed views (elev, azim) → view
    # For simplicity use a shared dict indexed by (elev, azim) → (B, C, H, W)
    # (all batch elements take the same position at each step)
    # For truly independent trajectories we'd need per-element — but the original
    # paper also batches with the same position sequence per batch. We keep it simple.
    shared_observed = {}

    recon_list = []
    log_probs = []

    # Relative position tracking (from start)
    rel_elev = torch.zeros(B, device=device)
    rel_azim = torch.zeros(B, device=device)
    d_elev_prev = torch.zeros(B, device=device)
    d_azim_prev = torch.zeros(B, device=device)

    for t in range(T):
        # --- Sense ---
        # Gather the view for each batch element at its current position
        # Since all elements move in sync we use the first element's position
        e = elev_cur[0].item()
        a = azim_cur[0].item()
        x_t = get_view(batch, int(e), int(a), n_azim=n_azim).to(device)  # (B, C, H, W)
        shared_observed[(e, a)] = x_t.detach()

        # Proprioceptive metadata
        p_t = torch.stack([
            elev_cur.float() / max(n_elev - 1, 1),   # abs elev norm
            d_elev_prev.float() / max(n_elev - 1, 1), # rel elev change
            d_azim_prev.float() / n_azim,             # rel azim change
        ], dim=1).to(device)  # (B, 3)

        # --- Encode + Combine ---
        patch_feat = encoder(x_t)              # (B, 256)
        loc_feat   = loc_sensor(p_t)           # (B, 16)
        f_t        = combine(patch_feat, loc_feat)  # (B, 256)

        # --- Aggregate ---
        a_t, (h, c) = memory(f_t, (h, c))     # a_t: (B, 256)

        # --- Decode ---
        recon_t = completion(a_t)              # (B, 40, 3, 32, 32)
        # Rotation compensation — use first element's delta_0 (shared trajectory)
        d0 = delta_0[0].item()
        recon_t_shifted = circ_shift_viewgrid(recon_t, int(d0), n_elev, n_azim)
        recon_t_shifted = paste_observed(recon_t_shifted, shared_observed,
                                         int(d0), n_azim)
        recon_list.append(recon_t_shifted)

        # --- Act (skip on last step) ---
        if t < T - 1:
            rel_pos = torch.stack([
                rel_elev.float() / max(n_elev - 1, 1),
                rel_azim.float() / n_azim,
            ], dim=1).to(device)  # (B, 2)
            time_frac = torch.full((B, 1), t / T,
                                   dtype=torch.float32, device=device)
            logits = actor(a_t, rel_pos, time_frac)        # (B, 14)
            action, log_prob, _ = actor.get_action(
                logits, deterministic=deterministic
            )
            log_probs.append(log_prob)

            # Step position (using first element's action — shared trajectory)
            act_idx = action[0].item()
            de, da = config.action_deltas[act_idx]
            new_e, new_a = step_position(
                int(elev_cur[0].item()), int(azim_cur[0].item()),
                de, da, n_elev, n_azim
            )

            # Update relative position
            d_elev_prev = torch.full((B,), float(de), device=device)
            d_azim_prev = torch.full((B,), float(da), device=device)
            rel_elev = rel_elev + float(de)
            rel_azim = rel_azim + float(da)

            elev_cur = torch.full((B,), new_e, dtype=torch.long)
            azim_cur = torch.full((B,), new_a, dtype=torch.long)

    return recon_list, log_probs, delta_0


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_losses(recon_list, target, log_probs, ema_baseline, config):
    """
    recon_list: list of T tensors (B, N_views, C, H, W) — already shifted + pasted
    target:     (B, N_views, C, H, W)
    log_probs:  list of T-1 (B,) tensors
    ema_baseline: current scalar baseline

    Returns:
        total_loss, recon_loss_val, policy_loss_val, reward_val, new_baseline
    """
    # Reconstruction loss: sum over all timesteps
    recon_losses = [F.mse_loss(r, target) for r in recon_list]
    recon_loss = sum(recon_losses)

    # REINFORCE: reward = -MSE at FINAL step
    reward = -recon_losses[-1].detach()
    new_baseline = update_ema_baseline(ema_baseline, reward.item(),
                                       alpha=config.baseline_decay)

    policy_loss = compute_reinforce_loss(log_probs, reward, ema_baseline)
    total_loss = recon_loss + config.lambda_policy * policy_loss

    return (total_loss,
            recon_loss.item(),
            policy_loss.item() if isinstance(policy_loss, torch.Tensor) else 0.0,
            reward.item(),
            new_baseline)


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def pretrain(loader, encoder, loc_sensor, combine, memory, completion,
             optimizer, config, device, run):
    """Phase 1: T=1, reconstruction only."""
    encoder.train(); loc_sensor.train(); combine.train()
    memory.train(); completion.train()

    step = 0
    for epoch in range(config.pretrain_epochs):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)  # (B, 40, 3, 32, 32)
            recon_list, _, _ = run_episode(
                batch, encoder, loc_sensor, combine, memory, completion,
                actor=None,  # not used when T=1
                config=config, device=device, T=1,
            )
            loss = F.mse_loss(recon_list[0], batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(loc_sensor.parameters()) +
                list(combine.parameters()) + list(memory.parameters()) +
                list(completion.parameters()),
                config.max_grad_norm
            )
            optimizer.step()
            total_loss += loss.item()
            step += 1

        if (epoch + 1) % 10 == 0:
            avg = total_loss / len(loader)
            log_metrics({"pretrain/recon_loss": avg, "pretrain/epoch": epoch + 1},
                        step=step, run=run)

    return step


def train_full(loader, encoder, loc_sensor, combine, memory, completion, actor,
               optimizer, config, device, run, global_step=0):
    """Phase 2: T=6, reconstruction + REINFORCE."""
    ema_baseline = 0.0
    all_params = (list(encoder.parameters()) + list(loc_sensor.parameters()) +
                  list(combine.parameters()) + list(memory.parameters()) +
                  list(completion.parameters()) + list(actor.parameters()))

    encoder.train(); loc_sensor.train(); combine.train()
    memory.train(); completion.train(); actor.train()

    for epoch in range(config.n_epochs):
        epoch_recon = 0.0
        epoch_policy = 0.0
        epoch_reward = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)
            target = batch

            recon_list, log_probs, _ = run_episode(
                batch, encoder, loc_sensor, combine, memory, completion,
                actor=actor, config=config, device=device, T=config.T,
            )

            total_loss, recon_val, policy_val, reward_val, ema_baseline = \
                compute_losses(recon_list, target, log_probs,
                               ema_baseline, config)

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(all_params, config.max_grad_norm)
            optimizer.step()

            epoch_recon  += recon_val
            epoch_policy += policy_val
            epoch_reward += reward_val
            n_batches += 1
            global_step += 1

            if global_step % config.log_every == 0:
                log_metrics({
                    "train/recon_loss":  recon_val,
                    "train/policy_loss": policy_val,
                    "train/reward":      reward_val,
                    "train/baseline":    ema_baseline,
                }, step=global_step, run=run)

        # Epoch summary
        if (epoch + 1) % 10 == 0:
            log_metrics({
                "train/epoch_recon":  epoch_recon  / n_batches,
                "train/epoch_policy": epoch_policy / n_batches,
                "train/epoch_reward": epoch_reward / n_batches,
                "train/epoch":        epoch + 1,
            }, step=global_step, run=run)

        # Checkpoint
        if (epoch + 1) % config.save_every == 0:
            _save_checkpoint(epoch + 1, encoder, loc_sensor, combine,
                             memory, completion, actor, optimizer, config)

    return global_step


def _save_checkpoint(epoch, encoder, loc_sensor, combine, memory,
                     completion, actor, optimizer, config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"ckpt_epoch{epoch:04d}.pt")
    torch.save({
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "loc_sensor": loc_sensor.state_dict(),
        "combine": combine.state_dict(),
        "memory": memory.state_dict(),
        "completion": completion.state_dict(),
        "actor": actor.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"[checkpoint] saved to {path}")


def load_checkpoint(path, encoder, loc_sensor, combine, memory,
                    completion, actor, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    encoder.load_state_dict(ckpt["encoder"])
    loc_sensor.load_state_dict(ckpt["loc_sensor"])
    combine.load_state_dict(ckpt["combine"])
    memory.load_state_dict(ckpt["memory"])
    completion.load_state_dict(ckpt["completion"])
    actor.load_state_dict(ckpt["actor"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args=None):
    cfg = Config()
    if args:
        if hasattr(args, 'data_dir') and args.data_dir:
            cfg.data_dir = args.data_dir
        if hasattr(args, 'epochs') and args.epochs:
            cfg.n_epochs = args.epochs
        if hasattr(args, 'wandb') and args.wandb:
            cfg.use_wandb = True
        if hasattr(args, 'batch_size') and args.batch_size:
            cfg.batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset — load from split files in data_dir
    print(f"Loading dataset from {cfg.data_dir} ...")
    dataset = SUN360Dataset(cfg.data_dir, split="train")
    # Override config dimensions from actual data
    cfg.n_elev  = dataset.n_elev
    cfg.n_azim  = dataset.n_azim
    cfg.n_views = dataset.n_views
    cfg.view_height = dataset.view_H
    cfg.view_width  = dataset.view_W
    print(f"Grid: {cfg.n_elev}×{cfg.n_azim} = {cfg.n_views} views, each 3×{cfg.view_height}×{cfg.view_width}")

    loader = DataLoader(dataset, batch_size=cfg.batch_size,
                        shuffle=True, num_workers=2, pin_memory=False)
    print(f"Train samples: {len(dataset)}, batches/epoch: {len(loader)}")

    # Models
    encoder    = ViewEncoder(d_enc=cfg.d_enc).to(device)
    loc_sensor = LocationSensor(d_in=cfg.d_loc_in, d_out=cfg.d_loc).to(device)
    combine    = CombineModule(d_patch=cfg.d_enc, d_loc=cfg.d_loc,
                                d_out=cfg.d_hidden, dropout=cfg.dropout).to(device)
    memory     = AgentMemory(d_in=cfg.d_hidden, d_hidden=cfg.d_hidden).to(device)
    completion = CompletionHead(d_hidden=cfg.d_hidden, n_views=cfg.n_views,
                                 n_channels=3, view_size=cfg.view_height).to(device)
    actor      = Actor(d_hidden=cfg.d_hidden, n_actions=cfg.n_actions).to(device)

    # Count params
    total_params = sum(p.numel() for p in (
        list(encoder.parameters()) + list(loc_sensor.parameters()) +
        list(combine.parameters()) + list(memory.parameters()) +
        list(completion.parameters()) + list(actor.parameters())
    ))
    print(f"Total parameters: {total_params:,}")

    # Logging
    run = init_logging(cfg)

    # ---- Phase 1: Pretrain (T=1) ----
    print(f"\n=== Phase 1: Pretraining ({cfg.pretrain_epochs} epochs, T=1) ===")
    pretrain_params = (list(encoder.parameters()) + list(loc_sensor.parameters()) +
                       list(combine.parameters()) + list(memory.parameters()) +
                       list(completion.parameters()))
    optimizer = torch.optim.Adam(pretrain_params, lr=cfg.lr)
    global_step = pretrain(loader, encoder, loc_sensor, combine, memory,
                           completion, optimizer, cfg, device, run)

    # Save pretrain checkpoint
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "loc_sensor": loc_sensor.state_dict(),
        "combine": combine.state_dict(),
        "memory": memory.state_dict(),
        "completion": completion.state_dict(),
    }, os.path.join(cfg.checkpoint_dir, "pretrained.pt"))
    print("Pretraining complete. Checkpoint saved.")

    # ---- Phase 2: Full training (T=6) ----
    print(f"\n=== Phase 2: Full training ({cfg.n_epochs} epochs, T={cfg.T}) ===")
    all_params = (list(encoder.parameters()) + list(loc_sensor.parameters()) +
                  list(combine.parameters()) + list(memory.parameters()) +
                  list(completion.parameters()) + list(actor.parameters()))
    optimizer = torch.optim.Adam(all_params, lr=cfg.lr)

    train_full(loader, encoder, loc_sensor, combine, memory, completion,
               actor, optimizer, cfg, device, run, global_step=global_step)

    print("\nTraining complete.")
    if run:
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory with SUN360 split .h5 files")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of full-training epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    args = parser.parse_args()
    main(args)
