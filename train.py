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
from data.combined import make_combined_loaders
from data.sun360 import SUN360Dataset
from data.utils import circ_shift_viewgrid, get_view, paste_observed, step_position
from models.actor import Actor
from models.combine import CombineModule
from models.completion import CompletionHead
from models.encoder import ViewEncoder
from models.location import LocationSensor
from models.memory import AgentMemory
from utils.logging import init_logging, log_metrics, log_val_recon
from utils.rewards import LearnedBaseline, compute_reinforce_loss


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
    elev_0 = torch.randint(0, n_elev, (B,), device=device)
    azim_0 = torch.randint(0, n_azim, (B,), device=device)
    delta_0 = azim_0.clone()  # azimuth offset for rotation compensation

    # Current position
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

        # Proprioceptive metadata: [rel_elev_norm, rel_azim_norm, t/T, abs_elev_norm]
        # Matches original location_ipsz = 2+1+1 (rel_pos + time + knownElev)
        p_t = torch.stack([
            rel_elev.float() / max(n_elev - 1, 1),   # cumulative rel elev from start
            rel_azim.float() / n_azim,                # cumulative rel azim from start
            torch.full((B,), t / T, dtype=torch.float32, device=device),  # t/T
            elev_cur.float() / max(n_elev - 1, 1),   # absolute elevation norm
        ], dim=1).to(device)  # (B, 4)

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
        recon_t_shifted = paste_observed(recon_t_shifted, shared_observed, n_azim)
        recon_list.append(recon_t_shifted)

        # --- Act (skip on last step) ---
        if t < T - 1:
            rel_pos = torch.stack([
                rel_elev.float() / max(n_elev - 1, 1),
                rel_azim.float() / n_azim,
            ], dim=1).to(device)  # (B, 2)
            time_frac = torch.full((B, 1), t / T,
                                   dtype=torch.float32, device=device)
            abs_elev_norm = (elev_cur.float() / max(n_elev - 1, 1)).unsqueeze(1).to(device)  # (B, 1)
            logits = actor(a_t, rel_pos, time_frac, abs_elev_norm)  # (B, 14)
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

            elev_cur = torch.full((B,), new_e, dtype=torch.long, device=device)
            azim_cur = torch.full((B,), new_a, dtype=torch.long, device=device)

    return recon_list, log_probs, delta_0


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_losses(recon_list, target, log_probs, baseline_value):
    """
    recon_list:     list of T tensors (B, N_views, C, H, W) — already shifted + pasted
    target:         (B, N_views, C, H, W)  — mean-subtracted
    log_probs:      list of T-1 (B,) tensors
    baseline_value: scalar tensor from LearnedBaseline() (for advantage computation)

    Returns:
        recon_loss:  scalar tensor — sum of MSE over all T steps (for monitoring)
        policy_loss: scalar tensor — REINFORCE loss (for actor optimizer)
        reward:      (B,) tensor  — per-sample -MSE at final step (for baseline training)
    """
    # Reconstruction loss: sum over all timesteps (scalar, for monitoring only in phase 2)
    recon_losses = [F.mse_loss(r, target) for r in recon_list]
    recon_loss = sum(recon_losses)

    # Per-sample reward: -(mean pixel MSE at final step) — shape (B,)
    final_diff = (recon_list[-1].detach() - target.detach())
    reward = -final_diff.pow(2).mean(dim=[1, 2, 3, 4])  # (B,)

    policy_loss = compute_reinforce_loss(log_probs, reward, baseline_value)

    return recon_loss, policy_loss, reward


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def validate(val_loader, encoder, loc_sensor, combine, memory, completion,
             actor, config, device):
    """Run one pass over val set; return avg recon_loss and avg reward."""
    encoder.eval(); loc_sensor.eval(); combine.eval()
    memory.eval(); completion.eval()
    if actor is not None:
        actor.eval()

    total_recon = 0.0
    total_reward = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon_list, _, _ = run_episode(
                batch, encoder, loc_sensor, combine, memory, completion,
                actor=actor, config=config, device=device, T=config.T,
                deterministic=True,
            )
            recon_losses = [F.mse_loss(r, batch) for r in recon_list]
            total_recon  += sum(recon_losses).item()
            total_reward += (-recon_losses[-1]).item()
            n += 1

    encoder.train(); loc_sensor.train(); combine.train()
    memory.train(); completion.train()
    if actor is not None:
        actor.train()

    return total_recon / max(n, 1), total_reward / max(n, 1)


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
               config, device, run, global_step=0, val_loader=None, mean_vg=None):
    """Phase 2: T=6, REINFORCE only (encoder/decoder/memory frozen, actor trains).

    Matches original: finetune_lrMult=0 freezes pretrained modules; only Actor
    and the learned baseline scalar are updated. Baseline trained at 150× actor lr.
    """
    # Freeze all pretrained modules — only actor (and baseline) update in phase 2
    for module in [encoder, loc_sensor, combine, memory, completion]:
        for p in module.parameters():
            p.requires_grad = False

    # Actor-only optimizer with weight decay (matching original weightDecay=0.005)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=config.lr, weight_decay=0.005
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        actor_optimizer, mode="min", factor=0.5,
        patience=200, threshold=0.0002, min_lr=1e-6,
    )

    # Learned scalar baseline (matches original nn.Add(1), trained at 150× actor lr)
    learned_baseline = LearnedBaseline().to(device)
    baseline_optimizer = torch.optim.Adam(
        learned_baseline.parameters(), lr=config.lr * 150
    )

    encoder.eval(); loc_sensor.eval(); combine.eval()
    memory.eval(); completion.eval(); actor.train()

    # Cache one fixed val sample for visualization (grabbed once, reused every epoch)
    _vis_batch = None
    if val_loader is not None and run is not None:
        _vis_batch = next(iter(val_loader))[:1].to(device)  # (1, N_views, C, H, W)

    for epoch in range(config.n_epochs):
        actor.train()
        epoch_policy = 0.0
        epoch_reward = 0.0
        epoch_recon  = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)
            B = batch.shape[0]

            recon_list, log_probs, _ = run_episode(
                batch, encoder, loc_sensor, combine, memory, completion,
                actor=actor, config=config, device=device, T=config.T,
            )

            recon_loss, policy_loss, reward = compute_losses(
                recon_list, batch, log_probs, learned_baseline()
            )

            # Update baseline (MSE against per-sample reward)
            baseline_optimizer.zero_grad()
            baseline_loss = F.mse_loss(
                learned_baseline.value.expand(B), reward.detach()
            )
            baseline_loss.backward()
            baseline_optimizer.step()

            # Update actor (REINFORCE)
            actor_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            actor_optimizer.step()

            epoch_policy += policy_loss.item()
            epoch_reward += reward.mean().item()
            epoch_recon  += recon_loss.item()
            n_batches += 1
            global_step += 1

            if global_step % config.log_every == 0:
                log_metrics({
                    "train/policy_loss": policy_loss.item(),
                    "train/reward":      reward.mean().item(),
                    "train/recon_loss":  recon_loss.item(),
                    "train/baseline":    learned_baseline.value.item(),
                }, step=global_step, run=run)

        # Epoch summary + val
        if (epoch + 1) % 10 == 0:
            metrics = {
                "train/epoch_policy": epoch_policy / n_batches,
                "train/epoch_reward": epoch_reward / n_batches,
                "train/epoch_recon":  epoch_recon  / n_batches,
                "train/epoch":        epoch + 1,
            }
            if val_loader is not None:
                val_recon, val_reward = validate(
                    val_loader, encoder, loc_sensor, combine, memory,
                    completion, actor, config, device,
                )
                # validate() restores .train() on all modules; re-freeze frozen ones
                encoder.eval(); loc_sensor.eval(); combine.eval()
                memory.eval(); completion.eval()
                metrics["val/recon_loss"] = val_recon
                metrics["val/reward"]     = val_reward
                scheduler.step(val_recon)

                # Log reconstruction images for the fixed val sample
                if _vis_batch is not None:
                    actor.eval()
                    with torch.no_grad():
                        vis_recon, _, _ = run_episode(
                            _vis_batch, encoder, loc_sensor, combine, memory,
                            completion, actor=actor, config=config, device=device,
                            T=config.T, deterministic=True,
                        )
                    actor.train()
                    log_val_recon(run, global_step, vis_recon, _vis_batch,
                                  config.n_elev, config.n_azim, mean_vg=mean_vg)
            log_metrics(metrics, step=global_step, run=run)

        # Checkpoint
        if (epoch + 1) % config.save_every == 0:
            _save_checkpoint(epoch + 1, encoder, loc_sensor, combine,
                             memory, completion, actor, config,
                             actor_optimizer=actor_optimizer,
                             baseline=learned_baseline,
                             baseline_optimizer=baseline_optimizer)

    return global_step


def _save_checkpoint(epoch, encoder, loc_sensor, combine, memory,
                     completion, actor, config,
                     actor_optimizer=None, baseline=None, baseline_optimizer=None):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"ckpt_epoch{epoch:04d}.pt")
    state = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "loc_sensor": loc_sensor.state_dict(),
        "combine": combine.state_dict(),
        "memory": memory.state_dict(),
        "completion": completion.state_dict(),
        "actor": actor.state_dict(),
    }
    if actor_optimizer is not None:
        state["actor_optimizer"] = actor_optimizer.state_dict()
    if baseline is not None:
        state["baseline"] = baseline.state_dict()
    if baseline_optimizer is not None:
        state["baseline_optimizer"] = baseline_optimizer.state_dict()
    torch.save(state, path)
    print(f"[checkpoint] saved to {path}")


def load_checkpoint(path, encoder, loc_sensor, combine, memory,
                    completion, actor, actor_optimizer=None,
                    baseline=None, baseline_optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    encoder.load_state_dict(ckpt["encoder"])
    loc_sensor.load_state_dict(ckpt["loc_sensor"])
    combine.load_state_dict(ckpt["combine"])
    memory.load_state_dict(ckpt["memory"])
    completion.load_state_dict(ckpt["completion"])
    actor.load_state_dict(ckpt["actor"])
    if actor_optimizer and "actor_optimizer" in ckpt:
        actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
    if baseline and "baseline" in ckpt:
        baseline.load_state_dict(ckpt["baseline"])
    if baseline_optimizer and "baseline_optimizer" in ckpt:
        baseline_optimizer.load_state_dict(ckpt["baseline_optimizer"])
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
            cfg.use_wandb = args.wandb
        if hasattr(args, 'batch_size') and args.batch_size:
            cfg.batch_size = args.batch_size

    device_str = args.device if (args and hasattr(args, 'device') and args.device) else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_str)
    print(f"Device: {device}")

    # Dataset
    extra_data_dir = (args.extra_data_dir
                      if (args and hasattr(args, 'extra_data_dir') and args.extra_data_dir)
                      else None)

    val_dataset = None  # set below in single-dataset mode (provides mean_viewgrid)
    if extra_data_dir:
        # Combined mode: indoor360 (4:3:3 resplit) + sun360
        print(f"Combined mode: {cfg.data_dir} (4:3:3 resplit) + {extra_data_dir}")
        loader, val_loader, n_elev, n_azim = make_combined_loaders(
            indoor_dir=cfg.data_dir,
            sun360_dir=extra_data_dir,
            batch_size=cfg.batch_size,
        )
        cfg.n_elev = n_elev; cfg.n_azim = n_azim
        cfg.n_views = n_elev * n_azim
        cfg.view_height = 32; cfg.view_width = 32
    else:
        print(f"Loading dataset from {cfg.data_dir} ...")
        dataset = SUN360Dataset(cfg.data_dir, split="train", mean_subtract=True)
        cfg.n_elev  = dataset.n_elev
        cfg.n_azim  = dataset.n_azim
        cfg.n_views = dataset.n_views
        cfg.view_height = dataset.view_H
        cfg.view_width  = dataset.view_W
        loader = DataLoader(dataset, batch_size=cfg.batch_size,
                            shuffle=True, num_workers=2, pin_memory=False)
        print(f"Train samples: {len(dataset)}, batches/epoch: {len(loader)}")
        val_dataset = SUN360Dataset(cfg.data_dir, split="val", mean_subtract=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                                shuffle=False, num_workers=2, pin_memory=False)
        print(f"Val samples: {len(val_dataset)}")

    print(f"Grid: {cfg.n_elev}×{cfg.n_azim} = {cfg.n_views} views, each 3×{cfg.view_height}×{cfg.view_width}")
    print(f"Train batches/epoch: {len(loader)}  Val batches: {len(val_loader)}")

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
    run_name = args.run_name if (args and hasattr(args, 'run_name') and args.run_name) else None
    run = init_logging(cfg, run_name=run_name)

    # ---- Phase 1: Pretrain (T=1) ----
    print(f"\n=== Phase 1: Pretraining ({cfg.pretrain_epochs} epochs, T=1) ===")
    pretrain_params = (list(encoder.parameters()) + list(loc_sensor.parameters()) +
                       list(combine.parameters()) + list(memory.parameters()) +
                       list(completion.parameters()))
    optimizer = torch.optim.Adam(pretrain_params, lr=cfg.lr, weight_decay=0.005)
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

    mean_vg = getattr(val_dataset, "mean_viewgrid", None)  # (N_views, C, H, W) or None
    train_full(loader, encoder, loc_sensor, combine, memory, completion,
               actor, cfg, device, run, global_step=global_step,
               val_loader=val_loader, mean_vg=mean_vg)

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
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g. 'cuda:2' (default: auto)")
    parser.add_argument("--extra-data-dir", type=str, default=None,
                        help="Second dataset dir; triggers combined mode: "
                             "--data-dir is indoor360 (4:3:3 resplit), "
                             "--extra-data-dir is SUN360 (existing splits)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Wandb run name (auto-generated if omitted)")
    args = parser.parse_args()
    main(args)
