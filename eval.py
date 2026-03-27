"""
Evaluation script: per-pixel MSE × 1000 on the validation/test set.

Compares:
  - Learned policy (Actor)
  - RandomPolicy
  - LargeActionPolicy

All results are saved to results/:
  results/eval_mse_curve.png  — MSE vs timestep plot
  results/eval_metrics.json   — numerical results

Usage:
  uv run python eval.py --checkpoint checkpoints/ckpt_epoch2000.pt
  uv run python eval.py --checkpoint checkpoints/ckpt_epoch2000.pt --split test
  uv run python eval.py --checkpoint checkpoints/ckpt_epoch2000.pt --wandb
"""
import argparse
import json
import os
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from data.sun360 import SUN360Dataset
from data.utils import circ_shift_viewgrid, get_view, paste_observed, step_position
from models.actor import Actor
from models.baselines import LargeActionPolicy, RandomPolicy
from models.combine import CombineModule
from models.completion import CompletionHead
from models.encoder import ViewEncoder
from models.location import LocationSensor
from models.memory import AgentMemory
from train import load_checkpoint
from utils.logging import init_logging


def eval_policy(loader, encoder, loc_sensor, combine, memory, completion,
                policy, config: Config, device: torch.device,
                policy_name: str = "policy",
                seed: int = 42) -> List[float]:
    """
    Evaluate a policy and return MSE×1000 at each timestep.

    Uses a fixed seed so all policies start from identical positions,
    making the per-timestep comparison fair.

    Returns:
        mse_per_step: list of T floats (MSE×1000 averaged over dataset)
    """
    encoder.eval(); loc_sensor.eval(); combine.eval()
    memory.eval(); completion.eval()

    n_elev, n_azim = config.n_elev, config.n_azim
    T = config.T
    cumulative_mse = [0.0] * T
    n_batches = 0

    # Fixed RNG so every policy sees the same starting positions
    rng = torch.Generator()
    rng.manual_seed(seed)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            B = batch.shape[0]

            elev_cur = torch.randint(0, n_elev, (B,), generator=rng).to(device)
            azim_cur = torch.randint(0, n_azim, (B,), generator=rng).to(device)
            delta_0 = azim_cur[0].item()

            h, c = memory.init_hidden(B, device)
            shared_observed = {}
            d_elev_prev = torch.zeros(B, device=device)
            d_azim_prev = torch.zeros(B, device=device)
            rel_elev = torch.zeros(B, device=device)
            rel_azim = torch.zeros(B, device=device)

            for t in range(T):
                e, a = int(elev_cur[0].item()), int(azim_cur[0].item())
                x_t = get_view(batch, e, a, n_azim=n_azim).to(device)
                shared_observed[(e, a)] = x_t

                p_t = torch.stack([
                    elev_cur.float() / max(n_elev - 1, 1),
                    d_elev_prev.float() / max(n_elev - 1, 1),
                    d_azim_prev.float() / n_azim,
                ], dim=1).to(device)

                patch_feat = encoder(x_t)
                loc_feat   = loc_sensor(p_t)
                f_t        = combine(patch_feat, loc_feat)
                a_t, (h, c) = memory(f_t, (h, c))

                recon_t = completion(a_t)
                recon_t = circ_shift_viewgrid(recon_t, int(delta_0), n_elev, n_azim)
                recon_t = paste_observed(recon_t, shared_observed, n_azim)

                mse_t = F.mse_loss(recon_t, batch).item() * 1000
                cumulative_mse[t] += mse_t

                if t < T - 1:
                    if isinstance(policy, Actor):
                        rel_pos = torch.stack([
                            rel_elev.float() / max(n_elev - 1, 1),
                            rel_azim.float() / n_azim,
                        ], dim=1).to(device)
                        time_frac = torch.full((B, 1), t / T,
                                               dtype=torch.float32, device=device)
                        logits = policy(a_t, rel_pos, time_frac)
                        action, _, _ = policy.get_action(logits, deterministic=True)
                    else:
                        action = policy.get_action(B, device)

                    act_idx = action[0].item()
                    de, da = config.action_deltas[act_idx]
                    new_e, new_a = step_position(e, a, de, da, n_elev, n_azim)
                    d_elev_prev = torch.full((B,), float(de), device=device)
                    d_azim_prev = torch.full((B,), float(da), device=device)
                    rel_elev = rel_elev + float(de)
                    rel_azim = rel_azim + float(da)
                    elev_cur = torch.full((B,), new_e, dtype=torch.long, device=device)
                    azim_cur = torch.full((B,), new_a, dtype=torch.long, device=device)

            n_batches += 1

    return [v / n_batches for v in cumulative_mse]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory with SUN360 split .h5 files")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--wandb", action="store_true",
                        help="Log eval results to wandb")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Wandb run name")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g. 'cuda:2'")
    args = parser.parse_args()

    cfg = Config()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    cfg.batch_size = args.batch_size
    cfg.use_wandb = args.wandb

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    dataset = SUN360Dataset(cfg.data_dir, split=args.split)
    cfg.n_elev = dataset.n_elev; cfg.n_azim = dataset.n_azim; cfg.n_views = dataset.n_views
    loader = DataLoader(dataset, batch_size=cfg.batch_size,
                        shuffle=False, num_workers=2)

    encoder    = ViewEncoder(d_enc=cfg.d_enc).to(device)
    loc_sensor = LocationSensor(d_in=cfg.d_loc_in, d_out=cfg.d_loc).to(device)
    combine    = CombineModule(d_patch=cfg.d_enc, d_loc=cfg.d_loc,
                                d_out=cfg.d_hidden, dropout=cfg.dropout).to(device)
    memory     = AgentMemory(d_in=cfg.d_hidden, d_hidden=cfg.d_hidden).to(device)
    completion = CompletionHead(d_hidden=cfg.d_hidden, n_views=cfg.n_views,
                                 view_size=cfg.view_height).to(device)
    actor      = Actor(d_hidden=cfg.d_hidden, n_actions=cfg.n_actions).to(device)

    load_checkpoint(args.checkpoint, encoder, loc_sensor, combine,
                    memory, completion, actor)
    print(f"Loaded checkpoint: {args.checkpoint}")

    random_policy = RandomPolicy(n_actions=cfg.n_actions)
    large_policy  = LargeActionPolicy(action_deltas=cfg.action_deltas)

    # Init logging
    run_name = args.run_name or f"eval_{args.split}_{os.path.basename(args.checkpoint)}"
    run = None
    if cfg.use_wandb:
        cfg_copy = Config()
        cfg_copy.use_wandb = True
        run = init_logging(cfg_copy, run_name=run_name)

    results = {}
    for name, policy in [("ours", actor),
                          ("random", random_policy),
                          ("large-action", large_policy)]:
        print(f"Evaluating: {name} ...")
        mse_curve = eval_policy(loader, encoder, loc_sensor, combine, memory,
                                completion, policy, cfg, device, name, seed=42)
        results[name] = mse_curve
        print(f"  Final MSE×1000: {mse_curve[-1]:.2f}")

    # Print table
    print(f"\n{'Method':<15} {'MSE×1000 (final)':>18}")
    print("-" * 35)
    for name, curve in results.items():
        print(f"{name:<15} {curve[-1]:>18.2f}")

    os.makedirs("results", exist_ok=True)

    # Save JSON metrics
    metrics_path = "results/eval_metrics.json"
    metrics = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "data_dir": cfg.data_dir,
        "n_samples": len(dataset),
        "results": {name: {"curve": curve, "final_mse": curve[-1]}
                    for name, curve in results.items()}
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {metrics_path}")

    # Plot MSE vs timestep
    fig, ax = plt.subplots(figsize=(7, 4))
    # Paper reference values (SUN360, from Fig 4)
    paper_ref = {
        "ours (paper)":        [None, None, None, None, None, 23.16],
        "random (paper)":      [None, None, None, None, None, 31.88],
        "large-action (paper)":[None, None, None, None, None, 30.76],
    }
    colors = {"ours": "tab:blue", "random": "tab:orange", "large-action": "tab:green"}
    for name, curve in results.items():
        ax.plot(range(1, cfg.T + 1), curve, marker='o',
                label=name, color=colors.get(name))
    # Mark paper reference as dashed horizontal lines at t=T
    for name, vals in paper_ref.items():
        final = vals[-1]
        base_name = name.replace(" (paper)", "")
        ax.axhline(final, linestyle="--", alpha=0.5,
                   color=colors.get(base_name, "gray"),
                   label=f"{name}: {final:.1f}")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("MSE × 1000")
    ax.set_title(f"Active Completion: MSE vs. Time (SUN360, {args.split})")
    ax.legend(fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plot_path = "results/eval_mse_curve.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")
    plt.close()

    # Log to wandb
    if run is not None:
        import wandb
        wandb_metrics = {}
        for name, curve in results.items():
            for t, val in enumerate(curve):
                wandb_metrics[f"eval/{name}/mse_t{t+1}"] = val
            wandb_metrics[f"eval/{name}/final_mse"] = curve[-1]
        run.log(wandb_metrics)
        run.log({"eval/mse_curve": wandb.Image(plot_path)})
        run.finish()
        print("Logged to wandb.")


if __name__ == "__main__":
    main()
