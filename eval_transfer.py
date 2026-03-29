"""
Policy Transfer Evaluation (Fig 5).

Evaluates active categorization accuracy using the completion policy
(trained without labels) to guide exploration, then classifying with
a linear head trained on frozen hidden representations — per timestep.

Compares (Fig 5 format: accuracy vs. time):
  - 1-view        : single random view, no exploration (horizontal baseline)
  - random-policy : random action selection
  - large-action  : always take largest-magnitude action
  - ours          : completion policy (unsupervised transfer)

Usage:
  uv run python eval_transfer.py \\
    --checkpoint checkpoints/ckpt_epoch2000.pt \\
    --data-dir  data/indoor360_torchfeed \\
    --wandb

Note:
  Requires a dataset with at least 2 unique category labels (labs field in HDF5).
  Run data/patch_labels.py first to assign empty/full labels to indoor360.
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from data.sun360 import SUN360Dataset
from data.utils import get_view, step_position
from models.actor import Actor
from models.baselines import LargeActionPolicy, RandomPolicy
from models.classifier import CategorizationHead
from models.combine import CombineModule
from models.completion import CompletionHead
from models.encoder import ViewEncoder
from models.location import LocationSensor
from models.memory import AgentMemory
from train import load_checkpoint
from utils.logging import init_logging


# -------------------------------------------------------------------------
# Per-timestep feature extraction
# -------------------------------------------------------------------------

def extract_features_per_step(
    loader,
    encoder, loc_sensor, combine, memory,
    policy,
    config: Config,
    device: torch.device,
    seed: int = 42,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Run active exploration and collect hidden states at every timestep.

    Returns:
        feats_per_t: list of T tensors, each (N, d_hidden)
        labels:      (N,) int64
    """
    encoder.eval(); loc_sensor.eval(); combine.eval(); memory.eval()

    n_elev, n_azim = config.n_elev, config.n_azim
    T = config.T

    rng = torch.Generator()
    rng.manual_seed(seed)

    # Accumulators: one list per timestep
    feats_per_t: List[List[torch.Tensor]] = [[] for _ in range(T)]
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 2:
                batch, labels = batch_data
            else:
                batch = batch_data
                labels = torch.zeros(batch.shape[0], dtype=torch.long)

            batch = batch.to(device)
            B = batch.shape[0]

            elev_cur = torch.randint(0, n_elev, (B,), generator=rng)
            azim_cur = torch.randint(0, n_azim, (B,), generator=rng)

            h, c = memory.init_hidden(B, device)
            d_elev_prev = torch.zeros(B, device=device)
            d_azim_prev = torch.zeros(B, device=device)
            rel_elev    = torch.zeros(B, device=device)
            rel_azim    = torch.zeros(B, device=device)

            for t in range(T):
                e, a = int(elev_cur[0].item()), int(azim_cur[0].item())
                x_t = get_view(batch, e, a, n_azim=n_azim).to(device)

                p_t = torch.stack([
                    elev_cur.float().to(device) / max(n_elev - 1, 1),
                    d_elev_prev.float() / max(n_elev - 1, 1),
                    d_azim_prev.float() / n_azim,
                ], dim=1)

                patch_feat = encoder(x_t)
                loc_feat   = loc_sensor(p_t)
                f_t        = combine(patch_feat, loc_feat)
                a_t, (h, c) = memory(f_t, (h, c))

                # Record h_t for this timestep
                feats_per_t[t].append(h.squeeze(0).cpu())   # (B, d_hidden)

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
                    elev_cur = torch.full((B,), new_e, dtype=torch.long)
                    azim_cur = torch.full((B,), new_a, dtype=torch.long)

            all_labels.append(labels)

    labels_cat = torch.cat(all_labels, dim=0)                    # (N,)
    feats_cat  = [torch.cat(feats_per_t[t], dim=0) for t in range(T)]  # T x (N, d)
    return feats_cat, labels_cat


# -------------------------------------------------------------------------
# Classifier training / evaluation
# -------------------------------------------------------------------------

def train_classifier(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    n_classes: int,
    d_hidden: int,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-2,
) -> CategorizationHead:
    clf = CategorizationHead(d_hidden=d_hidden, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    ds = TensorDataset(train_feats.to(device), train_labels.to(device))
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    clf.train()
    for _ in range(epochs):
        for feats, labs in loader:
            logits = clf(feats)
            loss = F.cross_entropy(logits, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return clf


def evaluate_classifier(
    clf: CategorizationHead,
    feats: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    clf.eval()
    with torch.no_grad():
        preds = clf(feats.to(device)).argmax(dim=1).cpu()
    return (preds == labels).float().mean().item() * 100.0


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     type=str, required=True)
    parser.add_argument("--data-dir",       type=str, default=None)
    parser.add_argument("--batch-size",     type=int, default=32)
    parser.add_argument("--device",         type=str, default=None)
    parser.add_argument("--wandb",          action="store_true")
    parser.add_argument("--run-name",       type=str, default=None)
    parser.add_argument("--out-json",       type=str, default="results/transfer_metrics.json",
                        help="Path to save per-timestep accuracy results JSON")
    parser.add_argument("--dataset-label",  type=str, default=None,
                        help="Human-readable dataset name stored in the JSON (used by plot_fig5.py)")
    args = parser.parse_args()

    cfg = Config()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    cfg.batch_size = args.batch_size
    cfg.use_wandb  = args.wandb

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load datasets
    import h5py
    train_dataset = SUN360Dataset(cfg.data_dir, split="train")
    val_dataset   = SUN360Dataset(cfg.data_dir, split="val")
    cfg.n_elev  = train_dataset.n_elev
    cfg.n_azim  = train_dataset.n_azim
    cfg.n_views = train_dataset.n_views

    trn_h5 = os.path.join(cfg.data_dir, "pixels_trn_torchfeed.h5")
    val_h5 = os.path.join(cfg.data_dir, "pixels_val_torchfeed.h5")
    with h5py.File(trn_h5, "r") as f:
        trn_labels = torch.from_numpy(f["labs"][:].flatten().astype(np.int64))
    with h5py.File(val_h5, "r") as f:
        val_labels = torch.from_numpy(f["labs"][:].flatten().astype(np.int64))

    n_classes = int(max(trn_labels.max().item(), val_labels.max().item()) + 1)
    print(f"Classes: {n_classes}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    if n_classes < 2:
        print("\nWARNING: Only 1 class in dataset. Run data/patch_labels.py first.")
        return

    class LabeledDataset(torch.utils.data.Dataset):
        def __init__(self, viewgrids, labels):
            self.viewgrids = viewgrids
            self.labels    = labels
        def __len__(self): return len(self.viewgrids)
        def __getitem__(self, i): return self.viewgrids[i], self.labels[i]

    trn_ds     = LabeledDataset(train_dataset.viewgrids, trn_labels)
    val_ds     = LabeledDataset(val_dataset.viewgrids,   val_labels)
    trn_loader = DataLoader(trn_ds, batch_size=cfg.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # Load models
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

    run = None
    if cfg.use_wandb:
        run_name = args.run_name or f"transfer_{os.path.basename(args.checkpoint)}"
        run = init_logging(cfg, run_name=run_name)

    T = cfg.T
    policies = [("ours", actor), ("random", random_policy), ("large-action", large_policy)]
    results: Dict[str, List[float]] = {}

    for name, policy in policies:
        print(f"\nExtracting per-timestep features: {name} ...")
        trn_feats_t, trn_labs = extract_features_per_step(
            trn_loader, encoder, loc_sensor, combine, memory, policy, cfg, device, seed=42)
        val_feats_t, val_labs = extract_features_per_step(
            val_loader, encoder, loc_sensor, combine, memory, policy, cfg, device, seed=42)

        accs = []
        for t in range(T):
            clf = train_classifier(trn_feats_t[t], trn_labs, n_classes,
                                   cfg.d_hidden, device, epochs=200, lr=1e-2)
            acc = evaluate_classifier(clf, val_feats_t[t], val_labs, device)
            accs.append(acc)
            print(f"  t={t+1}: val_acc={acc:.1f}%")
        results[name] = accs

    # 1-view: constant accuracy from a single random view (t=1 of random policy)
    print("\nExtracting 1-view features ...")
    trn_feats_t, trn_labs = extract_features_per_step(
        trn_loader, encoder, loc_sensor, combine, memory, random_policy, cfg, device, seed=42)
    val_feats_t, val_labs = extract_features_per_step(
        val_loader, encoder, loc_sensor, combine, memory, random_policy, cfg, device, seed=42)
    clf_1v = train_classifier(trn_feats_t[0], trn_labs, n_classes,
                               cfg.d_hidden, device, epochs=200, lr=1e-2)
    acc_1v = evaluate_classifier(clf_1v, val_feats_t[0], val_labs, device)
    results["1-view"] = [acc_1v] * T
    print(f"  1-view accuracy: {acc_1v:.1f}%")

    # Print table
    print(f"\n{'Method':<15}", "  ".join(f"t={t+1}" for t in range(T)))
    print("-" * (15 + 8 * T))
    for name, accs in results.items():
        row = "  ".join(f"{a:6.1f}" for a in accs)
        print(f"{name:<15} {row}")

    os.makedirs("results", exist_ok=True)

    dataset_label = args.dataset_label or os.path.basename(os.path.normpath(cfg.data_dir))
    metrics_path = args.out_json
    with open(metrics_path, "w") as fh:
        json.dump({"checkpoint": args.checkpoint,
                   "dataset": dataset_label,
                   "n_classes": n_classes,
                   "T": T,
                   "results": results}, fh, indent=2)
    print(f"Saved {metrics_path}")

    # Plot — Fig 5 style: accuracy vs. time
    timesteps = list(range(1, T + 1))
    colors = {"ours": "tab:purple", "random": "tab:orange",
              "large-action": "tab:green", "1-view": "tab:gray"}
    styles = {"1-view": "--"}

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, accs in results.items():
        ax.plot(timesteps, accs,
                color=colors.get(name, "tab:blue"),
                linestyle=styles.get(name, "-"),
                marker="o", markersize=4, label=name)
    ax.set_xlabel("time $t$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Active categorization\n(indoor360: empty vs full)")
    ax.set_xticks(timesteps)
    ax.legend(fontsize=8)
    plt.tight_layout()

    plot_path = "results/eval_transfer.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved {plot_path}")

    if run is not None:
        import wandb
        wandb_metrics = {}
        for name, accs in results.items():
            for t, acc in enumerate(accs):
                wandb_metrics[f"transfer/{name}/acc_t{t+1}"] = acc
        run.log(wandb_metrics)
        run.log({"transfer/curve": wandb.Image(plot_path)})
        run.finish()


if __name__ == "__main__":
    main()
