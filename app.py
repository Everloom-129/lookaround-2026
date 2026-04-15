"""
Interactive visualization app for lookaround_2026.

Two modes:
  Offline — browse pre-computed results and val panoramas
  Live    — step through the model episode sample-by-sample, then run full eval

Launch:
  uv run python app.py
  uv run python app.py --port 7861 --device cuda:2
"""
import argparse
import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import gradio as gr

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from data.sun360 import SUN360Dataset
from data.utils import circ_shift_viewgrid, get_view, paste_observed, step_position
from models.actor import Actor
from models.baselines import LargeActionPolicy, RandomPolicy
from models.classifier import CategorizationHead
from models.combine import CombineModule
from models.completion import CompletionHead
from models.encoder import ViewEncoder
from models.location import LocationSensor
from models.memory import AgentMemory
from train import load_checkpoint


# ── CLI args (parsed before Gradio launch) ────────────────────────────────────
_cli = argparse.ArgumentParser(add_help=False)
_cli.add_argument("--checkpoint", default="checkpoints/ckpt_epoch2000.pt")
_cli.add_argument("--data-dir",   default="data/indoor360_torchfeed")
_cli.add_argument("--device",     default=None)
_cli.add_argument("--port",       type=int, default=7860)
_cli.add_argument("--share",      action="store_true")
_args, _ = _cli.parse_known_args()

CHECKPOINT = _args.checkpoint
DATA_DIR   = _args.data_dir
PORT       = _args.port
SHARE      = _args.share

if _args.device:
    DEVICE = torch.device(_args.device)
else:
    DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# ── Global singletons (loaded once) ───────────────────────────────────────────
cfg          : Config               = None
val_dataset  : SUN360Dataset        = None
encoder      : ViewEncoder          = None
loc_sensor   : LocationSensor       = None
combine      : CombineModule        = None
memory       : AgentMemory          = None
completion   : CompletionHead       = None
actor        : Actor                = None
rand_policy  : RandomPolicy         = None
large_policy : LargeActionPolicy    = None
_load_error  : str                  = ""


def load_model_and_data() -> str:
    """Load model + val dataset into globals. Returns status string."""
    global cfg, val_dataset, encoder, loc_sensor, combine
    global memory, completion, actor, rand_policy, large_policy, _load_error

    try:
        cfg = Config()
        cfg.data_dir = DATA_DIR

        val_dataset = SUN360Dataset(DATA_DIR, split="val", mean_subtract=True)
        cfg.n_elev  = val_dataset.n_elev
        cfg.n_azim  = val_dataset.n_azim
        cfg.n_views = val_dataset.n_views

        encoder    = ViewEncoder(d_enc=cfg.d_enc).to(DEVICE)
        loc_sensor = LocationSensor(d_in=cfg.d_loc_in, d_out=cfg.d_loc).to(DEVICE)
        combine    = CombineModule(d_patch=cfg.d_enc, d_loc=cfg.d_loc,
                                   d_out=cfg.d_hidden, dropout=0.0).to(DEVICE)
        memory     = AgentMemory(d_in=cfg.d_hidden, d_hidden=cfg.d_hidden).to(DEVICE)
        completion = CompletionHead(d_hidden=cfg.d_hidden, n_views=cfg.n_views,
                                    view_size=cfg.view_height).to(DEVICE)
        actor      = Actor(d_hidden=cfg.d_hidden, n_actions=cfg.n_actions).to(DEVICE)

        load_checkpoint(CHECKPOINT, encoder, loc_sensor, combine,
                        memory, completion, actor)
        for m in [encoder, loc_sensor, combine, memory, completion, actor]:
            m.eval()

        rand_policy  = RandomPolicy(n_actions=cfg.n_actions)
        large_policy = LargeActionPolicy(action_deltas=cfg.action_deltas)

        msg = (f"Loaded checkpoint: {CHECKPOINT}\n"
               f"Device: {DEVICE} | Val set: {len(val_dataset)} samples "
               f"({cfg.n_elev}×{cfg.n_azim} grid, T={cfg.T})")
        print(msg)
        return msg
    except Exception as e:
        _load_error = str(e)
        print(f"[load_model_and_data] ERROR: {e}")
        return f"ERROR: {e}"


# ── Visualization helpers ──────────────────────────────────────────────────────
GRID_SCALE = 5   # pixel upscale for viewgrid display
PAD        = 3   # border pixels between views


def viewgrid_to_pil(
    vg_np: np.ndarray,
    n_elev: int,
    n_azim: int,
    highlight: Optional[Tuple[int, int]] = None,    # red border — current pos
    visited: Optional[List[Tuple[int, int]]] = None, # gold border — previous
    scale: int = GRID_SCALE,
) -> Image.Image:
    """
    Render a viewgrid as a PIL image.

    Args:
        vg_np:    (N_views, C, H, W) float32 in [0, 1]
        highlight: (elev, azim) — current position (red border)
        visited:   ordered list of (elev, azim) visited so far (gold border + step #)
    """
    V, C, H, W = vg_np.shape
    visited     = list(visited) if visited else []
    visited_set = set(visited)

    # Canvas dimensions
    rows, cols = n_elev, n_azim
    ch = rows * H + (rows + 1) * PAD
    cw = cols * W + (cols + 1) * PAD
    canvas = np.full((ch, cw, 3), 25, dtype=np.uint8)   # dark background

    for e in range(rows):
        for a in range(cols):
            flat = e * n_azim + a
            patch = vg_np[flat]                               # (C, H, W)
            patch_u8 = (np.clip(patch.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)

            y0 = PAD + e * (H + PAD)
            x0 = PAD + a * (W + PAD)
            canvas[y0:y0 + H, x0:x0 + W] = patch_u8

            # Gold border for visited cells
            if (e, a) in visited_set:
                canvas[y0 - PAD:y0, x0 - PAD:x0 + W + PAD]             = [255, 210, 0]
                canvas[y0 + H:y0 + H + PAD, x0 - PAD:x0 + W + PAD]     = [255, 210, 0]
                canvas[y0:y0 + H, x0 - PAD:x0]                          = [255, 210, 0]
                canvas[y0:y0 + H, x0 + W:x0 + W + PAD]                  = [255, 210, 0]

            # Red border for current position (overwrites gold if same)
            if highlight is not None and (e, a) == highlight:
                canvas[y0 - PAD:y0, x0 - PAD:x0 + W + PAD]             = [255, 60, 60]
                canvas[y0 + H:y0 + H + PAD, x0 - PAD:x0 + W + PAD]     = [255, 60, 60]
                canvas[y0:y0 + H, x0 - PAD:x0]                          = [255, 60, 60]
                canvas[y0:y0 + H, x0 + W:x0 + W + PAD]                  = [255, 60, 60]

    img = Image.fromarray(canvas)
    return img.resize((cw * scale, ch * scale), Image.NEAREST)


def trajectory_to_pil(
    visited: List[Tuple[int, int]],
    current: Optional[Tuple[int, int]],
    n_elev: int,
    n_azim: int,
) -> Image.Image:
    """Draw the navigation grid with step numbers and current position."""
    fig, ax = plt.subplots(figsize=(n_azim * 0.72, n_elev * 0.72 + 0.3))
    ax.set_xlim(0, n_azim)
    ax.set_ylim(0, n_elev)
    ax.set_aspect("equal")
    ax.set_xticks(range(n_azim + 1))
    ax.set_yticks(range(n_elev + 1))
    ax.set_xticklabels([str(a) for a in range(n_azim + 1)], fontsize=7)
    ax.set_yticklabels([str(e) for e in range(n_elev + 1)], fontsize=7)
    ax.set_xlabel("Azimuth", fontsize=8)
    ax.set_ylabel("Elevation", fontsize=8)
    ax.set_title("Trajectory", fontsize=9)
    ax.grid(True, linewidth=0.5, color="gray", alpha=0.4)

    visited_set = set(visited)
    for e in range(n_elev):
        for a in range(n_azim):
            # y goes bottom-up: elev 0 at top → flip
            y = n_elev - 1 - e
            x = a
            if current is not None and (e, a) == current:
                color = "#ff3c3c"
            elif (e, a) in visited_set:
                idx = visited.index((e, a)) + 1
                color = "#ffd700"
            else:
                color = "#222222"
            rect = plt.Rectangle((x, y), 1, 1, facecolor=color,
                                  edgecolor="#444444", linewidth=0.8)
            ax.add_patch(rect)
            if (e, a) in visited_set:
                idx = visited.index((e, a)) + 1
                ax.text(x + 0.5, y + 0.5, str(idx),
                        ha="center", va="center", fontsize=8, color="black",
                        fontweight="bold")

    # Draw arrow trajectory
    for i in range(len(visited) - 1):
        e0, a0 = visited[i]
        e1, a1 = visited[i + 1]
        y0 = n_elev - 1 - e0 + 0.5
        x0 = a0 + 0.5
        y1 = n_elev - 1 - e1 + 0.5
        x1 = a1 + 0.5
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="white",
                                   lw=1.2, connectionstyle="arc3,rad=0.1"))

    legend = [
        mpatches.Patch(color="#ff3c3c", label="current"),
        mpatches.Patch(color="#ffd700", label="visited"),
        mpatches.Patch(color="#222222", label="unvisited"),
    ]
    ax.legend(handles=legend, fontsize=6, loc="upper right",
              framealpha=0.6, labelcolor="white",
              facecolor="#111111", edgecolor="none")

    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, facecolor="#111111")
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


def mse_curve_to_pil(mse_history: List[float], T: int) -> Image.Image:
    """Line chart of MSE × 1000 at each timestep so far."""
    fig, ax = plt.subplots(figsize=(4, 2.4))
    ts = list(range(1, len(mse_history) + 1))
    ax.plot(ts, mse_history, marker="o", color="#5b9cf6", linewidth=2, markersize=6)
    ax.set_xlim(0.5, T + 0.5)
    ax.set_xticks(range(1, T + 1))
    ax.set_xlabel("Timestep t", fontsize=9)
    ax.set_ylabel("MSE × 1000", fontsize=9)
    ax.set_title("Reconstruction MSE vs. time", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


def results_table_html(data: dict, key: str) -> str:
    """Render a dict of {method: [values]} as an HTML table."""
    if key not in data:
        return "<p>No data</p>"
    rows = data[key]
    methods = list(rows.keys())
    T = max(len(v) for v in rows.values())
    header = "<tr><th>Method</th>" + "".join(f"<th>t={t}</th>" for t in range(1, T + 1)) + "</tr>"
    body = ""
    for m, vals in rows.items():
        cells = "".join(f"<td>{v:.1f}</td>" for v in vals)
        body += f"<tr><td><b>{m}</b></td>{cells}</tr>"
    return f"<table>{header}{body}</table>"


# ── Episode state helpers ──────────────────────────────────────────────────────
def new_episode_state(sample_idx: int, policy_name: str,
                      elev_start: int, azim_start: int) -> dict:
    vg_tensor = val_dataset[sample_idx].unsqueeze(0).to(DEVICE)   # (1, N, C, H, W)
    h, c = memory.init_hidden(1, DEVICE)
    return {
        "vg_tensor"    : vg_tensor,
        "vg_np"        : val_dataset[sample_idx].numpy(),          # (N, C, H, W)
        "step"         : 0,
        "elev_cur"     : elev_start,
        "azim_cur"     : azim_start,
        "delta_0"      : azim_start,
        "h"            : h,
        "c"            : c,
        "d_elev_prev"  : 0,
        "d_azim_prev"  : 0,
        "rel_elev"     : 0,
        "rel_azim"     : 0,
        "observed"     : {},
        "mse_history"  : [],
        "visited"      : [],
        "policy_name"  : policy_name,
        "recon_np"     : None,
        "done"         : False,
    }


def pick_policy(policy_name: str):
    return {"ours": actor, "random": rand_policy, "large-action": large_policy}[policy_name]


def run_one_step(state: dict) -> dict:
    """Advance the episode by one step. Mutates and returns state."""
    if state["done"]:
        return state

    n_elev = cfg.n_elev
    n_azim = cfg.n_azim
    T      = cfg.T
    B      = 1

    e, a   = state["elev_cur"], state["azim_cur"]
    vg     = state["vg_tensor"]
    h, c   = state["h"], state["c"]
    t      = state["step"]

    state["visited"].append((e, a))

    with torch.no_grad():
        x_t = get_view(vg, e, a, n_azim=n_azim).to(DEVICE)
        state["observed"][(e, a)] = x_t

        rel_elev_n = torch.tensor([[state["rel_elev"] / max(n_elev - 1, 1)]], dtype=torch.float32, device=DEVICE)
        rel_azim_n = torch.tensor([[state["rel_azim"] / n_azim]], dtype=torch.float32, device=DEVICE)
        time_n     = torch.tensor([[t / T]], dtype=torch.float32, device=DEVICE)
        elev_n     = torch.tensor([[e / max(n_elev - 1, 1)]], dtype=torch.float32, device=DEVICE)
        p_t = torch.cat([rel_elev_n, rel_azim_n, time_n, elev_n], dim=1)

        patch_feat = encoder(x_t)
        loc_feat   = loc_sensor(p_t)
        f_t        = combine(patch_feat, loc_feat)
        a_t, (h, c) = memory(f_t, (h, c))

        recon = completion(a_t)
        recon = circ_shift_viewgrid(recon, state["delta_0"], n_elev, n_azim)
        recon = paste_observed(recon, state["observed"], n_azim)

        mse = F.mse_loss(recon, vg).item() * 1000
        state["mse_history"].append(mse)
        state["recon_np"] = recon[0].cpu().numpy()    # (N, C, H, W)

        state["h"], state["c"] = h, c
        state["step"] = t + 1

        if t < T - 1:
            policy = pick_policy(state["policy_name"])
            if isinstance(policy, Actor):
                rel_pos = torch.tensor([[
                    state["rel_elev"] / max(n_elev - 1, 1),
                    state["rel_azim"] / n_azim,
                ]], dtype=torch.float32, device=DEVICE)
                time_frac = torch.tensor([[t / T]], dtype=torch.float32, device=DEVICE)
                abs_elev_norm = torch.tensor([[e / max(n_elev - 1, 1)]], dtype=torch.float32, device=DEVICE)
                logits = policy(a_t, rel_pos, time_frac, abs_elev_norm)
                action, _, _ = policy.get_action(logits, deterministic=True)
            else:
                action = policy.get_action(B, DEVICE)

            act_idx = action[0].item()
            de, da  = cfg.action_deltas[act_idx]
            new_e, new_a = step_position(e, a, de, da, n_elev, n_azim)

            state["d_elev_prev"] = de
            state["d_azim_prev"] = da
            state["rel_elev"]   += de
            state["rel_azim"]   += da
            state["elev_cur"]    = new_e
            state["azim_cur"]    = new_a
        else:
            state["done"] = True

    return state


def render_state(state: dict) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image, str]:
    """Return (gt_img, recon_img, traj_img, mse_img, status_text) from current state."""
    n_elev = cfg.n_elev
    n_azim = cfg.n_azim
    T      = cfg.T
    e, a   = state["elev_cur"], state["azim_cur"]

    # Ground truth with current pos + history
    gt_img = viewgrid_to_pil(
        state["vg_np"], n_elev, n_azim,
        highlight=(e, a),
        visited=state["visited"],
    )

    # Predicted viewgrid (or blank before first step)
    if state["recon_np"] is not None:
        recon_img = viewgrid_to_pil(
            state["recon_np"], n_elev, n_azim,
            highlight=(e, a),
        )
    else:
        recon_img = gt_img  # placeholder

    traj_img = trajectory_to_pil(state["visited"], (e, a), n_elev, n_azim)
    mse_img  = mse_curve_to_pil(state["mse_history"], T)

    step     = state["step"]
    policy   = state["policy_name"]
    done_str = " — DONE" if state["done"] else ""
    mse_str  = f"MSE×1000 = {state['mse_history'][-1]:.2f}" if state["mse_history"] else ""
    status   = f"Step {step}/{T} | policy={policy} | pos=({e},{a}) | {mse_str}{done_str}"

    return gt_img, recon_img, traj_img, mse_img, status


# ── Full eval_transfer run (for the "Run Full Eval" tab) ──────────────────────
_eval_running = threading.Event()
_eval_progress: Dict[str, Any] = {"pct": 0.0, "msg": "idle"}


def run_eval_transfer_bg(data_dir: str):
    """Background eval run — populates _eval_progress, saves results/eval_transfer.png."""
    import h5py
    from torch.utils.data import DataLoader, TensorDataset

    _eval_progress["pct"] = 0.0
    _eval_progress["msg"] = "Loading datasets..."

    try:
        tr_ds = SUN360Dataset(data_dir, split="train", mean_subtract=True)
        va_ds = SUN360Dataset(data_dir, split="val", mean_subtract=True)
        _cfg  = Config()
        _cfg.data_dir = data_dir
        _cfg.n_elev   = tr_ds.n_elev
        _cfg.n_azim   = tr_ds.n_azim
        _cfg.n_views  = tr_ds.n_views

        trn_h5 = os.path.join(data_dir, "pixels_trn_torchfeed.h5")
        val_h5 = os.path.join(data_dir, "pixels_val_torchfeed.h5")
        with h5py.File(trn_h5, "r") as f:
            trn_labels = torch.from_numpy(f["labs"][:].flatten().astype(np.int64))
        with h5py.File(val_h5, "r") as f:
            val_labels = torch.from_numpy(f["labs"][:].flatten().astype(np.int64))

        n_classes = int(max(trn_labels.max(), val_labels.max()) + 1)
        if n_classes < 2:
            _eval_progress["msg"] = "ERROR: dataset has < 2 classes. Run data/patch_labels.py first."
            _eval_progress["pct"] = -1.0
            _eval_running.clear()
            return

        class _LabelDS(torch.utils.data.Dataset):
            def __init__(self, ds, labs):
                self.ds   = ds
                self.labs = labs
            def __len__(self): return len(self.ds)
            def __getitem__(self, i): return self.ds[i], self.labs[i]

        trn_loader = DataLoader(_LabelDS(tr_ds, trn_labels), batch_size=32, shuffle=False)
        val_loader = DataLoader(_LabelDS(va_ds, val_labels), batch_size=32, shuffle=False)

        from eval_transfer import extract_features_per_step, train_classifier, evaluate_classifier

        T       = _cfg.T
        results = {}
        policies_list = [("ours", actor), ("random", rand_policy), ("large-action", large_policy)]
        total_stages  = len(policies_list) + 1   # policies + 1-view

        for stage, (name, policy) in enumerate(policies_list):
            _eval_progress["pct"] = (stage / total_stages) * 0.8
            _eval_progress["msg"] = f"Extracting features: {name} (train)..."
            trn_feats, trn_labs = extract_features_per_step(
                trn_loader, encoder, loc_sensor, combine, memory,
                policy, _cfg, DEVICE, seed=42)
            _eval_progress["msg"] = f"Extracting features: {name} (val)..."
            val_feats, val_labs = extract_features_per_step(
                val_loader, encoder, loc_sensor, combine, memory,
                policy, _cfg, DEVICE, seed=42)

            accs = []
            for t in range(T):
                _eval_progress["msg"] = f"{name}: training classifier t={t+1}/{T}..."
                clf = train_classifier(trn_feats[t], trn_labs, n_classes,
                                       _cfg.d_hidden, DEVICE, epochs=200)
                acc = evaluate_classifier(clf, val_feats[t], val_labs, DEVICE)
                accs.append(acc)
            results[name] = accs

        # 1-view baseline
        _eval_progress["pct"] = 0.85
        _eval_progress["msg"] = "Computing 1-view baseline..."
        trn_feats, trn_labs = extract_features_per_step(
            trn_loader, encoder, loc_sensor, combine, memory,
            rand_policy, _cfg, DEVICE, seed=42)
        val_feats, val_labs = extract_features_per_step(
            val_loader, encoder, loc_sensor, combine, memory,
            rand_policy, _cfg, DEVICE, seed=42)
        clf_1v = train_classifier(trn_feats[0], trn_labs, n_classes,
                                   _cfg.d_hidden, DEVICE, epochs=200)
        acc_1v = evaluate_classifier(clf_1v, val_feats[0], val_labs, DEVICE)
        results["1-view"] = [acc_1v] * T

        # Save plot
        _eval_progress["pct"] = 0.95
        _eval_progress["msg"] = "Saving plot..."
        os.makedirs("results", exist_ok=True)
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
        ax.set_title(f"Active categorization (n_classes={n_classes})")
        ax.set_xticks(timesteps)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plot_path = "results/eval_transfer.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

        # Save JSON
        with open("results/transfer_metrics.json", "w") as fh:
            json.dump({"n_classes": n_classes, "T": T, "results": results}, fh, indent=2)

        _eval_progress["pct"] = 1.0
        _eval_progress["msg"] = f"Done! Saved {plot_path}"
        _eval_progress["plot_path"] = plot_path
        _eval_progress["results"] = results
        _eval_progress["n_classes"] = n_classes

    except Exception as exc:
        import traceback
        _eval_progress["msg"] = f"ERROR: {exc}"
        _eval_progress["pct"] = -1.0
        traceback.print_exc()
    finally:
        _eval_running.clear()


# ── Offline tab helpers ────────────────────────────────────────────────────────
def load_offline_mse():
    path = "results/eval_metrics.json"
    if not os.path.exists(path):
        return None, "No results/eval_metrics.json found. Run eval.py first."
    with open(path) as f:
        data = json.load(f)

    res = data.get("results", {})
    T = max(len(v["curve"]) for v in res.values()) if res else 6
    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = {"ours": "tab:blue", "random": "tab:orange", "large-action": "tab:green"}
    for name, info in res.items():
        curve = info["curve"]
        ax.plot(range(1, len(curve) + 1), curve, marker="o",
                color=colors.get(name, "tab:gray"), label=name, linewidth=2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("MSE × 1000")
    ax.set_title("Reconstruction MSE vs. Timestep (val set)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130)
    plt.close()
    buf.seek(0)
    img = Image.open(buf).copy()

    rows = {name: info["curve"] for name, info in res.items()}
    table = results_table_html({"results": rows}, "results")
    return img, table


def load_offline_transfer():
    path = "results/transfer_metrics.json"
    if not os.path.exists(path):
        return None, "No results/transfer_metrics.json found."
    with open(path) as f:
        data = json.load(f)
    rows = data.get("results", {})
    table = results_table_html({"results": rows}, "results")

    fig5 = "results/fig5_policy_transfer.png"
    transfer_png = "results/eval_transfer.png"
    img_path = fig5 if os.path.exists(fig5) else (transfer_png if os.path.exists(transfer_png) else None)
    img = Image.open(img_path) if img_path else None
    return img, table


def browse_val_panorama(idx: int):
    if val_dataset is None:
        return None, "Model not loaded yet."
    idx = int(idx) % len(val_dataset)
    vg  = val_dataset[idx].numpy()   # (N, C, H, W)
    img = viewgrid_to_pil(vg, cfg.n_elev, cfg.n_azim, scale=4)
    return img, f"Sample {idx} — all {cfg.n_views} views ({cfg.n_elev}×{cfg.n_azim} grid, 32×32 px)"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="LookAround 2026 — Viewer") as demo:
        gr.Markdown("# LookAround 2026 — Interactive Viewer")
        gr.Markdown(
            "PyTorch reimplementation of *Learning to Look Around* (Jayaraman & Grauman, CVPR 2018). "
            f"**Checkpoint:** `{CHECKPOINT}` | **Device:** `{DEVICE}` | **Data:** `{DATA_DIR}`"
        )

        with gr.Tabs():

            # ── Tab 1: Offline results ────────────────────────────────────────
            with gr.Tab("Offline Results"):
                gr.Markdown("### Pre-computed results from `results/`")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Reconstruction MSE (Fig 4)")
                        mse_img   = gr.Image(label="MSE vs timestep", type="pil")
                        mse_table = gr.HTML()
                        btn_mse   = gr.Button("Load MSE results")
                        btn_mse.click(fn=load_offline_mse, outputs=[mse_img, mse_table])

                    with gr.Column():
                        gr.Markdown("#### Policy Transfer Accuracy (Fig 5)")
                        tr_img    = gr.Image(label="Transfer accuracy vs timestep", type="pil")
                        tr_table  = gr.HTML()
                        btn_tr    = gr.Button("Load Transfer results")
                        btn_tr.click(fn=load_offline_transfer, outputs=[tr_img, tr_table])

                gr.Markdown("---")
                gr.Markdown("#### Browse val panoramas")
                with gr.Row():
                    pano_slider = gr.Slider(0, max(len(val_dataset) - 1, 1) if val_dataset else 460,
                                            step=1, value=0, label="Sample index")
                    pano_btn    = gr.Button("Show panorama")
                pano_img    = gr.Image(label="Viewgrid (4×8 = 32 views)", type="pil")
                pano_status = gr.Textbox(label="Info", interactive=False)
                pano_btn.click(fn=browse_val_panorama, inputs=[pano_slider],
                               outputs=[pano_img, pano_status])
                pano_slider.change(fn=browse_val_panorama, inputs=[pano_slider],
                                   outputs=[pano_img, pano_status])

            # ── Tab 2: Live episode ───────────────────────────────────────────
            with gr.Tab("Live Episode"):
                gr.Markdown(
                    "### Step through the model episode interactively\n"
                    "Configure a sample + policy, initialize, then step through T=6 timesteps."
                )

                episode_state = gr.State(value=None)

                with gr.Row():
                    with gr.Column(scale=1):
                        sample_slider = gr.Slider(
                            0, max(len(val_dataset) - 1, 1) if val_dataset else 460,
                            step=1, value=0, label="Val sample index")
                        policy_dd = gr.Dropdown(
                            choices=["ours", "random", "large-action"],
                            value="ours", label="Policy")
                        with gr.Row():
                            elev_dd = gr.Dropdown(
                                choices=list(range(cfg.n_elev if cfg else 4)),
                                value=0, label="Start elev")
                            azim_dd = gr.Dropdown(
                                choices=list(range(cfg.n_azim if cfg else 8)),
                                value=0, label="Start azim")
                        with gr.Row():
                            btn_rand_pos = gr.Button("Random start pos")
                            btn_init     = gr.Button("Initialize", variant="primary")
                        with gr.Row():
                            btn_step     = gr.Button("Next Step ▶")
                            btn_run_all  = gr.Button("Run All 6 Steps ⏩")

                        live_status = gr.Textbox(label="Status", interactive=False,
                                                 value="Press Initialize to start.")

                    with gr.Column(scale=3):
                        with gr.Row():
                            gt_img    = gr.Image(label="Ground Truth Viewgrid", type="pil")
                            recon_img = gr.Image(label="Predicted Viewgrid",    type="pil")
                        with gr.Row():
                            traj_img  = gr.Image(label="Trajectory Map", type="pil")
                            mse_live  = gr.Image(label="MSE vs. time",   type="pil")

                def randomize_pos():
                    import random
                    n_e = cfg.n_elev if cfg else 4
                    n_a = cfg.n_azim if cfg else 8
                    return random.randint(0, n_e - 1), random.randint(0, n_a - 1)

                btn_rand_pos.click(fn=randomize_pos, outputs=[elev_dd, azim_dd])

                def do_init(sample_idx, policy_name, elev_s, azim_s):
                    if val_dataset is None:
                        return (None, None, None, None,
                                "ERROR: Model not loaded.", None)
                    state = new_episode_state(int(sample_idx), policy_name,
                                              int(elev_s), int(azim_s))
                    gt, recon, traj, mse_c, status = render_state(state)
                    return gt, recon, traj, mse_c, status, state

                btn_init.click(
                    fn=do_init,
                    inputs=[sample_slider, policy_dd, elev_dd, azim_dd],
                    outputs=[gt_img, recon_img, traj_img, mse_live, live_status, episode_state],
                )

                def do_step(state):
                    if state is None:
                        return None, None, None, None, "Press Initialize first.", state
                    if state["done"]:
                        gt, recon, traj, mse_c, status = render_state(state)
                        return gt, recon, traj, mse_c, "Episode complete. Press Initialize to restart.", state
                    state = run_one_step(state)
                    gt, recon, traj, mse_c, status = render_state(state)
                    return gt, recon, traj, mse_c, status, state

                btn_step.click(
                    fn=do_step,
                    inputs=[episode_state],
                    outputs=[gt_img, recon_img, traj_img, mse_live, live_status, episode_state],
                )

                def do_run_all(state):
                    if state is None:
                        return None, None, None, None, "Press Initialize first.", state
                    T = cfg.T if cfg else 6
                    while not state["done"] and state["step"] < T:
                        state = run_one_step(state)
                    gt, recon, traj, mse_c, status = render_state(state)
                    return gt, recon, traj, mse_c, status, state

                btn_run_all.click(
                    fn=do_run_all,
                    inputs=[episode_state],
                    outputs=[gt_img, recon_img, traj_img, mse_live, live_status, episode_state],
                )

            # ── Tab 3: Run full eval_transfer ─────────────────────────────────
            with gr.Tab("Run Full Eval"):
                gr.Markdown(
                    "### Generate a new `results/eval_transfer.png`\n"
                    "Runs policy transfer evaluation (extract features → train linear classifier per "
                    "timestep → accuracy vs. time). Takes a few minutes."
                )
                with gr.Row():
                    eval_data_dd = gr.Dropdown(
                        choices=[DATA_DIR,
                                 "data/modelnet10_torchfeed",
                                 "data/panocontext_torchfeed"],
                        value=DATA_DIR, label="Dataset directory",
                        allow_custom_value=True,
                    )
                    btn_run_eval = gr.Button("Run eval_transfer ▶", variant="primary")

                eval_progress = gr.Slider(0, 1, value=0, label="Progress",
                                          interactive=False, step=0.01)
                eval_status   = gr.Textbox(label="Status", value="idle", interactive=False)
                eval_result   = gr.Image(label="New eval_transfer.png", type="pil")
                eval_table    = gr.HTML()

                def run_eval_streaming(data_dir):
                    """Generator: yields progress updates, then final result."""
                    if _eval_running.is_set():
                        yield 0.0, "Already running — please wait.", None, ""
                        return

                    _eval_running.set()
                    _eval_progress.clear()
                    _eval_progress.update({"pct": 0.0, "msg": "Starting..."})

                    thread = threading.Thread(
                        target=run_eval_transfer_bg,
                        args=(data_dir,),
                        daemon=True,
                    )
                    thread.start()

                    while _eval_running.is_set():
                        time.sleep(1.0)
                        pct = _eval_progress.get("pct", 0.0)
                        msg = _eval_progress.get("msg", "...")
                        yield pct, msg, None, ""

                    # Final result
                    if _eval_progress.get("pct", -1) < 0:
                        yield 0.0, _eval_progress.get("msg", "Error"), None, ""
                        return

                    plot_path = _eval_progress.get("plot_path", "results/eval_transfer.png")
                    img = Image.open(plot_path) if os.path.exists(plot_path) else None
                    rows = _eval_progress.get("results", {})
                    table = results_table_html({"results": rows}, "results") if rows else ""
                    yield 1.0, _eval_progress.get("msg", "Done!"), img, table

                btn_run_eval.click(
                    fn=run_eval_streaming,
                    inputs=[eval_data_dd],
                    outputs=[eval_progress, eval_status, eval_result, eval_table],
                )

        gr.Markdown("---")
        gr.Markdown(
            "_LookAround 2026 — reimplementation of Jayaraman & Grauman, CVPR 2018 "
            "([arXiv:1709.00507](https://arxiv.org/abs/1709.00507))_"
        )

    return demo


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading model and data...")
    status = load_model_and_data()
    print(status)

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=SHARE,
        show_error=True,
        theme=gr.themes.Soft(),
    )
