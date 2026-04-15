"""
Preprocess ModelNet-10 (or ModelNet-40) OFF files → HDF5 viewgrid torchfeed.

Renders each 3D CAD model from N_ELEV × N_AZIM viewpoints using pyrender + EGL
(headless, no display needed), producing 32×32 RGB images stored in the same
HDF5 schema as prepare_sun360.py.

Viewgrid layout matches our 4-elev × 8-azim = 32-view config:
  elevations:  -45°, -15°, +15°, +45°   (4 rings)
  azimuths:    0°, 45°, 90°, ..., 315°  (8 positions)

Output HDF5 schema (same as prepare_sun360.py):
  target_viewgrid    uint8  (N, 3, PANO_W, PANO_H)  — Torch7 layout (C, W, H)
  labs               float64 (N, 1)                  — category index
  average_target_viewgrid  uint8  (3, PANO_W, PANO_H)
  shuffle_ord        float64 (N,)
  gridshape          float64 [N_ELEV, N_AZIM]
  view_snapshape     float64 [VIEW_H, VIEW_W]
  pano_dims          float64 [PANO_H, PANO_W]

Usage:
  # Extract ModelNet10.zip first:
  unzip /tmp/ModelNet10.zip -d data/modelnet_raw/

  uv run python data/prepare_modelnet.py \\
    --model-dir data/modelnet_raw/ModelNet10 \\
    --out-dir   data/modelnet10_torchfeed

  # For ModelNet40 (all 40 classes, for training 30 disjoint from ModelNet10):
  uv run python data/prepare_modelnet.py \\
    --model-dir data/modelnet_raw/ModelNet40 \\
    --out-dir   data/modelnet40_torchfeed
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

os.environ["PYOPENGL_PLATFORM"] = "egl"  # headless EGL — must be set before pyrender import
import pyrender
import trimesh

# ── Grid config (matches our 4×8=32 view model) ──────────────────────────────
N_ELEV   = 4
N_AZIM   = 8
VIEW_H   = 32
VIEW_W   = 32
PANO_H   = N_ELEV * VIEW_H   # 128
PANO_W   = N_AZIM  * VIEW_W  # 256
CAM_DIST = 2.5               # camera distance from origin (after mesh normalisation)

# Elevation angles in degrees — 4 evenly spaced rings
ELEV_DEGS = [-45.0, -15.0, 15.0, 45.0]
# Azimuth angles in degrees — 8 evenly spaced (0° = +X axis)
AZIM_DEGS = [i * 360.0 / N_AZIM for i in range(N_AZIM)]

SPLIT_FRAC = {"train": 0.73, "val": 0.13}  # test is remainder
SEED       = 62346

_SPLIT_FILENAMES = {
    "train": "pixels_trn_torchfeed.h5",
    "val":   "pixels_val_torchfeed.h5",
    "test":  "pixels_tst_torchfeed.h5",
}


# ── Rendering helpers ─────────────────────────────────────────────────────────

def camera_pose(elev_deg: float, azim_deg: float, dist: float) -> np.ndarray:
    """
    4×4 camera-to-world matrix placing the camera at (elev, azim, dist),
    looking toward the origin with Z-up convention.
    """
    el = np.radians(elev_deg)
    az = np.radians(azim_deg)
    # Camera position in world
    cx = dist * np.cos(el) * np.cos(az)
    cy = dist * np.cos(el) * np.sin(az)
    cz = dist * np.sin(el)
    pos = np.array([cx, cy, cz])

    # Forward: from camera toward origin
    fwd = -pos / np.linalg.norm(pos)
    # Up: world Z, but if camera is near pole use Y
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(fwd, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    # Build 4×4: columns are right, up, -fwd, pos (OpenGL convention)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -fwd   # pyrender camera looks along -Z
    pose[:3, 3] = pos
    return pose


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center and scale mesh to fit in unit sphere."""
    mesh = mesh.copy()
    mesh.vertices -= mesh.bounding_box.centroid
    scale = mesh.bounding_sphere.primitive.radius
    if scale > 0:
        mesh.vertices /= scale
    return mesh


def render_viewgrid(mesh_path: str, renderer: pyrender.OffscreenRenderer,
                    cam: pyrender.PerspectiveCamera,
                    light_poses: List[np.ndarray]) -> np.ndarray:
    """
    Render N_ELEV × N_AZIM views of a mesh, return (3, PANO_W, PANO_H) uint8
    in Torch7 layout (channels-first, W before H).
    """
    # Load & normalize mesh
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
        if not geoms:
            return None
        mesh_tm = trimesh.util.concatenate(geoms)
    else:
        mesh_tm = loaded
    mesh_tm = normalize_mesh(mesh_tm)

    # Build pyrender scene (mesh + lights)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    pm = pyrender.Mesh.from_trimesh(mesh_tm, smooth=True)
    scene.add(pm)
    cam_node = scene.add(cam)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    light_nodes = [scene.add(light, pose=lp) for lp in light_poses]

    # Render each view
    patches = []   # list of (3, VIEW_H, VIEW_W) uint8
    for el in ELEV_DEGS:
        for az in AZIM_DEGS:
            pose = camera_pose(el, az, CAM_DIST)
            scene.set_pose(cam_node, pose)
            for ln, lp in zip(light_nodes, light_poses):
                # Key light follows camera; fill light is fixed
                scene.set_pose(ln, pose)
                break
            color, _ = renderer.render(scene)   # (VIEW_H, VIEW_W, 3) uint8
            patches.append(color)

    # Assemble into (PANO_H, PANO_W, 3) → (3, PANO_W, PANO_H) Torch7
    grid_hw3 = np.zeros((PANO_H, PANO_W, 3), dtype=np.uint8)
    idx = 0
    for ei in range(N_ELEV):
        for ai in range(N_AZIM):
            r0, r1 = ei * VIEW_H, (ei + 1) * VIEW_H
            c0, c1 = ai * VIEW_W, (ai + 1) * VIEW_W
            grid_hw3[r0:r1, c0:c1] = patches[idx]
            idx += 1

    # (H, W, C) → (C, H, W) → (C, W, H) Torch7
    chw   = grid_hw3.transpose(2, 0, 1)
    torch7 = chw.transpose(0, 2, 1)   # (3, PANO_W, PANO_H)
    return torch7


# ── Data scanning ─────────────────────────────────────────────────────────────

def find_models(model_dir: str) -> List[Tuple[str, int]]:
    """
    Scan ModelNet directory structure:
      model_dir/<category>/{train,test}/*.off

    Returns list of (off_path, label_int), sorted for reproducibility.
    Categories are sorted alphabetically → int label.
    """
    model_dir = Path(model_dir)
    ext = {".off"}

    categories = sorted(d.name for d in model_dir.iterdir() if d.is_dir())
    cat2int    = {c: i for i, c in enumerate(categories)}
    print(f"Found {len(categories)} categories: {categories}")

    pairs = []
    for cat in categories:
        for split in ("train", "test"):
            split_dir = model_dir / cat / split
            if not split_dir.exists():
                continue
            for p in sorted(split_dir.rglob("*")):
                if p.suffix.lower() in ext:
                    pairs.append((str(p), cat2int[cat]))

    print(f"Total models: {len(pairs)}")
    return pairs, categories


# ── HDF5 writing ──────────────────────────────────────────────────────────────

def write_hdf5(out_path: str, items: List[Tuple[str, int]],
               renderer: pyrender.OffscreenRenderer,
               cam: pyrender.PerspectiveCamera,
               light_poses: List[np.ndarray],
               all_indices: List[int]) -> None:
    N = len(items)
    viewgrids  = np.zeros((N, 3, PANO_W, PANO_H), dtype=np.uint8)
    labs       = np.zeros((N, 1), dtype=np.float64)
    valid_mask = np.ones(N, dtype=bool)

    print(f"  Rendering {N} models → {out_path}")
    for i, (path, label) in enumerate(items):
        if i % 200 == 0:
            print(f"    [{i}/{N}]")
        vg = render_viewgrid(path, renderer, cam, light_poses)
        if vg is None:
            valid_mask[i] = False
            continue
        viewgrids[i] = vg
        labs[i, 0]   = float(label)

    viewgrids = viewgrids[valid_mask]
    labs      = labs[valid_mask]
    avg_vg    = viewgrids.mean(axis=0).astype(np.uint8)
    shuffle_ord = np.array([all_indices[i] for i in range(N) if valid_mask[i]],
                           dtype=np.float64)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("target_viewgrid",         data=viewgrids)
        f.create_dataset("average_target_viewgrid", data=avg_vg)
        f.create_dataset("labs",                    data=labs)
        f.create_dataset("shuffle_ord",             data=shuffle_ord)
        f.create_dataset("gridshape",   data=np.array([N_ELEV, N_AZIM], dtype=np.float64))
        f.create_dataset("view_snapshape", data=np.array([VIEW_H, VIEW_W], dtype=np.float64))
        f.create_dataset("pano_dims",   data=np.array([PANO_H, PANO_W], dtype=np.float64))

    print(f"    Saved {viewgrids.shape[0]} models to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True,
                        help="ModelNet10 or ModelNet40 root directory")
    parser.add_argument("--out-dir",   default="data/modelnet10_torchfeed")
    parser.add_argument("--seed",      type=int, default=SEED)
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    pairs, categories = find_models(args.model_dir)

    if args.dry_run:
        print("Dry run — no files written.")
        return

    # Split (same logic as prepare_sun360.py)
    rng      = random.Random(args.seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    N        = len(shuffled)
    n_train  = int(N * SPLIT_FRAC["train"])
    n_val    = int(N * SPLIT_FRAC["val"])
    splits   = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }
    for sn, items in splits.items():
        print(f"  {sn}: {len(items)} models")

    # Global indices for shuffle_ord
    rng2      = random.Random(args.seed)
    all_idx   = list(range(N))
    rng2.shuffle(all_idx)
    split_indices = {
        "train": all_idx[:n_train],
        "val":   all_idx[n_train:n_train + n_val],
        "test":  all_idx[n_train + n_val:],
    }

    os.makedirs(args.out_dir, exist_ok=True)

    # Shared renderer + camera
    renderer    = pyrender.OffscreenRenderer(VIEW_W, VIEW_H)
    cam         = pyrender.PerspectiveCamera(yfov=np.radians(60.0), aspectRatio=1.0)
    # Two fixed fill lights + one key light (updated per view in render_viewgrid)
    light_poses = [
        camera_pose(30,  45, CAM_DIST),   # key (overwritten per view)
        camera_pose(20, 225, CAM_DIST),   # fill
    ]

    # Save category list
    cat_path = os.path.join(args.out_dir, "categories.txt")
    with open(cat_path, "w") as f:
        f.write("\n".join(f"{i}\t{c}" for i, c in enumerate(categories)))
    print(f"Saved {cat_path}")

    print("\nWriting HDF5 files...")
    for split_name in ("train", "val", "test"):
        out_path = os.path.join(args.out_dir, _SPLIT_FILENAMES[split_name])
        write_hdf5(out_path, splits[split_name], renderer, cam,
                   light_poses, split_indices[split_name])

    renderer.delete()
    print("\nDone.")


if __name__ == "__main__":
    main()
