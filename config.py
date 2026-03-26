from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # Data
    data_dir: str = "code-2017/SUN360/data/minitorchfeed"  # dir with split .h5 files
    n_elev: int = 4
    n_azim: int = 8
    n_views: int = 32           # n_elev * n_azim  (actual SUN360 mini: 4x8=32)
    view_height: int = 32
    view_width: int = 32

    # Model
    d_enc: int = 256            # ViewEncoder output dim
    d_loc_in: int = 3           # LocationSensor input: [abs_elev, d_elev, d_azim]
    d_loc: int = 16             # LocationSensor output dim
    d_hidden: int = 256         # LSTM hidden dim
    n_actions: int = 14         # 3-elev × 5-azim neighborhood minus center
    dropout: float = 0.3

    # 14 2D action offsets (Δelev, Δazim) — 3×5 grid minus center (0,0)
    action_deltas: List[Tuple[int, int]] = field(default_factory=lambda: [
        (-1, -2), (-1, -1), (-1,  0), (-1,  1), (-1,  2),
        ( 0, -2), ( 0, -1),           ( 0,  1), ( 0,  2),
        ( 1, -2), ( 1, -1), ( 1,  0), ( 1,  1), ( 1,  2),
    ])

    # Episode
    T: int = 6

    # Training
    batch_size: int = 32
    lr: float = 1e-3
    lambda_policy: float = 1.0
    baseline_decay: float = 0.9     # EMA alpha for running baseline
    max_grad_norm: float = 5.0
    n_epochs: int = 2000
    pretrain_epochs: int = 50

    # Checkpointing / logging
    log_every: int = 100
    save_every: int = 5
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = False
