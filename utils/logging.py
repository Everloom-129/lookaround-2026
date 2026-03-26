"""
Logging utilities: optional wandb integration + console logging.
"""
import os
from typing import Any, Dict, Optional


def init_logging(config, run_name: Optional[str] = None) -> Any:
    """
    Initialize wandb run if config.use_wandb is True.

    Returns:
        wandb run object, or None if not using wandb.
    """
    if not config.use_wandb:
        return None
    try:
        import wandb
        run = wandb.init(
            project="lookaround-2026",
            name=run_name,
            config=vars(config),
        )
        return run
    except ImportError:
        print("[logging] wandb not installed, skipping.")
        return None


def log_metrics(metrics: Dict[str, float], step: int,
                run: Any = None) -> None:
    """
    Log metrics to wandb (if available) and print to stdout.
    """
    msg = f"[step {step:6d}] " + "  ".join(
        f"{k}={v:.4f}" for k, v in metrics.items()
    )
    print(msg)
    if run is not None:
        run.log(metrics, step=step)
