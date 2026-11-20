"""Save evaluation ready policies and resumable training state."""
import os
import random
from pathlib import Path

import numpy as np
import torch


def get_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(run_dir: Path, step: int, policy, preprocessor, postprocessor,
                    optimizer, lr_scheduler) -> Path:
    ckpt_dir = run_dir / "checkpoints" / f"{step:07d}"
    model_dir = ckpt_dir / "pretrained_model"
    policy.save_pretrained(model_dir)        # config.json + model.safetensors
    preprocessor.save_pretrained(model_dir)  # normalisation steps + dataset stats
    postprocessor.save_pretrained(model_dir)

    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "rng": get_rng_state(),
    }
    tmp = ckpt_dir / "training_state.pt.tmp"
    torch.save(state, tmp)
    os.replace(tmp, ckpt_dir / "training_state.pt")  # atomic: never a half-written file
    (run_dir / "checkpoints" / "last.txt").write_text(ckpt_dir.name)
    return ckpt_dir


def load_training_state(ckpt_dir: Path, optimizer, lr_scheduler) -> int:
    # weights_only=False: the file holds Python/NumPy RNG objects, and we wrote it ourselves.
    state = torch.load(ckpt_dir / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    set_rng_state(state["rng"])
    return state["step"]