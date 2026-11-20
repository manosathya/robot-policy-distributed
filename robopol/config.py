"""Stage 1 training run settings + Eval Configs. Defaults reproduce lerobot/diffusion_pusht."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    # --- data ---
    repo_id: str = "lerobot/pusht"
    episodes: list[int] | None = None  # None = all 206
    num_workers: int = 4

    # --- Diffusion Policy ---
    n_obs_steps: int = 2     #Observation history: t-1,t
    horizon: int = 16
    n_action_steps: int = 8
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] = (84, 84)
    use_group_norm: bool = True

    # --- optimisation ---
    steps: int = 20_000
    batch_size: int = 64
    lr: float = 1e-4
    betas: tuple[float, float] = (0.95, 0.999)
    weight_decay: float = 1e-6
    warmup_steps: int = 500
    grad_clip_norm: float = 10.0
    tf32: bool = True

    # --- logging ---
    seed: int = 1000
    run_name: str = "stage1"
    runs_dir: Path = Path.home() / "runs"
    log_every: int = 100
    save_every: int = 5_000
    overfit_one_batch: bool = False

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_name



@dataclass
class EvalConfig:
    policy_path: str = "lerobot/diffusion_pusht"  # local pretrained_model folder or Hub id
    n_episodes: int = 50
    start_seed: int = 100_000  # env seeds start_seed .. start_seed + n_episodes - 1
    max_steps: int = 300
    n_gifs: int = 5
    torch_seed: int = 0        # fixes the DDPM sampling noise
    out_dir: Path | None = None
    device: str = "cuda"