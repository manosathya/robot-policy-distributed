"""Windowed PushT dataset and an episode-aware DataLoader."""

from collections.abc import Iterator

from torch.utils.data import DataLoader

from lerobot.datasets import EpisodeAwareSampler, LeRobotDataset
from lerobot.policies.diffusion import DiffusionConfig

from robopol.config import TrainConfig



def make_delta_timestamps(policy_cfg: DiffusionConfig, fps: float) -> dict[str, list[float]]:
    obs = [i / fps for i in policy_cfg.observation_delta_indices]  # [-1, 0] -> [-0.1, 0.0]
    act = [i / fps for i in policy_cfg.action_delta_indices]       # [-1 .. 14] -> 16 offsets
    return {"observation.image": obs, "observation.state": obs, "action": act}


def make_dataset(cfg: TrainConfig, policy_cfg: DiffusionConfig, fps: float) -> LeRobotDataset:
    return LeRobotDataset(
        cfg.repo_id,
        episodes=cfg.episodes,
        delta_timestamps=make_delta_timestamps(policy_cfg, fps),
    )


def make_dataloader(cfg: TrainConfig, policy_cfg: DiffusionConfig, dataset: LeRobotDataset):
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=policy_cfg.drop_n_last_frames,
        shuffle=True,
        seed=cfg.seed,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    return loader, sampler


def cycle(loader: DataLoader) -> Iterator[dict]:
    """Endless stream of batches. Each pass over the loader is a new reshuffled epoch."""
    while True:
        yield from loader