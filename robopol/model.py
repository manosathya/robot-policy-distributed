"""Build LeRobot's Diffusion Policy and its normalisation processors from a TrainConfig."""

import torch
from torch import nn

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features

from robopol.config import TrainConfig


def make_policy_config(cfg: TrainConfig, meta: LeRobotDatasetMetadata, device: torch.device) -> DiffusionConfig:
    features = dataset_to_policy_features(meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}
    return DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        device=device.type,
        n_obs_steps=cfg.n_obs_steps,
        horizon=cfg.horizon,
        n_action_steps=cfg.n_action_steps,
        # Never sample the last frames as t: their targets would be mostly padding.
        drop_n_last_frames=cfg.horizon - cfg.n_action_steps - cfg.n_obs_steps + 1,
        vision_backbone=cfg.vision_backbone,
        crop_shape=cfg.crop_shape,
        pretrained_backbone_weights=None,  # GroupNorm can't reuse BatchNorm-trained weights
        use_group_norm=cfg.use_group_norm,
    )


def build_policy(cfg: TrainConfig, meta: LeRobotDatasetMetadata, device: torch.device):
    policy_cfg = make_policy_config(cfg, meta, device)
    policy = DiffusionPolicy(policy_cfg).to(device)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=meta.stats)
    return policy, preprocessor, postprocessor


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def parameter_report(policy: DiffusionPolicy) -> dict[str, int]:
    return {
        "rgb_encoder": count_parameters(policy.diffusion.rgb_encoder),
        "unet": count_parameters(policy.diffusion.unet),
        "total": count_parameters(policy),
    }


if __name__ == "__main__":
    cfg = TrainConfig()
    meta = LeRobotDatasetMetadata(cfg.repo_id)
    policy, _, _ = build_policy(cfg, meta, torch.device("cuda"))
    for name, n in parameter_report(policy).items():
        print(f"{name:12s} {n / 1e6:8.2f} M")