"""Samples/s the DataLoader can produce per num_workers"""
import time
from dataclasses import replace

import torch

from lerobot.datasets import LeRobotDatasetMetadata

from robopol.config import TrainConfig
from robopol.data import make_dataloader, make_dataset
from robopol.model import make_policy_config

WARMUP, MEASURE = 5, 50

base = TrainConfig()
meta = LeRobotDatasetMetadata(base.repo_id)
policy_cfg = make_policy_config(base, meta, torch.device("cuda"))
dataset = make_dataset(base, policy_cfg, meta.fps)

for workers in (2, 4, 6, 8, 0):
    cfg = replace(base, num_workers=workers)
    loader, _ = make_dataloader(cfg, policy_cfg, dataset)
    it = iter(loader)

    for _ in range(WARMUP):
        next(it)
    t0 = time.perf_counter()

    for _ in range(MEASURE):
        batch = next(it)

    dt = time.perf_counter() - t0

    print(f"num_workers={workers}: {MEASURE * cfg.batch_size / dt:7.0f} samples/s")
    
    del it, loader  