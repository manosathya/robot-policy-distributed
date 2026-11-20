import json
import logging
import random
import time
from dataclasses import asdict

import draccus
import numpy as np
import torch
from diffusers.optimization import get_scheduler
from torch.utils.tensorboard import SummaryWriter

from lerobot.datasets import LeRobotDatasetMetadata

from robopol.checkpoint import save_checkpoint
from robopol.config import TrainConfig
from robopol.data import cycle, make_dataloader, make_dataset
from robopol.model import build_policy, parameter_report

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    cfg = draccus.parse(config_class=TrainConfig)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", force=True)
    set_seeds(cfg.seed)

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32
    torch.backends.cudnn.benchmark = True

    cfg.run_dir.mkdir(parents=True, exist_ok=False)  # refuse to overwrite an existing run
    (cfg.run_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    meta = LeRobotDatasetMetadata(cfg.repo_id)
    policy, preprocessor, postprocessor = build_policy(cfg, meta, device)
    logging.info("parameters: %s", {k: f"{v / 1e6:.1f}M" for k, v in parameter_report(policy).items()})

    dataset = make_dataset(cfg, policy.config, meta.fps)
    loader, _ = make_dataloader(cfg, policy.config, dataset)
    logging.info("dataset: %d frames, %d episodes", dataset.num_frames, dataset.num_episodes)

    optimizer = torch.optim.Adam(
        policy.get_optim_params(),
        lr=cfg.lr,
        betas=cfg.betas,
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.steps,
    )
    writer = SummaryWriter(log_dir=str(cfg.run_dir / "tb"))

    batches = cycle(loader)
    fixed_batch = preprocessor(next(batches)) if cfg.overfit_one_batch else None

    policy.train()
    window_start = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        batch = fixed_batch if fixed_batch is not None else preprocessor(next(batches))

        loss, _ = policy.forward(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        lr_scheduler.step()

        if step % cfg.log_every == 0:
            loss_value = loss.item()  
            elapsed = time.perf_counter() - window_start
            samples_per_s = cfg.log_every * cfg.batch_size / elapsed
            peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
            lr = lr_scheduler.get_last_lr()[0]

            writer.add_scalar("train/loss", loss_value, step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("perf/samples_per_s", samples_per_s, step)
            writer.add_scalar("perf/peak_mem_gb", peak_mem_gb, step)
            logging.info(
                "step %6d | loss %.4f | grad %.2f | lr %.2e | %5.0f samples/s | peak %.2f GB",
                step, loss_value, grad_norm.item(), lr, samples_per_s, peak_mem_gb,
            )
            window_start = time.perf_counter()

        if step % cfg.save_every == 0 or step == cfg.steps:
            path = save_checkpoint(cfg.run_dir, step, policy, preprocessor, postprocessor,
                                   optimizer, lr_scheduler)
            logging.info("saved %s", path)

    writer.close()


if __name__ == "__main__":
    main()