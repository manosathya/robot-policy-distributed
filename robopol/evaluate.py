"""Success rate of a Diffusion Policy checkpoint in the PushT simulator.
Run: python -m robopol.evaluate --policy_path=runs/stage1/checkpoints/0020000/pretrained_model
"""
import json
import copy
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from functools import partial


import draccus
import gym_pusht  # noqa: F401  (import registers "gym_pusht/PushT-v0")
import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy

from robopol.config import EvalConfig, RUNS_DIR

def load_processors(policy, policy_path: str, meta):
    if (Path(policy_path) / "policy_preprocessor.json").exists():
        # Our own checkpoints: use the exact processors saved with the model.
        return make_pre_post_processors(policy.config, pretrained_path=policy_path)
    # Legacy Hub model: its normalisation stats live inside model.safetensors.
    weights = load_file(hf_hub_download(policy_path, "model.safetensors"))
    stats = copy.deepcopy(meta.stats)
    for s in ("mean", "std"):
        stats["observation.image"][s] = weights[f"normalize_inputs.buffer_observation_image.{s}"].numpy()
    return make_pre_post_processors(policy.config, dataset_stats=stats)

def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def run_episode(env, policy, preprocessor, postprocessor, seed: int, max_steps: int, record: bool):
    policy.reset()
    obs, info = env.reset(seed=seed)
    frames = [env.render()] if record else []
    max_coverage, success, steps = 0.0, False, 0
    for steps in range(1, max_steps + 1):
        batch = preprocessor(preprocess_observation(obs))
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = postprocessor(action)  # back to pixel coordinates, on CPU
        obs, reward, terminated, truncated, info = env.step(action.squeeze(0).numpy())
        max_coverage = max(max_coverage, float(info["coverage"]))
        success = success or bool(info["is_success"])
        if record:
            frames.append(env.render())
        if terminated or truncated:
            break
    return {"seed": seed, "success": success, "max_coverage": max_coverage, "steps": steps}, frames


def save_gif(frames: list[np.ndarray], path: Path, fps: int = 10) -> None:
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000 // fps, loop=0)

def make_env(max_steps: int):
    return gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        visualization_width=384,
        visualization_height=384,
        max_episode_steps=max_steps,
    )

from functools import partial


def make_env(max_steps: int):
    return gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        visualization_width=384,
        visualization_height=384,
        max_episode_steps=max_steps,
    )


def run_vectorised(policy, preprocessor, postprocessor, seeds: list[int], max_steps: int, n_gifs: int):
    n = len(seeds)
    env = gym.vector.SyncVectorEnv([partial(make_env, max_steps) for _ in range(n)])
    policy.reset()
    obs, info = env.reset(seed=seeds)

    done = np.zeros(n, dtype=bool)
    success = np.zeros(n, dtype=bool)
    max_coverage = np.zeros(n)
    steps = np.zeros(n, dtype=int)
    frames = [[env.envs[i].render()] for i in range(min(n_gifs, n))]

    for t in range(1, max_steps + 1):
        batch = preprocessor(preprocess_observation(obs))
        with torch.inference_mode():
            action = policy.select_action(batch)         # (n, 2), normalised
        action = postprocessor(action).numpy()           # (n, 2), pixel targets
        obs, reward, terminated, truncated, info = env.step(action)

        active = ~done                                   # envs still in their first episode
        coverage = np.asarray(info["coverage"], dtype=float)
        max_coverage[active] = np.maximum(max_coverage[active], coverage[active])
        success[active] |= np.asarray(info["is_success"], dtype=bool)[active]
        steps[active] = t
        for i, f in enumerate(frames):
            if active[i]:
                f.append(env.envs[i].render())

        done |= terminated | truncated
        if done.all():
            break
    env.close()

    results = [
        {"seed": s, "success": bool(success[i]), "max_coverage": float(max_coverage[i]), "steps": int(steps[i])}
        for i, s in enumerate(seeds)
    ]
    return results, frames


def main() -> None:
    cfg = draccus.parse(config_class=EvalConfig)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", force=True)
    torch.manual_seed(cfg.torch_seed)

    policy_path = str(Path(cfg.policy_path).expanduser()) if Path(cfg.policy_path).expanduser().exists() else cfg.policy_path
    out_dir = cfg.out_dir or (Path(policy_path).parent / "eval" if Path(policy_path).exists() else RUNS_DIR / "eval")
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = DiffusionPolicy.from_pretrained(policy_path)
    policy.to(cfg.device).eval()
    # Normalisation from the full dataset's stats: identical to what training used.
    meta = LeRobotDatasetMetadata("lerobot/pusht")
    preprocessor, postprocessor = load_processors(policy, policy_path, meta)

    seeds = [cfg.start_seed + i for i in range(cfg.n_episodes)]
    results, t0 = [], time.perf_counter()
    for start in range(0, len(seeds), cfg.n_envs):
        chunk = seeds[start:start + cfg.n_envs]
        n_gifs = max(0, cfg.n_gifs - start)  # GIFs only from the first chunk(s)
        chunk_results, frames = run_vectorised(policy, preprocessor, postprocessor, chunk, cfg.max_steps, n_gifs)
        results.extend(chunk_results)
        for r, f in zip(chunk_results, frames):
            save_gif(f, out_dir / f"episode_{r['seed']}_{'success' if r['success'] else 'fail'}.gif")
        for r in chunk_results:
            logging.info("seed %d: success=%s max_coverage=%.2f steps=%d",
                         r["seed"], r["success"], r["max_coverage"], r["steps"])


    n_success = sum(r["success"] for r in results)
    low, high = wilson_interval(n_success, len(results))
    summary = {
        "policy_path": policy_path,
        "n_episodes": len(results),
        "success_rate": n_success / len(results),
        "success_ci95": [low, high],
        "mean_max_coverage": float(np.mean([r["max_coverage"] for r in results])),
        "eval_seconds": time.perf_counter() - t0,
        "per_episode": results,
    }
    (out_dir / "eval_results.json").write_text(json.dumps(summary, indent=2))
    logging.info("success %.1f%% (95%% CI %.1f-%.1f%%), mean max coverage %.2f, %.0f s",
                 100 * summary["success_rate"], 100 * low, 100 * high,
                 summary["mean_max_coverage"], summary["eval_seconds"])


if __name__ == "__main__":
    main()