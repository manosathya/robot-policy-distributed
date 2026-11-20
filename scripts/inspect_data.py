# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt

import torch
from torchvision.utils import save_image

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

REPO_ID = "lerobot/pusht"
OUT = Path.home() / "runs" / "inspect"
OUT.mkdir(parents=True, exist_ok=True)

# %% Metadata Analysis
meta = LeRobotDatasetMetadata(REPO_ID)

print("fps:", meta.fps)
print("episodes:", meta.total_episodes, "frames:", meta.total_frames)
for key, ft in meta.features.items():
    print(f"  {key:22s} dtype={ft['dtype']:8s} shape={ft['shape']}")
print("action min/max:", meta.stats["action"]["min"], meta.stats["action"]["max"])
print("state  min/max:", meta.stats["observation.state"]["min"], meta.stats["observation.state"]["max"])


# %% plain dataset

plain = LeRobotDataset(REPO_ID)
item = plain[301]
print("\nplain item:")
for k, v in item.items():
    print(f"  {k:22s} {tuple(v.shape) if torch.is_tensor(v) else v!r}, {v}")

# %% Windowed Dataset

fps = meta.fps
delta_timestamps = {
    "observation.image": [-1 / fps, 0.0],
    "observation.state": [-1 / fps, 0.0],
    "action": [i / fps for i in range(-1, 15)],
}

windowed = LeRobotDataset(REPO_ID, delta_timestamps=delta_timestamps)

first = windowed[0]
print("\nwindowed item at an episode start:")
for k, v in first.items():
    print(f"  {k:22s} {tuple(v.shape) if torch.is_tensor(v) else v!r}")
print("action_is_pad at frame 0:", first["action_is_pad"].tolist())


# %% Save windowed Image (two frames)
save_image(windowed[50]["observation.image"], OUT / "obs_pair_50.png")

# %% Action trajectory of one episode
ep = 0
start = meta.episodes["dataset_from_index"][ep]
end = meta.episodes["dataset_to_index"][ep]

actions = torch.stack([plain[i]["action"] for i in range(start, end)])
print(f"\nepisode {ep}: frames {start}..{end - 1} ({end - start} frames, {(end - start) / fps:.1f} s)")
print("action_is_pad at the last frame:", windowed[end - 1]["action_is_pad"].tolist())


fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(actions[:, 0], label="x"); ax[0].plot(actions[:, 1], label="y")
ax[0].set_xlabel("frame"); ax[0].set_ylabel("target position (px)"); ax[0].legend()
ax[1].plot(actions[:, 0], actions[:, 1]); ax[1].invert_yaxis(); ax[1].set_aspect("equal")
ax[1].set_title("action path")
fig.tight_layout(); fig.savefig(OUT / f"episode_{ep}_actions.png", dpi=120)
print("figures written to", OUT)
# %%
