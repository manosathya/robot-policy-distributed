# Distributed training of a robot imitation-learning policy

Diffusion Policy (LeRobot) trained on the PushT task with a PyTorch training loop:
single GPU first, then DDP and FSDP across multiple GPUs, with profiling and resume tests.
Personal project, work in progress.

## Stage 1 results: single GPU

Evaluation: 50 episodes per checkpoint, fixed seeds, run as 50 parallel environments.
Success = the block covers more than 95% of the target within 300 steps.

| Run | Steps | Success (95% CI) | Mean max coverage | Eval time |
|---|---|---|---|---|
| LeRobot reference (my evaluator) | 200k | 66.0% (52.2–77.6) | 0.90 | 149 s |
| stage1_20k @ 10k | 10k | 4.0% (1.1–13.5) | 0.65 | 163 s |
| stage1_20k @ 20k | 20k | 30.0% (19.1–43.8) | 0.84 | 156 s |

Training: ~239 samples/s, 7.18 GB peak GPU memory, 1.7 h for 20k steps on an RTX 4060 Laptop GPU (8 GB).
Batching evaluation across 50 environments cut a 50-episode evaluation from ~70 min to ~2.5 min.