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
| stage1_100k @ 25k | 25k | 34.0% (22.4–47.8) | 0.87 | 153 s |
| stage1_100k @ 50k | 50k | 40.0% (27.6–53.8) | 0.90 | 160 s |
| stage1_100k @ 75k | 75k | 56.0% (42.3–68.8) | 0.90 | 163 s |
| stage1_100k @ 100k | 100k | **62.0% (48.2–74.1)** | 0.92 | 162 s |

At 100k steps, half the reference's training length, the hand-written training loop reaches a success rate
within the confidence interval of LeRobot's reference checkpoint.

Training: ~258 samples/s, 7.18 GB peak GPU memory, 7.2 h for 100k steps on an RTX 4060 Laptop GPU (8 GB).
Batching evaluation across 50 environments cut a 50-episode evaluation from ~70 min to ~2.5 min.

## Next

- Stage 2: DDP, bf16 mixed precision and FSDP2 with sharded checkpoints (in progress).
- Stage 3: scaling on 1, 2 and 4 rented GPUs, profiling and a bottleneck fix, a kill-and-resume test.
- Stage 4: dataset-fraction ablation (25/50/100% of demonstrations).