# Pretrained policies

The files in `checkpoints/` are TorchScript policies containing an actor and its observation normalizer. Select the matching task and use `--torchscript`.

| Skill | Task | Policy | Observations / actions |
|---|---|---|---|
| Walk | `Isaac-G1-Velocity-Walk` | `walk.pt` | 588 / 12 |
| Crawl | `Isaac-G1-Velocity-Crawl` | `crawl.pt` | 588 / 23 |
| Climb up | `Isaac-ClimbUp-G1` | `climbup.pt` | 1234 / 23 |
| Climb down | `Isaac-Down-G1` | `climbdown.pt` | 1234 / 23 |
| Stand up | `Isaac-StandUp-G1` | `standup.pt` | 558 / 23 |
| Lie down | `Isaac-LyDown-G1` | `liedown.pt` | 558 / 23 |

After [installation](installation.md), run from the repository root:

```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
  --task Isaac-G1-Velocity-Walk \
  --checkpoint checkpoints/walk.pt --torchscript --num_envs 1
```

Choose another task and policy from the table as needed. For a server without a display, add `--headless --steps 1000`. Files must be present locally.

Use `--seed 42` to select an environment seed. Task observation noise, domain randomization and command sampling remain enabled.

The bundled policies support inference, not training resume: they do not contain a critic or optimizer state. To resume training or play a full training checkpoint, use `model_*.pt` with the corresponding train/play command and omit `--torchscript`.

File sizes, SHA256 values and dimensions are in [manifest.json](../checkpoints/manifest.json). On Linux, check the files with:

```bash
(cd checkpoints && sha256sum -c SHA256SUMS)
```
