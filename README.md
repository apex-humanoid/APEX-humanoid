# APEX-humanoid

Six whole-body locomotion skills for the Unitree G1 humanoid: walk, crawl, climb up, climb down, stand up, and lie down. Built with **Isaac Lab 1.3.0** and **Isaac Sim 4.2.0**.

## Installation

Requires Linux, an NVIDIA GPU, Git, and an **Isaac Sim 4.2.0** binary installation with Python 3.10. Isaac Lab and RSL-RL are included in this repository.

```bash
git clone https://github.com/apex-humanoid/APEX-humanoid.git
cd APEX-humanoid
ln -s /path/to/isaac-sim _isaac_sim
./isaaclab.sh -i rsl_rl
```

Replace `/path/to/isaac-sim` with your installation directory. Use the simulator's bundled Python without activating another Conda environment. For pip installation and container dependencies, see [installation instructions](docs/installation.md).

## Skills

| Skill | Task | Pretrained policy |
|---|---|---|
| Walk | `Isaac-G1-Velocity-Walk` | `checkpoints/walk.pt` |
| Crawl | `Isaac-G1-Velocity-Crawl` | `checkpoints/crawl.pt` |
| Climb up | `Isaac-ClimbUp-G1` | `checkpoints/climbup.pt` |
| Climb down | `Isaac-Down-G1` | `checkpoints/climbdown.pt` |
| Stand up | `Isaac-StandUp-G1` | `checkpoints/standup.pt` |
| Lie down | `Isaac-LyDown-G1` | `checkpoints/liedown.pt` |

## Train

Run from the repository root, replacing `--task` with a task from the table:

```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-G1-Velocity-Walk --headless
```

Use `--num_envs` to adjust the number of parallel environments and `--max_iterations` to set the training budget. Logs and checkpoints are saved to `logs/rsl_rl/`.

Environment configurations and PPO settings are in [config/g1](source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/config/g1/).

To resume training, add `--resume --checkpoint /path/to/model_5000.pt` to the training command. Resuming requires a full training checkpoint; the bundled policies below are for inference.

## Run a pretrained policy

The six pretrained policies are included in `checkpoints/`. Select the matching task and policy from the table:

```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
  --task Isaac-G1-Velocity-Walk \
  --checkpoint checkpoints/walk.pt --torchscript --num_envs 1
```

For a server without a display, add `--headless --steps 1000`. To run a full checkpoint produced by training, pass its `model_*.pt` path and omit `--torchscript`.

## License

See [LICENSE](LICENSE) and the third-party notices in [docs/licenses](docs/licenses/) and [source/rsl_rl/licenses](source/rsl_rl/licenses/).
