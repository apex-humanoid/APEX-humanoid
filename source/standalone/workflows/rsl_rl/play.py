# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play an RSL-RL training checkpoint or exported TorchScript policy."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RSL-RL checkpoint or TorchScript policy.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--follow_robot",
    action="store_true",
    help="Keep the playback camera centered on the robot in environment 0.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument(
    "--steps", type=int, default=None, help="Stop after this many simulation steps."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Override the environment seed for playback.")
parser.add_argument(
    "--torchscript",
    action="store_true",
    help="Load --checkpoint as a TorchScript actor with its embedded observation normalizer.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.torchscript and args_cli.checkpoint is None:
    parser.error("--torchscript requires --checkpoint pointing to an exported policy file")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunnerLip

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli
    )

    if args_cli.seed is not None:
        env_cfg.seed = agent_cfg.seed

    if args_cli.follow_robot:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (3.0, -3.0, 2.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    if args_cli.torchscript:
        resume_path = os.path.abspath(args_cli.checkpoint)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"TorchScript policy file not found: {resume_path}")
    else:
        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = (
            os.path.abspath(agent_cfg.load_checkpoint)
            if os.path.isfile(agent_cfg.load_checkpoint)
            else get_checkpoint_path(
                log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
            )
        )
    log_dir = os.path.dirname(resume_path)

    env_cfg.scene.terrain.num_envs = env_cfg.scene.num_envs
    agent_cfg.device = env_cfg.sim.device
    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    # wrap for video recording
    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos", "play")
        if args_cli.torchscript:
            video_folder = os.path.join(video_folder, os.path.splitext(os.path.basename(resume_path))[0])
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if args_cli.torchscript:
        # The exported module already contains the actor and observation normalizer.
        policy = torch.jit.load(resume_path, map_location=env.unwrapped.device).eval()
    else:
        # load previously trained model
        ppo_runner = OnPolicyRunnerLip(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        ppo_runner.load(resume_path, warm_up=False)

        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        # export policy to onnx/jit
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.policy,
            ppo_runner.obs_normalizer,
            path=export_model_dir,
            filename="policy.pt",
        )
        export_policy_as_onnx(
            ppo_runner.alg.policy,
            normalizer=ppo_runner.obs_normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        if args_cli.steps is not None and timestep >= args_cli.steps:
            break
        timestep += 1
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    print(f"[INFO] Playback completed successfully. Steps: {timestep}.", flush=True)
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
