# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Walk training configuration.

Ordinary Isaac Lab configclasses; edit these values to train a new experiment.
"""
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl.rl_cfg import RslRlPpoActorCriticCfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl.rl_cfg import RslRlPpoAlgorithmCfg


@configclass
class G1WalkPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunnerLip"
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 8000
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation="elu"
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOLIP",
        grad_penalty_coef=0,
        value_loss_coef=1,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.001,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1,
        symmetry_cfg={
            "use_mirror_loss": False,
            "mirror_loss_coeff": 0,
            "use_data_augmentation": True,
            "data_augmentation_func": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.walk_symmetry:data_augmentation_func_walk",
        },
    )
    save_interval = 250
    experiment_name = "g1_walk"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "isaaclab"
    wandb_project = "g1_walk"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    no_normalize_heightmap = False
    heightmap_dim = 0
    load_warm_up = False
    save_env_state = False
