# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from omni.isaac.lab.utils import configclass


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

@configclass
class RslRlPpoCompositeStudentTeacherCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "CompositeStudentTeacher"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    student_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    adapters_indices:  list[list[int]]  = MISSING


@configclass
class RslRlPpoStudentTeacherCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "StudentTeacher"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    student_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    teacher_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""



@configclass
class RslRlDistillAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "Distillation"
    """The algorithm class name. Default is PPO."""

    loss_type: str = MISSING

    gradient_length: int = MISSING

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""



@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    grad_penalty_coef: float = MISSING
    """The coefficient for the gradient penalty."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""
    
    rnd_cfg: dict | None = None
    
    symmetry_cfg: dict | None = None


@configclass
class RslRlOnPolicyRunnerCfg:
    load_warm_up: bool = False
    save_env_state: bool = False

    """Configuration of the runner for on-policy algorithms."""
    class_name: str = "OnPolicyRunner"

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg|RslRlPpoCompositeStudentTeacherCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg | RslRlDistillAlgorithmCfg = MISSING
    """The algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

    no_normalize_heightmap: bool = False
    ''' When normalizing, do not normalize heightmap.
    '''

    heightmap_dim: int = 0
    ''' Dimension of height map.
    '''

@configclass
class OfflineDistillAlgorithmCfg:
    """Configuration for the Offline Distillation algorithm."""

    class_name: str = "OfflineDistillation"
    """The algorithm class name. Default is PPO."""

    loss_type: str = MISSING

    gradient_length: int = MISSING
    # similar to batch_steps

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    initial_epoches: int = 2000
    incremental_epoches: int = 500
    final_epoches: int = 0
    symmetry_cfg: dict | None = None

    # batch_steps: int = 20000
    # distill_lr = 5.0e-4
    # final_lr = 5.0e-4