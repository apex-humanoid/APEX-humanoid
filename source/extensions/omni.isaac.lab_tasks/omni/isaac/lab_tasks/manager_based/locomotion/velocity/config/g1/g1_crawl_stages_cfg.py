# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Crawl training stage configurations."""

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import EventTermCfg
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.events import (
    push_by_adding_velocity,
    reset_joints_to_lying_by_offset,
)
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.rewards import crawl_joint_deviation_l2

from .g1_crawl_cfg import EventCfg, G1CrawlEnvCfg, G1Rewards


# configclass deep-copies these member defaults into each instance. Sharing the
# unchanged definitions avoids copying their numerical values into another cfg.
_final_events = EventCfg()
_final_rewards = G1Rewards()


@configclass
class G1CrawlStage1EventsCfg:
    """Crawl training stage settings."""

    base_com = _final_events.base_com
    physics_material = _final_events.physics_material
    add_base_mass = _final_events.add_base_mass
    add_joint_default_pos = _final_events.add_joint_default_pos
    reset_base_on_ground = _final_events.reset_base_on_ground
    reset_joint_lying = _final_events.reset_joint_lying
    push_robot = EventTermCfg(
        func=push_by_adding_velocity,
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        mode="interval",
        interval_range_s=(4.5, 5.5),
    )


@configclass
class G1CrawlStage1RewardsCfg:
    """Crawl training stage settings."""

    track_lin_vel_xy_exp = _final_rewards.track_lin_vel_xy_exp
    track_ang_vel_z_exp = _final_rewards.track_ang_vel_z_exp
    lin_vel_z = _final_rewards.lin_vel_z
    ang_vel_xy = _final_rewards.ang_vel_xy
    base_height = _final_rewards.base_height
    joint_acc_penalty = _final_rewards.joint_acc_penalty
    joint_vel_penalty = _final_rewards.joint_vel_penalty
    action_rate_penalty = _final_rewards.action_rate_penalty
    joint_pos_limits = _final_rewards.joint_pos_limits
    alive = _final_rewards.alive
    termination = _final_rewards.termination
    joint_deviation_penalty = _final_rewards.joint_deviation_penalty.replace(
        func=crawl_joint_deviation_l2, weight=-0.1
    )
    torque_penalty = _final_rewards.torque_penalty
    undesired_contact = _final_rewards.undesired_contact


@configclass
class G1CrawlStage2EnvCfg(G1CrawlEnvCfg):
    """Crawl training stage settings."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_joint_lying.func = reset_joints_to_lying_by_offset
        self.events.reset_joint_lying.params["position_range"] = (-0.15, 0.15)


@configclass
class G1CrawlStage1EnvCfg(G1CrawlStage2EnvCfg):
    """Crawl training stage settings."""

    events = G1CrawlStage1EventsCfg()
    rewards = G1CrawlStage1RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.pos = (0, 0.0, 0.35)
        self.actions.joint_pos.action_start_step = 15
        self.rewards.undesired_contact.params["sensor_cfg"].body_names = [
            "torso_link",
            "head_link",
            ".*shoulder.*",
            "pelvis",
            ".*hip_pitch_link",
            ".*hip_roll_link",
            ".*ankle.*",
        ]
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
