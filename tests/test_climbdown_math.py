"""Execute reward formulas on CPU tensors, without importing Isaac Sim.

AST extraction executes the actual delivered definitions, not replicas of their
formulas. PyTorch is supplied by the training environment; dependency-light
inspection hosts skip this file explicitly.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="Reward formula checks require CPU PyTorch")
ROOT = Path(__file__).resolve().parents[1]
TASK_REWARDS = "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/mdp/locomotion_rewards.py"
TERMINATIONS = "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/mdp/terminations.py"


class Selector(SimpleNamespace):
    def __init__(self, name="robot", **kwargs):
        super().__init__(name=name, **kwargs)


class Scene(dict):
    pass


def _load(relative, name):
    path = ROOT / relative
    source = ast.parse(path.read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == name)
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(tree)
    namespace = {"torch": torch, "SceneEntityCfg": Selector}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[name]


def _environment(count):
    data = SimpleNamespace(
        body_pos_w=torch.zeros(count, 4, 3, dtype=torch.float64),
        body_com_pos_w=torch.zeros(count, 4, 3, dtype=torch.float64),
        body_lin_vel_w=torch.zeros(count, 4, 3, dtype=torch.float64),
        body_ang_vel_w=torch.zeros(count, 4, 3, dtype=torch.float64),
        joint_pos=torch.zeros(count, 2, dtype=torch.float64),
    )
    scene = Scene(robot=SimpleNamespace(data=data))
    scene.env_origins = torch.zeros(count, 3, dtype=torch.float64)
    scene.sensors = {"bodies_ground_contact": SimpleNamespace()}
    return SimpleNamespace(scene=scene, episode_length_buf=torch.full((count,), 201, dtype=torch.int64))


def test_knee_termination_strict_thresholds_early_window_and_both_joints():
    fn = _load(TERMINATIONS, "knee_straight_down")
    env = _environment(10)
    env.scene["robot"].data.joint_pos[:] = torch.tensor([
        [.299, .299], [.299, .299], [.3, .2], [.09, .09], [.1, .09],
        [.09, .11], [-.1, .09], [.31, .29], [.1, .1], [.1, .1],
    ], dtype=torch.float64)
    env.episode_length_buf[:] = torch.tensor([49, 50, 49, 500, 500, 500, 500, 0, 49, 50])
    expected = torch.tensor([True, False, False, True, False, False, True, False, True, False])
    actual = fn(env, asset_cfg=Selector(joint_ids=[0, 1]))
    assert actual.shape == (10,)
    assert actual.dtype == torch.bool
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("name,velocity_field,kernel", [
    ("standing_lin_vel_down", "body_lin_vel_w", 5.0),
    ("standing_ang_vel_down", "body_ang_vel_w", 2.0),
])
def test_velocity_rewards_gate_steps_both_feet_and_use_torso_speed(name, velocity_field, kernel):
    fn = _load(TASK_REWARDS, name)
    env = _environment(7)
    data = env.scene["robot"].data
    data.body_pos_w[:, :2, 2] = .099
    env.episode_length_buf[1] = 200
    data.body_pos_w[2, 1, 2] = .1  # equality is outside the gate
    data.body_pos_w[3, 1, 2] = .2  # both feet must be low
    getattr(data, velocity_field)[4, 2] = torch.tensor([1., 2., 0.], dtype=torch.float64)
    getattr(data, velocity_field)[5, 0] = 100.0  # foot speed is not torso speed
    env.scene.env_origins[6, 2] = 1.0
    data.body_pos_w[6, :2, 2] = 1.05  # gate uses world z, not relative z
    actual = fn(env, feet_cfg=Selector(body_ids=[0, 1]), asset_cfg=Selector(body_ids=[2]))
    expected = torch.tensor([1., 0., 0., 0., torch.exp(torch.tensor(-kernel * 5., dtype=torch.float64)).item(), 1., 0.], dtype=torch.float64)
    assert actual.shape == (7,)
    torch.testing.assert_close(actual, expected)


class Command:
    def __init__(self, minimum, mass):
        self.min_com_x = minimum.clone()
        self.mass = mass
        self.updates = []

    def update_min_com_x(self, current):
        self.updates.append(current.clone())
        self.min_com_x = torch.minimum(self.min_com_x, current)


def _forward_environment(relative_x, minima, upper_z=None, origin_z=None, masses=None):
    x = torch.tensor(relative_x, dtype=torch.float64)
    env = _environment(len(relative_x))
    # Distinct nonzero x origins catch accidental use of absolute position.
    env.scene.env_origins[:, 0] = torch.arange(len(relative_x), dtype=torch.float64) * 3.
    env.scene["robot"].data.body_pos_w[:, :, 0] = x + env.scene.env_origins[:, 0, None]
    if origin_z is not None:
        env.scene.env_origins[:, 2] = torch.tensor(origin_z, dtype=torch.float64)
    if upper_z is not None:
        env.scene["robot"].data.body_com_pos_w[:, 2:, 2] = torch.tensor(upper_z, dtype=torch.float64)
    command = Command(torch.tensor(minima, dtype=torch.float64), torch.ones_like(x) if masses is None else torch.tensor(masses, dtype=torch.float64))
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    return env, command


def _forward(fn, env):
    return fn(env, upper_cfg=Selector(body_ids=[2, 3]), upper_sensor_cfg=Selector(name="bodies_ground_contact", body_ids=[2, 3]),
              sensor_cfg=Selector(name="bodies_ground_contact", body_ids=[0, 1]), asset_cfg=Selector(body_ids=[0, 1]))


def test_forward_penalty_progress_tolerance_mass_weighting_clamp_and_state_update():
    fn = _load(TASK_REWARDS, "com_forward_penalty")
    xs = [[-.5] * 4] * 4 + [[-2., -.8, -.5, -.2]]
    env, command = _forward_environment(xs, [-.6, -.5, -.49995, -.4998, -.35],
                                        masses=[[1.] * 4] * 4 + [[1., 1., 2., 4.]])
    # The final weighted mean uses per-body x clipped at -1: (-1-.8-1-.8)/8 = -.45.
    actual = _forward(fn, env)
    assert actual.shape == (5,)
    torch.testing.assert_close(actual, torch.tensor([1., 1., 1., 0., 0.], dtype=torch.float64))
    expected_current = torch.tensor([-.5, -.5, -.5, -.5, -.45], dtype=torch.float64)
    assert len(command.updates) == 1
    torch.testing.assert_close(command.updates[0], expected_current)
    torch.testing.assert_close(command.min_com_x, torch.tensor([-.6, -.5, -.5, -.5, -.45], dtype=torch.float64))
    # With no further progress, the next call is penalized even in rows that
    # previously improved their running minimum.
    torch.testing.assert_close(_forward(fn, env), torch.ones(5, dtype=torch.float64))
    assert len(command.updates) == 2


def test_forward_success_requires_all_bodies_back_and_upper_com_height():
    fn = _load(TASK_REWARDS, "com_forward_penalty")
    env, command = _forward_environment(
        [[-.9] * 4, [-.8, -.9, -.9, -.9], [-.9] * 4, [-.9] * 4, [-.9] * 4],
        [-1.] * 5,
        upper_z=[[.0501, .0501], [.1, .1], [.05, .1], [.829, .84], [.831, .831]],
        origin_z=[0., 0., 0., 1., 1.],
    )
    # Row 3 is below the upper-body height threshold after subtracting .78;
    # row 4 proves origin z is clipped at .78 before subtracting.
    actual = _forward(fn, env)
    torch.testing.assert_close(actual, torch.tensor([0., 1., 1., 1., 0.], dtype=torch.float64))
    assert len(command.updates) == 1  # success still updates progress state

