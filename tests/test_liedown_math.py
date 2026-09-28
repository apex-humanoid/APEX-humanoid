"""Check contact rewards and orientation boundaries."""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="Reward checks require PyTorch")
ROOT = Path(__file__).resolve().parents[1]
REWARDS = ROOT / "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/mdp/rewards.py"


class BaseTerm:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env


class Scene(dict):
    pass


def make_reward(count):
    node = next(n for n in ast.parse(REWARDS.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "lying_contact_down")
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(tree)
    namespace = {"torch": torch, "ManagerTermBase": BaseTerm,
                 "math_utils": SimpleNamespace(quat_rotate_inverse=lambda quat, gravity: gravity)}
    exec(compile(tree, str(REWARDS), "exec"), namespace)
    body_pos = torch.zeros(count, 11, 3)
    body_pos[:, :, 2] = 0.5
    body_pos[:, [3, 7], 2] = 0.1  # wrists are lower than their elbows
    data = SimpleNamespace(body_pos_w=body_pos, body_quat_w=torch.zeros(count, 11, 4),
                           GRAVITY_VEC_W=torch.tensor([1., 0., 0.]).repeat(count, 1))
    lookups = []
    groups = {"left_upper": [0, 1, 2, 3], "right_upper": [4, 5, 6, 7], "left_lower": [8], "right_lower": [9]}
    def find_bodies(names, preserve_order):
        assert preserve_order is True
        lookups.append(names)
        return groups[names], []
    scene = Scene(robot=SimpleNamespace(data=data, find_bodies=find_bodies))
    forces = torch.zeros(count, 10, 1, 3)
    scene.sensors = {"ground": SimpleNamespace(_data=SimpleNamespace(force_matrix_w=forces))}
    params = {"asset_cfg": SimpleNamespace(name="robot", body_ids=[10])}
    for name, ids in zip(groups, ([9, 8, 7, 6], [5, 4, 3, 2], [1], [0])):
        params[name + "_cfg"] = SimpleNamespace(name="ground", body_names=name, body_ids=ids)
    env = SimpleNamespace(scene=scene, episode_length_buf=torch.full((count,), 201))
    reward = namespace["lying_contact_down"](SimpleNamespace(params=params), env)
    return reward, env, params, forces, lookups


@pytest.mark.parametrize("count", [1, 4, 32, 4096])
def test_wrist_and_single_knee_groups_produce_one_reward_per_environment(count):
    reward, env, params, forces, lookups = make_reward(count)
    actual = reward(env, **params)
    expected = 2 * math.exp(-0.15) + 2 * math.exp(-3.75)
    assert actual.shape == (count,)
    torch.testing.assert_close(actual, torch.full((count,), expected))
    forces[:, 6, 0, 2] = 1.001  # left wrist contact in the sensor's different ordering
    actual = reward(env, **params)
    torch.testing.assert_close(actual, torch.full((count,), 1 + math.exp(-0.15) + 2 * math.exp(-3.75)))
    assert len(lookups) == 4  # body lookup is cached


def test_contact_orientation_and_episode_boundaries():
    reward, env, params, forces, _ = make_reward(4)
    env.scene["robot"].data.body_pos_w[:, :, 2] = 0.5
    forces[:, :, 0, 2] = 1.001
    env.episode_length_buf[0] = 200
    env.scene["robot"].data.GRAVITY_VEC_W[1, 0] = 0.75
    env.scene["robot"].data.GRAVITY_VEC_W[2, 0] = 0.75001
    forces[3, :, 0, 2] = 1.0
    expected = torch.tensor([0., 2., 4., 4 * math.exp(-3.75)])
    torch.testing.assert_close(reward(env, **params), expected)


def test_height_reduction_preserves_signed_world_height_semantics():
    reward, env, params, _, _ = make_reward(1)
    env.scene["robot"].data.body_pos_w[:, 0, 2] = -0.4
    expected = math.exp(-2.4) + math.exp(-0.15) + 2 * math.exp(-3.75)
    torch.testing.assert_close(reward(env, **params), torch.tensor([expected]))


def load_function(name):
    node = next(n for n in ast.parse(REWARDS.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    namespace = {"torch": torch, "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
                 "math_utils": SimpleNamespace(quat_rotate_inverse=lambda quat, gravity: gravity)}
    exec(compile(ast.fix_missing_locations(tree), str(REWARDS), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("height,step,gravity_z,expected", [
    (0.29, 201, -0.6, 0.8), (0.30, 201, -0.6, 0.0),
    (0.325, 201, -0.6, 0.0), (0.35, 201, -0.6, 0.0),
    (0.29, 200, -0.6, 0.0), (0.29, 201, 0.6, 0.9),
])
def test_launch_orientation_height_and_time_boundaries(height, step, gravity_z, expected):
    positions = torch.zeros(2, 4, 3)
    positions[:, :, 2] = height
    data = SimpleNamespace(body_pos_w=positions, projected_gravity_b=torch.tensor([[0.8, 0., gravity_z]]).repeat(2, 1))
    scene = Scene(robot=SimpleNamespace(data=data))
    scene.env_origins = torch.zeros(2, 3)
    env = SimpleNamespace(scene=scene, episode_length_buf=torch.full((2,), step))
    actual = load_function("lydown_orientation")(env, SimpleNamespace(name="robot"))
    torch.testing.assert_close(actual, torch.full((2,), expected))
