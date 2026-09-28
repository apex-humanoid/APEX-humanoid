"""Check walk actions, rewards, terrain, and symmetry without Isaac Sim."""
from __future__ import annotations

import ast
import copy
import functools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / 'source/extensions/omni.isaac.lab/omni/isaac/lab'
MDP = ROOT / 'source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/mdp'


class Selector(SimpleNamespace):
    def __init__(self, name='robot', **kwargs):
        super().__init__(name=name, **kwargs)


def definitions(path, names, namespace=None, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == class_name)
    selected = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    namespace = dict(namespace or {}, torch=torch, SceneEntityCfg=Selector)
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


def function(module, name):
    return definitions(MDP / (module + '.py'), {name})[name]


def test_walk_action_copies_previous_full_body_targets():
    process = definitions(MDP / 'walk_actions.py', {'process_actions'}, class_name='WalkJointAction')['process_actions']
    ids = [0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]
    offset = torch.linspace(-.3, .3, 29).repeat(2, 1)
    state = SimpleNamespace(
        _raw_actions=torch.zeros(2, 12), _processed_actions=torch.zeros(2, 29),
        last_processed_actions=torch.zeros(2, 29), _offset=offset, _scale=.25,
        _joint_ids=ids, cfg=SimpleNamespace(clip=True),
        _clip=torch.tensor([-100., 100.]).repeat(2, 12, 1),
    )
    state.processed_actions = state._processed_actions
    process(state, torch.ones(2, 12))
    previous = state._processed_actions.clone()
    process(state, torch.full((2, 12), 2.))
    torch.testing.assert_close(state.last_processed_actions, previous)
    assert state.last_processed_actions.data_ptr() != state._processed_actions.data_ptr()
    expected = offset.clone()
    expected[:, ids] += .5
    torch.testing.assert_close(state._processed_actions, expected)
    reward = function('rewards', 'processed_action_rate_l2')
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: state))
    torch.testing.assert_close(reward(env, 'joint_pos'), torch.full((2,), 12 * .25**2))


@pytest.mark.parametrize('name,field', [('joint_vel_with_action', 'joint_vel'), ('joint_acc_with_action', 'joint_acc')])
def test_walk_joint_penalties_use_action_joint_indices(name, field):
    values = torch.full((2, 29), 100.)
    values[:, [0, 3]] = torch.tensor([2., 3.])
    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(**{field: values}))},
                          action_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(_joint_ids=[0, 3])))
    reward = function('walk_rewards', name)
    # asset_cfg deliberately selects every joint; only the action's joints contribute.
    torch.testing.assert_close(reward(env, 'joint_pos', Selector(joint_ids=slice(None))), torch.full((2,), 13.))


def test_xy_deadband_keeps_small_yaw_command():
    sample = definitions(LAB / 'envs/mdp/commands/velocity_command.py', {'_resample_command'},
                         class_name='UniformVelocityCommand')['_resample_command']
    state = SimpleNamespace(device='cpu', vel_command_b=torch.zeros(2, 3), is_standing_env=torch.zeros(2, dtype=torch.bool),
                            cfg=SimpleNamespace(planar_deadband=.1, heading_command=False, rel_standing_envs=0.,
                                                ranges=SimpleNamespace(lin_vel_x=(.05, .05), lin_vel_y=(-.05, -.05), ang_vel_z=(.05, .05))))
    sample(state, [0, 1])
    torch.testing.assert_close(state.vel_command_b, torch.tensor([[0., 0., .05], [0., 0., .05]]))


def test_terrain_origin_uses_two_meter_window():
    def terrain(difficulty, cfg):
        heights = np.zeros(tuple(round(v / cfg.horizontal_scale) for v in cfg.size), dtype=np.int16)
        heights[9, 17] = 1000  # Inside +/-1m, outside +/-0.5m after adding the border.
        return heights
    cfg = SimpleNamespace(size=(4., 4.), border_width=.2, horizontal_scale=.1, vertical_scale=.005, slope_threshold=.75)
    ns = definitions(LAB / 'terrains/height_field/utils.py', {'height_field_to_mesh', 'convert_height_field_to_mesh'},
                     {'copy': copy, 'functools': functools, 'np': np,
                      'trimesh': SimpleNamespace(Trimesh=lambda **kwargs: SimpleNamespace(**kwargs))})
    meshes, origin = ns['height_field_to_mesh'](terrain)(.5, cfg)
    np.testing.assert_allclose(origin, [2., 2., 5.])
    assert cfg.size == (4., 4.)
    assert meshes[0].vertices.shape == (41 * 41, 3)


@pytest.mark.parametrize('is_critic,width,command_start', [(False, 588, 36), (True, 606, 54)])
def test_walk_symmetry_keeps_layout_yaw_and_phase(is_critic, width, command_start):
    common_names = [n.name for n in ast.parse((MDP / 'symmetry.py').read_text()).body if isinstance(n, ast.FunctionDef)]
    ns = definitions(MDP / 'symmetry.py', common_names, {'_CACHED_INDICES': {}, '_CACHED_INDICES_climbup': {}, '_CACHED_INDICES_crawl': {}})
    ns['_get_mapping_indices'] = ns['_get_mapping_indices_crawl']
    ns['_remap_and_flip_joints'] = ns['_remap_and_flip_joints_crawl']
    walk_names = [n.name for n in ast.parse((MDP / 'walk_symmetry.py').read_text()).body if isinstance(n, ast.FunctionDef)]
    augment = definitions(MDP / 'walk_symmetry.py', walk_names, ns)['data_augmentation_func_walk']
    offset = torch.linspace(-.4, .4, 29).reshape(1, 29)
    env = SimpleNamespace(env=SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(default_joint_pos=offset))}))
    obs = torch.linspace(-1., 1., 2 * width).reshape(2, width)
    actions = torch.linspace(-1., 1., 24).reshape(2, 12)
    mirrored, mirrored_actions = augment(obs, actions, env, is_critic)
    assert mirrored.shape == (4, width) and mirrored_actions.shape == (4, 12)
    before = obs[:, command_start:command_start+18].reshape(2, 6, 3)
    after = mirrored[2:, command_start:command_start+18].reshape(2, 6, 3)
    torch.testing.assert_close(after[:, :, 1], -before[:, :, 1])
    torch.testing.assert_close(after[:, :, 2], before[:, :, 2])  # Yaw retains its sign.
    torch.testing.assert_close(mirrored[2:, -12:], obs[:, -12:])
    twice, twice_actions = augment(mirrored[2:], mirrored_actions[2:], env, is_critic)
    torch.testing.assert_close(twice[2:], obs)
    torch.testing.assert_close(twice_actions[2:], actions)


def test_walk_height_reward_and_termination_keep_different_frames():
    class Scene(dict): pass
    positions = torch.tensor([[0., 0., .78], [0., 0., 2.78], [0., 0., .3]])
    scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=positions, root_link_pos_w=positions)))
    scene.env_origins = torch.tensor([[0., 0., 0.], [0., 0., 2.], [0., 0., 0.]])
    env = SimpleNamespace(scene=scene)
    reward = function('walk_locomotion_rewards', 'base_height')
    terminate = function('terminations', 'root_height_below_minimum')
    torch.testing.assert_close(reward(env, .78), torch.tensor([0., 0., .48**2]), atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(terminate(env, .35), torch.tensor([False, False, True]))
