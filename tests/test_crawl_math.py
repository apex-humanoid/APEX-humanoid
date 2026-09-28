"""Check crawl behavior on CPU tensors without loading Isaac Sim."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch', reason='Crawl checks require CPU PyTorch')
ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / 'source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/mdp'


class Selector(SimpleNamespace):
    def __init__(self, name='robot', **kwargs):
        super().__init__(name=name, **kwargs)


class Scene(dict):
    pass


def _functions(module):
    path = MDP / module
    tree = ast.parse(path.read_text())
    tree.body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + [
        node for node in tree.body if isinstance(node, ast.FunctionDef)]
    ast.fix_missing_locations(tree)
    namespace = {'torch': torch, 'SceneEntityCfg': Selector, '_CACHED_INDICES': {}}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


def test_phase_zeroes_standing_and_low_xy_commands_with_strict_threshold():
    phase = _functions('observations.py')['tri_phase']
    commands = SimpleNamespace(
        vel_command_b=torch.tensor([[.099, -.099, 0], [.1, 0, 0], [.2, .2, 0], [0, 0, 1]], dtype=torch.float64),
        is_standing_env=torch.tensor([False, False, True, False]),
    )
    env = SimpleNamespace(num_envs=4, device='cpu', episode_length_buf=torch.full((4,), 2),
                          step_dt=.02, command_manager=SimpleNamespace(get_term=lambda name: commands))
    result = phase(env, period=.65, command_name='base_velocity')
    assert result.shape == (4, 2)
    torch.testing.assert_close(result[[0, 2, 3]], torch.zeros(3, 2))
    assert torch.linalg.vector_norm(result[1]).item() == pytest.approx(1.0)


def test_height_penalty_uses_environment_origin_and_both_range_bounds():
    fn = _functions('locomotion_rewards.py')['base_height_range']
    origins = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 10]], dtype=torch.float64)
    positions = origins.clone()
    positions[:, 2] += torch.tensor([.3, .31, .35, .36, .33], dtype=torch.float64)
    scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=positions)))
    scene.env_origins = origins
    result = fn(SimpleNamespace(scene=scene), .31, .35)
    torch.testing.assert_close(result, torch.tensor([.0001, 0, 0, .0001, 0], dtype=torch.float64))


def test_lying_joint_penalty_uses_target_and_selected_joints():
    fn = _functions('rewards.py')['lying_joint_deviation_l2']
    actual = torch.tensor([[1., 99., 4.], [2., 99., 0.]])
    target = torch.tensor([[1., 0., 2.], [1., 0., 2.]])
    scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(joint_pos=actual)))
    env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(
        get_term=lambda name: SimpleNamespace(lying_joint=target)))
    torch.testing.assert_close(fn(env, Selector(joint_ids=[0, 2])), torch.tensor([4., 5.]))


@pytest.mark.parametrize('is_critic,width', [(False, 588), (True, 606)])
def test_symmetry_matches_six_frame_layout_and_is_an_involution(is_critic, width):
    namespace = _functions('symmetry.py')
    augment = namespace['data_augmentation_func_g1_crawl']
    offset = torch.linspace(-.4, .4, 29).reshape(1, 29)
    env = SimpleNamespace(env=SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(default_joint_pos=offset))}))
    observations = torch.linspace(-1., 1., 2 * width).reshape(2, width)
    actions = torch.linspace(-1., 1., 46).reshape(2, 23)
    augmented, augmented_actions = augment(observations, actions, env, is_critic)
    assert augmented.shape == (4, width)
    assert augmented_actions.shape == (4, 23)
    torch.testing.assert_close(augmented[:2], observations)
    torch.testing.assert_close(augmented_actions[:2], actions)
    mirrored_twice, actions_twice = augment(augmented[2:], augmented_actions[2:], env, is_critic)
    torch.testing.assert_close(mirrored_twice[2:], observations)
    torch.testing.assert_close(actions_twice[2:], actions)
    torch.testing.assert_close(observations, torch.linspace(-1., 1., 2 * width).reshape(2, width))

def test_first_stage_pose_penalty_preserves_yaw_gate_and_strict_threshold():
    functions = _functions('rewards.py')
    first_stage = functions['crawl_joint_deviation_l2']
    final_stage = functions['lying_joint_deviation_l2']
    actual = torch.tensor([[1., 99., 2.]]).repeat(4, 1)
    target = torch.zeros_like(actual)
    velocity = SimpleNamespace(
        vel_command_b=torch.tensor([[.099, -.099, .099], [.1, 0., 0.], [0., 0., .2], [.2, 0., .2]]),
        is_standing_env=torch.tensor([False, False, False, True]),
    )
    terms = {'base_velocity': velocity, 'climb_command': SimpleNamespace(lying_joint=target)}
    env = SimpleNamespace(
        scene=Scene(robot=SimpleNamespace(data=SimpleNamespace(joint_pos=actual))),
        command_manager=SimpleNamespace(get_term=terms.__getitem__),
    )
    selected = Selector(joint_ids=[0, 2])
    # A pure yaw command keeps the first-stage penalty weak even though the
    # phase observation is zero for low planar commands.
    torch.testing.assert_close(first_stage(env, selected), torch.tensor([50., 5., 5., 50.]))
    torch.testing.assert_close(final_stage(env, selected), torch.full((4,), 5.))
