"""CPU tests: python -m unittest discover -s tests -p 'test_lite3_flat_gait.py'."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

SOURCE = Path(__file__).resolve().parents[1] / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/lite3_flat_gait.py"
SPEC = importlib.util.spec_from_file_location("lite3_flat_gait", SOURCE)
gait = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gait)
XY = ((0.18062449, 0.15935), (0.18062449, -0.15935),
      (-0.16837551, 0.15935), (-0.16837551, -0.15935))


class GaitTests(unittest.TestCase):
    def test_keyboard_deadband(self):
        command = torch.tensor([[0., 0., 0.], [0., .1, 0.], [0., 0., -.15], [-.25, 0., 0.]])
        self.assertEqual(gait.moving_mask(command).tolist(), [False, True, True, True])

    def test_command_balance_and_bounds(self):
        torch.manual_seed(2026)
        command = gait.sample_flat_commands(50000, "cpu", ((-1, 1), (-.4, .4), (-.6, .6)),
                                             (.08, .18, .27, .25, .20), (.15, .1, .15))
        zero = (command == 0).all(-1)
        xonly = (command[:, 1:] == 0).all(-1) & ~zero
        yonly = (command[:, [0, 2]] == 0).all(-1) & ~zero
        yawonly = (command[:, :2] == 0).all(-1) & ~zero
        masks = (zero, xonly & (command[:, 0] > 0), xonly & (command[:, 0] < 0), yonly, yawonly)
        for mask, expected in zip(masks, (.08, .18, .27, .25, .20)):
            self.assertAlmostEqual(mask.float().mean().item(), expected, delta=.008)
        for axis, mask, minimum, maximum in ((0, xonly, .15, 1), (1, yonly, .1, .4), (2, yawonly, .15, .6)):
            speed = command[mask, axis].abs()
            self.assertGreaterEqual(speed.min().item(), minimum - 1e-6)
            self.assertLessEqual(speed.max().item(), maximum + 1e-6)

    def test_sampler_respects_curriculum_and_empty_batch(self):
        ranges = ((-.12, .12), (-.04, .04), (-.06, .06))
        command = gait.sample_flat_commands(300, "cpu", ranges, (.08, .18, .27, .25, .20), (.15, .1, .15))
        self.assertTrue((command.abs() <= torch.tensor([.12, .04, .06]) + 1e-6).all())
        empty = gait.sample_flat_commands(0, "cpu", ranges, (.08, .18, .27, .25, .20), (.15, .1, .15))
        self.assertEqual(empty.shape, (0, 3))
        with self.assertRaises(ValueError):
            gait.sample_flat_commands(1, "cpu", ranges, (.5,) * 5, (.15, .1, .15))

    def test_stride_direction_and_yaw(self):
        command = torch.tensor([[.5, 0., 0.], [-.5, 0., 0.], [0., .2, 0.], [0., 0., .6]])
        xy, velocity, _, _ = gait.trot_reference(torch.zeros(4), command, XY)
        nominal = torch.tensor(XY)
        torch.testing.assert_close(xy[0] - nominal, -(xy[1] - nominal))
        self.assertGreater((xy[2, 0, 1] - nominal[0, 1]).item(), 0)
        self.assertLess((xy[3, 0, 0] - nominal[0, 0]).item(), 0)
        self.assertGreater((xy[3, 0, 1] - nominal[0, 1]).item(), 0)
        torch.testing.assert_close(velocity[0, 0], torch.tensor([-.5, 0.]))

    def test_diagonal_pair_and_clearance(self):
        command = torch.tensor([[.5, 0., 0.], [-.5, 0., 0.]])
        _, _, lift, contact = gait.trot_reference(torch.full((2,), .465), command, XY)
        self.assertEqual(contact[0].tolist(), [False, True, True, False])
        torch.testing.assert_close(lift[:, 0], torch.tensor([.06, .08]))
        torch.testing.assert_close(lift[:, 0], lift[:, 3])

    def test_trajectory_boundary_continuity(self):
        command = torch.tensor([[.5, .1, .3]], dtype=torch.float64).repeat(2, 1)
        for boundary in (.33, .60):
            eps = 1e-7
            time = torch.tensor([boundary - eps, boundary + eps], dtype=torch.float64)
            xy, velocity, lift, _ = gait.trot_reference(time, command, XY)
            torch.testing.assert_close(xy[0], xy[1], atol=2e-6, rtol=0)
            torch.testing.assert_close(velocity[0], velocity[1], atol=2e-5, rtol=0)
            self.assertLess(abs((lift[1, 0] - lift[0, 0]).item()) / (2 * eps), .001)

    def test_rotating_frame_velocity(self):
        position = torch.tensor([[[1., 0., 0.]]])
        velocity = torch.tensor([[[0., 2., 0.]]])
        zeros = torch.zeros(1, 3)
        _, actual = gait.body_foot_state(position, velocity, zeros, zeros,
                                         torch.tensor([[1., 0., 0., 0.]]), torch.tensor([[0., 0., 2.]]))
        torch.testing.assert_close(actual, torch.zeros_like(actual))
        q = torch.tensor([[2**-.5, 0., 0., 2**-.5]])
        rotated = gait.rotate_inverse(q, torch.tensor([[1., 0., 0.]]))
        torch.testing.assert_close(rotated, torch.tensor([[0., -1., 0.]]), atol=1e-6, rtol=0)

    def test_smoothness_zero_is_valid_and_reset_is_masked(self):
        result = gait.action_second_difference(torch.ones(3, 12), torch.zeros(3, 12),
                                               torch.ones(3, 12), torch.tensor([1, 2, 3]))
        torch.testing.assert_close(result, torch.tensor([0., 0., 48.]))

    def make_env(self):
        command = torch.tensor([[-.5, 0., 0.]])
        time = torch.tensor([.46])
        xy, velocity, lift, contact = gait.trot_reference(time, command, XY)
        root = torch.tensor([[0., 0., .35]])
        position = torch.cat((xy, (.022 + lift)[..., None]), -1)
        data = NS(body_pos_w=position, body_lin_vel_w=torch.zeros(1, 4, 3), root_pos_w=root,
                  root_lin_vel_w=torch.zeros(1, 3), root_quat_w=torch.tensor([[1., 0., 0., 0.]]),
                  root_ang_vel_b=torch.zeros(1, 3))
        forces = torch.zeros(1, 4, 3)
        forces[..., 2] = contact * 20
        sensor = NS(data=NS(net_forces_w=forces))
        class Scene(dict):
            pass
        scene = Scene(robot=NS(data=data))
        scene.env_origins = torch.zeros(1, 3)
        scene.sensors = {"contact_forces": sensor}
        env = NS(scene=scene, episode_length_buf=torch.tensor([23]), step_dt=.02,
                 command_manager=NS(get_command=lambda _: command))
        return env, NS(name="robot", body_ids=[0, 1, 2, 3]), NS(name="contact_forces", body_ids=[0, 1, 2, 3])

    def test_ground_shuffle_worse_than_swing_and_zero_command_gates(self):
        env, asset, sensor = self.make_env()
        params = dict(command_name="base_velocity", nominal_xy=XY)
        self.assertLess(gait.foot_clearance_error(env, asset_cfg=asset, **params).item(), 1e-10)
        self.assertEqual(gait.diagonal_contact_error(env, sensor_cfg=sensor, **params).item(), 0)
        env.scene["robot"].data.body_pos_w[..., 2] = .022
        env.scene.sensors["contact_forces"].data.net_forces_w[..., 2] = 20
        self.assertGreater(gait.foot_clearance_error(env, asset_cfg=asset, **params).item(), 1)
        self.assertEqual(gait.diagonal_contact_error(env, sensor_cfg=sensor, **params).item(), .5)
        env.command_manager.get_command = lambda _: torch.zeros(1, 3)
        self.assertEqual(gait.foot_clearance_error(env, asset_cfg=asset, **params).item(), 0)
        self.assertEqual(gait.standing_contact(env, "base_velocity", sensor).item(), 1)

    def test_slip_uses_world_velocity_and_current_contacts(self):
        env, asset, sensor = self.make_env()
        env.scene["robot"].data.root_lin_vel_w[:, 0] = .5
        self.assertEqual(gait.stance_slip(env, "base_velocity", sensor, asset).item(), 0)
        env.scene["robot"].data.body_lin_vel_w[..., 0] = .2
        self.assertGreater(gait.stance_slip(env, "base_velocity", sensor, asset).item(), 0)
        env.scene.sensors["contact_forces"].data.net_forces_w[:] = 0
        self.assertEqual(gait.stance_slip(env, "base_velocity", sensor, asset).item(), 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
