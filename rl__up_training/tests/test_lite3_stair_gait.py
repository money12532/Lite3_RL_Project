"""CPU regression tests for the stair-clearance reward."""

import importlib.util
from pathlib import Path
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[1] / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/lite3_stair_gait.py"
SPEC = importlib.util.spec_from_file_location("lite3_stair_gait", SOURCE)
gait = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gait)


class StairGaitTests(unittest.TestCase):
    def test_trot_profile_is_diagonal_and_smooth(self):
        contact, lift = gait.trot_swing_profile(torch.tensor([0.0, 0.15, 0.36]))
        self.assertEqual(contact[0].tolist(), [True, True, True, True])
        self.assertEqual(contact[1].tolist(), [True, False, False, True])
        self.assertEqual(contact[2].tolist(), [False, True, True, False])
        self.assertTrue((lift[contact] == 0.0).all())
        self.assertGreater(lift[1, 1].item(), 0.0)

    def test_clearance_target_is_zero_at_swing_boundaries(self):
        stance_fraction = 0.55
        times = torch.tensor([
            stance_fraction * 0.60,
            (stance_fraction + (1.0 - stance_fraction) * 0.5) * 0.60,
            0.60 - 1.0e-6,
        ])
        _, lift = gait.trot_swing_profile(times)
        self.assertLess(lift[0, 0].item(), 1.0e-6)
        self.assertAlmostEqual(lift[1, 0].item(), 1.0, places=5)
        self.assertLess(lift[2, 0].item(), 1.0e-6)

    def test_terrain_clearance_penalizes_only_shortfall(self):
        desired = torch.tensor([[False, True, True, False]])
        actual = torch.tensor([[False, True, True, False]])
        profile = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        command = torch.tensor([[0.3, 0.0, 0.0]])
        low = gait.terrain_clearance_error_from_state(
            torch.tensor([[0.16, 0.10, 0.10, 0.16]]), actual, desired, profile, command
        )
        clear = gait.terrain_clearance_error_from_state(
            torch.tensor([[0.23, 0.10, 0.10, 0.23]]), actual, desired, profile, command
        )
        self.assertGreater(low.item(), 1.0)
        self.assertEqual(clear.item(), 0.0)

    def test_terrain_clearance_is_gated_when_standing(self):
        desired = torch.tensor([[False, True, True, False]])
        actual = torch.tensor([[False, True, True, False]])
        profile = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        value = gait.terrain_clearance_error_from_state(
            torch.tensor([[0.10, 0.10, 0.10, 0.10]]), actual, desired, profile, torch.zeros(1, 3)
        )
        self.assertEqual(value.item(), 0.0)

    def test_contact_pattern_error_and_standing_target(self):
        desired = torch.tensor([[False, True, True, False]])
        command = torch.tensor([[0.3, 0.0, 0.0]])
        correct = gait.contact_pattern_error_from_state(desired, desired, command)
        missing = gait.contact_pattern_error_from_state(
            torch.tensor([[False, True, False, False]]), desired, command
        )
        stopped = gait.contact_pattern_error_from_state(
            torch.ones(1, 4, dtype=torch.bool), desired, torch.zeros(1, 3)
        )
        self.assertEqual(correct.item(), 0.0)
        self.assertEqual(missing.item(), 0.25)
        self.assertEqual(stopped.item(), 0.0)

    def test_riser_collision_detects_horizontal_impact(self):
        command = torch.tensor([[0.3, 0.0, 0.0]])
        tread_force = torch.tensor([[[5.0, 0.0, 20.0]] * 4])
        riser_force = tread_force.clone()
        riser_force[0, 0] = torch.tensor([40.0, 0.0, 2.0])
        tread = gait.riser_collision_error_from_state(tread_force, command)
        riser = gait.riser_collision_error_from_state(riser_force, command)
        self.assertEqual(tread.item(), 0.0)
        self.assertGreater(riser.item(), 0.0)

    def test_stable_landing_prefers_low_speed_contact(self):
        first_contact = torch.tensor([[True, False, False, False]])
        desired_contact = torch.ones(1, 4, dtype=torch.bool)
        command = torch.tensor([[0.3, 0.0, 0.0]])
        calm_velocity = torch.zeros(1, 4, 3)
        impact_velocity = calm_velocity.clone()
        impact_velocity[0, 0] = torch.tensor([1.0, 0.0, -1.0])
        calm = gait.landing_stability_reward_from_state(
            first_contact, desired_contact, calm_velocity, command
        )
        impact = gait.landing_stability_reward_from_state(
            first_contact, desired_contact, impact_velocity, command
        )
        self.assertAlmostEqual(calm.item(), 1.0, places=6)
        self.assertLess(impact.item(), calm.item())

    def test_support_slip_ignores_swing_foot_and_penalizes_support(self):
        actual_contact = torch.tensor([[True, True, False, False]])
        desired_contact = torch.tensor([[True, False, False, True]])
        command = torch.tensor([[0.3, 0.0, 0.0]])
        velocity = torch.zeros(1, 4, 2)
        velocity[0, 1, 0] = 1.0
        swing_collision = gait.support_slip_error_from_state(
            actual_contact, desired_contact, velocity, command
        )
        velocity[0, 0, 0] = 1.0
        support_slide = gait.support_slip_error_from_state(
            actual_contact, desired_contact, velocity, command
        )
        self.assertEqual(swing_collision.item(), 0.0)
        self.assertGreater(support_slide.item(), 0.0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
