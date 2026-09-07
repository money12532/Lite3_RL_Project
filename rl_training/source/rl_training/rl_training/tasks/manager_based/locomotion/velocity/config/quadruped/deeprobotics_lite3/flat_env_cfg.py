# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.tasks.manager_based.locomotion.velocity.mdp import lite3_flat_gait

from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg


@configclass
class DeeproboticsLite3FlatEnvCfg(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        # self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        # self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Flat footwork v2: keep the existing 47 observations, 12 actions,
        # action scales, PD gains and 0.60 s clock compatible with deployment.
        feet = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        # FK at HipX=0, HipY=-0.65, Knee=1.3 in the official Lite3 model.
        # Do not cache a random first episode as the standing reference.
        nominal_xy = ((0.18062449, 0.15935), (0.18062449, -0.15935),
                      (-0.16837551, 0.15935), (-0.16837551, -0.15935))
        trajectory = {"command_name": "base_velocity", "nominal_xy": nominal_xy,
                      "cycle_time": 0.60, "stance_fraction": 0.55,
                      "max_half_stride": 0.18, "forward_clearance": 0.06, "other_clearance": 0.08}

        def foot_asset():
            return SceneEntityCfg("robot", body_names=feet, preserve_order=True)

        def foot_sensor():
            return SceneEntityCfg("contact_forces", body_names=feet, preserve_order=True)

        self.commands.base_velocity.class_type = mdp.Lite3FlatVelocityCommand
        self.commands.base_velocity.rel_standing_envs = 0.0  # sampler supplies exactly 8% stops
        self.commands.base_velocity.flat_mode_proportions = (0.08, 0.18, 0.27, 0.25, 0.20)
        self.commands.base_velocity.flat_minimum_speeds = (0.15, 0.10, 0.15)

        self.rewards.base_height_l2.params["target_height"] = 0.35
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.20
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25

        # Suppress rapid reversals without suppressing the entire swing.
        self.rewards.action_rate_l2.weight = -0.08
        self.rewards.action_smooth_l2.func = lite3_flat_gait.smooth_actions
        self.rewards.action_smooth_l2.weight = -0.02
        self.rewards.knee_joint_pos_penalty.weight = -0.35
        self.rewards.hipx_joint_pos_penalty.weight = -0.20
        self.rewards.knee_joint_pos_penalty.params["command_threshold"] = 0.05
        self.rewards.hipx_joint_pos_penalty.params["command_threshold"] = 0.05
        self.rewards.stand_still.params["command_threshold"] = 0.05
        # Equal diagonal joint angles are not a valid target during lateral/yaw motion.
        self.rewards.joint_mirror.weight = 0.0

        self.rewards.phase_foot_trajectory_exp.func = lite3_flat_gait.horizontal_foot_tracking
        self.rewards.phase_foot_trajectory_exp.weight = 3.0
        self.rewards.phase_foot_trajectory_exp.params = {
            **trajectory, "asset_cfg": foot_asset(), "position_std": 0.05, "velocity_std": 0.6}
        # Dense world-ground clearance term: lifting the body or shuffling on the
        # floor cannot substitute for alternating 6 cm / 8 cm swing clearance.
        self.rewards.feet_height.func = lite3_flat_gait.foot_clearance_error
        self.rewards.feet_height.weight = -2.0
        self.rewards.feet_height.params = {
            **trajectory, "asset_cfg": foot_asset(), "foot_radius": 0.022, "height_std": 0.035}
        self.rewards.feet_contact.func = lite3_flat_gait.diagonal_contact_error
        self.rewards.feet_contact.weight = -3.0
        self.rewards.feet_contact.params = {**trajectory, "sensor_cfg": foot_sensor()}
        # Avoid overlapping positive contact-time rewards that also pay out for
        # continuous four-foot contact. The phase contact target is now explicit.
        self.rewards.feet_gait.weight = 0.0
        self.rewards.feet_slide.func = lite3_flat_gait.stance_slip
        self.rewards.feet_slide.weight = -1.5
        self.rewards.feet_slide.params = {
            "command_name": "base_velocity", "asset_cfg": foot_asset(), "sensor_cfg": foot_sensor()}
        self.rewards.feet_contact_without_cmd.func = lite3_flat_gait.standing_contact
        self.rewards.feet_contact_without_cmd.weight = 0.4
        self.rewards.feet_contact_without_cmd.params = {
            "command_name": "base_velocity", "sensor_cfg": foot_sensor()}
        self.rewards.feet_air_time_lin_xy.weight = 1.0
        self.rewards.feet_air_time_lin_xy.params["cmd_threshold"] = 0.05
        self.rewards.feet_air_time_ang_z_lite3.weight = 1.0
        self.rewards.feet_air_time_ang_z_lite3.params["cmd_threshold"] = 0.05

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "DeeproboticsLite3FlatEnvCfg":
            self.disable_zero_weight_rewards()
