"""Independent stair-climbing task based on the validated Lite3 move policy."""

from __future__ import annotations

import copy

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg
from rl_training.tasks.manager_based.locomotion.velocity.mdp import lite3_stair_gait

from .flat_env_cfg import DeeproboticsLite3FlatEnvCfg


@configclass
class DeeproboticsLite3StairRewardsCfg(RewardsCfg):
    """楼梯任务额外使用的奖励项。"""

    stable_landing = RewTerm(
        func=lite3_stair_gait.stair_stable_landing_reward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_FOOT"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_FOOT"),
        },
    )


@configclass
class DeeproboticsLite3StairEnvCfg(DeeproboticsLite3FlatEnvCfg):
    """Curriculum stair training that preserves the validated flat trot."""

    rewards: DeeproboticsLite3StairRewardsCfg = DeeproboticsLite3StairRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 以离散的四种踏面深度覆盖 25～40 cm。台阶高度继续由课程学习
        # 在 2～10 cm 之间连续变化，上楼、下楼各占 45%。
        stair_terrain = copy.deepcopy(ROUGH_TERRAINS_CFG)
        stair_terrain.curriculum = True
        for terrain_cfg in stair_terrain.sub_terrains.values():
            terrain_cfg.proportion = 0.0
        stair_terrain.sub_terrains["random_rough"].proportion = 0.10
        stair_terrain.sub_terrains["random_rough"].noise_range = (0.0, 0.02)
        for source_name in ("pyramid_stairs", "pyramid_stairs_inv"):
            source_cfg = stair_terrain.sub_terrains[source_name]
            for step_width in (0.25, 0.30, 0.35, 0.40):
                terrain_cfg = copy.deepcopy(source_cfg)
                terrain_cfg.proportion = 0.1125
                terrain_cfg.step_height_range = (0.02, 0.10)
                terrain_cfg.step_width = step_width
                terrain_cfg.platform_width = 2.0
                width_cm = int(round(step_width * 100.0))
                stair_terrain.sub_terrains[f"{source_name}_{width_cm}cm"] = terrain_cfg

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = stair_terrain
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

        # Keep the deployable 47-dimensional proprioceptive actor observation.
        # The critic retains privileged terrain scans during training.
        self.observations.policy.height_scan = None
        self.observations.policy.phase.params["cycle_time"] = 0.60
        self.observations.critic.phase.params["cycle_time"] = 0.60

        # Prioritize forward/backward stair traversal. Small lateral/yaw samples
        # preserve balance and correction ability without dominating training.
        command = self.commands.base_velocity
        command.ranges.lin_vel_x = (-0.45, 0.55)
        command.ranges.lin_vel_y = (-0.10, 0.10)
        command.ranges.ang_vel_z = (-0.25, 0.25)
        command.flat_mode_proportions = (0.10, 0.38, 0.22, 0.08, 0.08)
        command.flat_minimum_speeds = (0.12, 0.06, 0.10)

        # 随机改变机器人到最近台阶边缘的距离，使到达边缘时的步态相位不固定。
        # 同时限制横向偏移和朝向，避免把训练样本浪费在横着面对楼梯上。
        reset = self.events.randomize_reset_base.params
        reset["pose_range"]["x"] = (-0.45, 0.45)
        reset["pose_range"]["y"] = (-0.15, 0.15)
        reset["pose_range"]["roll"] = (-0.05, 0.05)
        reset["pose_range"]["pitch"] = (-0.05, 0.05)
        reset["pose_range"]["yaw"] = (-0.10, 0.10)
        reset["velocity_range"]["x"] = (-0.10, 0.10)
        reset["velocity_range"]["y"] = (-0.10, 0.10)
        reset["velocity_range"]["z"] = (-0.05, 0.05)

        # 摩擦在每个并行环境中随机取值；控制延迟仅修改楼梯任务所用的
        # 机器人配置副本，不改变已经验证过的平面移动配置。
        material = self.events.randomize_rigid_body_material.params
        material["static_friction_range"] = (0.45, 1.25)
        material["dynamic_friction_range"] = (0.35, 1.00)
        material["restitution_range"] = (0.0, 0.10)
        self.scene.robot = copy.deepcopy(self.scene.robot)
        for actuator_cfg in self.scene.robot.actuators.values():
            actuator_cfg.min_delay = 0
            actuator_cfg.max_delay = 3

        feet = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        foot_asset = SceneEntityCfg("robot", body_names=feet, preserve_order=True)
        foot_sensor = SceneEntityCfg("contact_forces", body_names=feet, preserve_order=True)

        # 只在摆动中段平滑达到 14 cm，离地和落地边界的目标仍为 0；
        # 因此不会持续强迫足端保持某个固定高度。
        self.rewards.feet_height.func = lite3_stair_gait.stair_trot_clearance_error
        self.rewards.feet_height.weight = -3.0
        self.rewards.feet_height.params = {
            "command_name": "base_velocity",
            "asset_cfg": foot_asset,
            "sensor_cfg": foot_sensor,
            "cycle_time": 0.60,
            "stance_fraction": 0.55,
            "force_threshold": 1.0,
            "peak_clearance": 0.14,
            "clearance_std": 0.04,
        }

        # Preserve the flat policy's diagonal trot and body-frame foot trajectory.
        # Requiring three contacts rewarded shuffling and contradicted that
        # warm-start, so it is deliberately removed.
        self.rewards.feet_contact.func = lite3_stair_gait.stair_trot_contact_error
        self.rewards.feet_contact.weight = -2.0
        self.rewards.feet_contact.params = {
            "command_name": "base_velocity",
            "sensor_cfg": foot_sensor,
            "cycle_time": 0.60,
            "stance_fraction": 0.55,
            "force_threshold": 1.0,
        }
        self.rewards.feet_gait.weight = 0.0

        # 水平力显著大于竖直力时，通常是脚尖撞到了台阶立面。
        # 使用连续惩罚替代原先只有 0/1 的 stumble 判定。
        self.rewards.feet_stumble.func = lite3_stair_gait.stair_riser_collision_error
        self.rewards.feet_stumble.weight = -4.0
        self.rewards.feet_stumble.params = {
            "command_name": "base_velocity",
            "sensor_cfg": foot_sensor,
            "horizontal_ratio": 2.0,
            "force_scale": 20.0,
        }

        # 奖励低水平速度、低向下速度的首次落脚，并对实际处于支撑相的
        # 足端世界坐标系滑动施加更强惩罚。
        self.rewards.stable_landing.weight = 2.0
        self.rewards.stable_landing.params = {
            "command_name": "base_velocity",
            "sensor_cfg": foot_sensor,
            "asset_cfg": foot_asset,
            "cycle_time": 0.60,
            "stance_fraction": 0.55,
            "horizontal_speed_std": 0.25,
            "downward_speed_std": 0.35,
        }
        self.rewards.feet_slide.func = lite3_stair_gait.stair_support_slip_error
        self.rewards.feet_slide.weight = -2.0
        self.rewards.feet_slide.params = {
            "command_name": "base_velocity",
            "sensor_cfg": foot_sensor,
            "asset_cfg": foot_asset,
            "cycle_time": 0.60,
            "stance_fraction": 0.55,
            "force_threshold": 1.0,
        }
        self.rewards.foot_impact_velocity.weight = -3.0
        self.rewards.foot_impact_velocity.params["sensor_cfg"] = foot_sensor
        self.rewards.foot_impact_velocity.params["asset_cfg"] = foot_asset
        self.rewards.foot_impact_velocity.params["speed_threshold"] = 0.05

        # Climbing necessarily creates vertical body velocity and modest pitch;
        # the flat-task penalties would otherwise discourage the required motion.
        self.rewards.lin_vel_z_l2.weight = -3.0
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.track_lin_vel_xy_exp.weight = 7.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0

        # Retain smooth control but allow faster leg repositioning near an edge.
        self.rewards.action_rate_l2.weight = -0.08
        self.rewards.action_smooth_l2.weight = -0.02

        # bad_orientation_2 already ends an episode near 44 degrees of roll or
        # pitch. This adds a strong one-time cost to that terminal transition.
        self.rewards.is_terminated.weight = -100.0

        if self.__class__.__name__ == "DeeproboticsLite3StairEnvCfg":
            self.disable_zero_weight_rewards()
