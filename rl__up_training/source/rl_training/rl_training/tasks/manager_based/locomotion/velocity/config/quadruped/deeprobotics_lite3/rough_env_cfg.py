# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
)
# from isaaclab.sensors.ray_caster import GridPatternCfg
##
# 预定义配置
##
from rl_training.assets.deeprobotics import DEEPROBOTICS_LITE3_CFG  # isort: skip


@configclass
class DeeproboticsLite3ObservationsCfg(ObservationsCfg):
    """加入显式的 0.60 秒步态时钟，以生成确定性的对角小跑步态。"""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        phase = ObsTerm(func=mdp.phase, params={"cycle_time": 0.60}, clip=(-1.0, 1.0), scale=1.0)

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        phase = ObsTerm(func=mdp.phase, params={"cycle_time": 0.60}, clip=(-1.0, 1.0), scale=1.0)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class DeeproboticsLite3RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: DeeproboticsLite3ObservationsCfg = DeeproboticsLite3ObservationsCfg()
    base_link_name = "TORSO"
    foot_link_name = ".*_FOOT"
    # fmt: off
    joint_names = [
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
        "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
        "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
        "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
    ]

    link_names = [
       'TORSO', 
       'FL_HIP', 'FR_HIP', 'HL_HIP', 'HR_HIP', 
       'FL_THIGH', 'FR_THIGH', 'HL_THIGH', 'HR_THIGH', 
       'FL_SHANK', 'FR_SHANK', 'HL_SHANK', 'HR_SHANK', 
       'FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT',
    ]

    hipx_joint_names = [
        "FL_HipX_joint", "FR_HipX_joint", "HL_HipX_joint", "HR_HipX_joint",
    ]

    hipy_joint_names = [
        "FL_HipY_joint", "FR_HipY_joint", "HL_HipY_joint", "HR_HipY_joint",
    ]

    knee_joint_names = [
        "FL_Knee_joint", "FR_Knee_joint", "HL_Knee_joint", "HR_Knee_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # 执行父类的初始化后处理
        super().__post_init__()

        # ------------------------------场景------------------------------
        self.scene.robot = DEEPROBOTICS_LITE3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner.pattern_cfg.resolution = 0.07 #  = GridPatternCfg(resolution=0.07, size=[1.6, 1.0]),

        # ------------------------------观测------------------------------
        self.observations.policy.base_lin_vel = None # type: ignore
        self.observations.policy.height_scan = None # type: ignore
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------动作------------------------------
        # 减小动作缩放系数
        self.actions.joint_pos.scale = {".*_HipX_joint": 0.125, "^(?!.*_HipX_joint).*": 0.25}
        # 将策略输出限制在具有物理意义的范围内。旧的 +/-100 边界会使 Actor
        # 均值发散，而仿真器仅静默地对关节进行饱和，最终生成不可用的 ONNX 模型。
        self.actions.joint_pos.clip = {".*": (-5.0, 5.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------事件------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.0, 0.0),
            },
        }


        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = self.link_names # [self.base_link_name]
        # self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_com_positions.params["asset_cfg"].body_names = self.base_link_name # [self.base_link_name]
        # self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.joint_names

        # 将方块和楼梯地形的生成概率设为 0
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.4
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.3
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.3
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
        # 由于机器人尺寸较小，相应缩小地形尺度
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_width = 0.8
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------奖励------------------------------
        self.rewards.action_rate_l2.weight = -0.1 #-0.02
        # self.rewards.smoothness_2.weight = -0.0075

        self.rewards.base_height_l2.weight = -50.0
        self.rewards.base_height_l2.params["target_height"] = 0.55
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.feet_air_time_lin_xy.weight = 2.0
        self.rewards.feet_air_time_lin_xy.params["threshold"] = 0.20
        self.rewards.feet_air_time_lin_xy.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_x_neg.weight = 0.0 # 5.0
        self.rewards.feet_air_time_x_neg.params["threshold"] = 0.5
        self.rewards.feet_air_time_x_neg.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_ang_z_lite3.weight = 2.0
        self.rewards.feet_air_time_ang_z_lite3.params["threshold"] = 0.20
        self.rewards.feet_air_time_ang_z_lite3.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -0.0 # -8.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.75
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.foot_impact_velocity.weight = -2.0 # -10.0
        self.rewards.foot_impact_velocity.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.foot_impact_velocity.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.stand_still.weight = -0.5 # -1.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.stand_still.params["command_threshold"] = 0.1
        self.rewards.feet_height_body.weight = -0.0 # -2.5
        self.rewards.feet_height_body.params["target_height"] = -0.35
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = -0.0 # -0.2
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.contact_forces.weight = -1e-1 # -2e-2
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.lin_vel_z_l2.weight = -20.0 #-2.0
        self.rewards.ang_vel_xy_l2.weight = -0.25 # -0.05

        # 让命令跟踪奖励压过站立/平滑导致的局部最优。
        # 这对后退、横移和转向能力尤其重要。
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0

        self.rewards.undesired_contacts.weight = -0.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]

        self.rewards.joint_torques_l2.weight = -2.5e-4
        self.rewards.joint_acc_l2.weight = -1e-8
        self.rewards.joint_deviation_l1.weight = -0.0
        self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = [".*HipX.*"]
        self.rewards.joint_power.weight = -8e-4
        self.rewards.flat_orientation_l2.weight = -20.0

        # 添加以下奖励以改善步态
        self.rewards.feet_gait.weight = 3.0
        self.rewards.feet_gait.params["std"] = 0.05
        self.rewards.feet_gait.params["max_err"] = 0.30
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_FOOT", "HR_FOOT"],
            ["FR_FOOT", "HL_FOOT"]
        ]

        # 显式的对角接触目标可以阻止原步态奖励所允许的四脚同步运动
        # 和四脚全部滑动等投机行为。
        self.rewards.feet_contact.func = mdp.trot_contact_pattern
        self.rewards.feet_contact.weight = 4.0
        self.rewards.feet_contact.params = {
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            "cycle_time": 0.60,
            "phase_offsets": (0.0, 1.0, 1.0, 0.0),
            "stance_span": 1.10,
            "command_threshold": 0.1,
            "contact_force_threshold": 1.0,
        }

        self.rewards.phase_foot_trajectory_exp.weight = 3.0
        self.rewards.phase_foot_trajectory_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.phase_foot_trajectory_exp.params.update(
            {
                "cycle_time": 0.60,
                "phase_offsets": (0.0, 1.0, 1.0, 0.0),
                "gait_span": 0.10,
                "gait_psi": 0.08,
                "gait_delta": 0.0,
                "x_offset": 0.0,
                "stance_span": 1.10,
                "stand_ref_z_offset": 0.0,
                "velocity_weight": 0.02,
                "std": 0.20,
            }
        )
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FL_(HipX|HipY|Knee).*", "HR_(HipX|HipY|Knee).*"],
            ["FR_(HipX|HipY|Knee).*", "HL_(HipX|HipY|Knee).*"],
        ]

        self.rewards.joint_pos_limits.weight = -5.0
        # self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]

        # 新增奖励
        self.rewards.hipx_joint_pos_penalty.weight = -0.4
        self.rewards.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipx_joint_names
        self.rewards.hipy_joint_pos_penalty.weight = -0.0
        self.rewards.hipy_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipy_joint_names
        self.rewards.knee_joint_pos_penalty.weight = -2
        self.rewards.knee_joint_pos_penalty.params["asset_cfg"].joint_names = self.knee_joint_names


        # 将权重为 0 的奖励项设为 None
        if self.__class__.__name__ == "DeeproboticsLite3RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------终止条件------------------------------
        self.terminations.illegal_contact = None
        # self.terminations.bad_orientation_2 = None

        # ------------------------------课程学习------------------------------
        # 从接近一次键盘增量的命令开始，速度跟踪收敛后再扩展到完整范围。
        self.curriculum.command_levels.params["range_multiplier"] = (0.40, 1.0)

        # ------------------------------命令------------------------------
        command = self.commands.base_velocity
        command.ranges.lin_vel_x = (-1.0, 1.0)
        command.ranges.lin_vel_y = (-0.4, 0.4)
        command.ranges.ang_vel_z = (-0.6, 0.6)

        # 部署端通过 Q/E 直接提供偏航角速度。训练时使用相同接口，
        # 不把目标航向角转换成偏航角速度。
        command.heading_command = False
        command.rel_heading_envs = 0.0

        # 保留足够的静止样本以实现稳定停止，同时显式平衡三个可控速度轴。
        # lin_vel_x 采用对称采样，因此前进和后退具有相同概率。
        command.rel_standing_envs = 0.10
        command.rel_zero_vel_envs = 0.05
        # 横向命令需要足够多的专用样本，才能学会真正的交叉步，
        # 而不是利用低摩擦让足端滑动。
        command.rel_only_lin_x_envs = 0.30
        command.rel_only_lin_y_envs = 0.20
        command.rel_only_ang_z_envs = 0.25
