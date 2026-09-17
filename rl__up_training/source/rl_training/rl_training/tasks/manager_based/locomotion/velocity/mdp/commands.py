# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """从带阈值的均匀分布中生成 SE(2) 速度命令的命令生成器。

    支持固定比例的特殊样本，这些样本始终使用原始完整速度范围，防止课程学习
    缩小命令范围时灾难性遗忘基本运动能力。
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg
    """命令生成器的配置。"""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # 用于 TensorBoard 的附加指标。
        self.metrics["base_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["knee_pos"] = torch.zeros(self.num_envs, device=self.device)
        self._metric_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        knee_joint_ids = self.robot.find_joints(".*[Kk]nee.*")[0]
        self._knee_joint_ids = torch.tensor(knee_joint_ids, dtype=torch.long, device=self.device)

        # ---- 保存原始完整范围，供特殊采样使用 ----
        # 特殊环境始终从完整范围采样，不受课程学习范围变化影响。
        self._full_ranges_lin_vel_x = tuple(cfg.ranges.lin_vel_x)
        self._full_ranges_lin_vel_y = tuple(cfg.ranges.lin_vel_y)
        self._full_ranges_ang_vel_z = tuple(cfg.ranges.ang_vel_z)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)

        extras = {}
        for metric_name, metric_value in self.metrics.items():
            if metric_name in {"base_z", "knee_pos"}:
                step_count = torch.clamp(self._metric_step_counter[env_ids].float(), min=1.0)
                extras[metric_name] = torch.mean(metric_value[env_ids] / step_count).item()
            else:
                extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            metric_value[env_ids] = 0.0

        self._metric_step_counter[env_ids] = 0
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras

    def _update_metrics(self):
        super()._update_metrics()

        # 1）base_z 指标：root_pos_w[:, 2]
        base_z = self.robot.data.root_pos_w[:, 2]

        # 2）knee_pos 指标：采用与膝关节 joint_pos_penalty 相同的计算形式
        cmd = torch.linalg.norm(self.vel_command_b, dim=1)
        body_vel = torch.linalg.norm(self.robot.data.root_lin_vel_b[:, :2], dim=1)

        if self._knee_joint_ids.numel() > 0:
            running_reward = torch.linalg.norm(
                self.robot.data.joint_pos[:, self._knee_joint_ids]
                - self.robot.data.default_joint_pos[:, self._knee_joint_ids],
                dim=1,
            )
        else:
            running_reward = torch.zeros(self.num_envs, device=self.device)

        knee_pos = torch.where(
            torch.logical_or(cmd > 0.1, body_vel > 0.5),
            running_reward,
            5.0 * running_reward,
        )

        self.metrics["base_z"] += base_z
        self.metrics["knee_pos"] += knee_pos
        self._metric_step_counter += 1

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # 将较小的命令置零
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # ---- 固定比例的特殊样本 ----
        # 对固定比例环境覆盖原始采样命令，并始终使用完整速度范围，
        # 从而不受课程学习缩小范围的影响，防止灾难性遗忘基本能力。
        n = len(env_ids)
        if n == 0:
            return

        r = torch.empty(n, device=self.device)
        # 为每个环境分配 [0, 1) 内的区间，布局如下：
        #   [0,                          rel_zero_vel          ) -> 零速度
        #   [rel_zero_vel,              +rel_only_lin_y        ) -> 仅 lin_y
        #   [rel_zero_vel+rel_only_lin_y, +rel_only_lin_x      ) -> 仅 lin_x
        #   [...,                        +rel_only_ang_z        ) -> 仅 ang_z
        #   [...,                        1.0                    ) -> 常规采样（不覆盖）
        slot = r.uniform_(0.0, 1.0)
        cum = 0.0

        # --- 零速度（站立）---
        rel_zero_vel = self.cfg.rel_zero_vel_envs
        mask_zero = slot < cum + rel_zero_vel
        cum += rel_zero_vel

        # --- 仅 y 方向线速度 ---
        rel_only_lin_y = self.cfg.rel_only_lin_y_envs
        mask_only_y = (slot >= cum) & (slot < cum + rel_only_lin_y)
        cum += rel_only_lin_y

        # --- 仅 x 方向线速度 ---
        rel_only_lin_x = self.cfg.rel_only_lin_x_envs
        mask_only_x = (slot >= cum) & (slot < cum + rel_only_lin_x)
        cum += rel_only_lin_x

        # --- 仅 z 轴角速度 ---
        rel_only_ang_z = self.cfg.rel_only_ang_z_envs
        mask_only_ang = (slot >= cum) & (slot < cum + rel_only_ang_z)
        cum += rel_only_ang_z

        ids_zero = env_ids[mask_zero]
        ids_only_y = env_ids[mask_only_y]
        ids_only_x = env_ids[mask_only_x]
        ids_only_ang = env_ids[mask_only_ang]

        # 应用零速度命令
        if len(ids_zero) > 0:
            self.vel_command_b[ids_zero, :] = 0.0
            self.is_standing_env[ids_zero] = True
            self.is_heading_env[ids_zero] = False

        # 应用纯 lin_y：vx=0，vy 从完整范围采样，wz=0
        if len(ids_only_y) > 0:
            r_y = torch.empty(len(ids_only_y), device=self.device)
            self.vel_command_b[ids_only_y, 0] = 0.0
            self.vel_command_b[ids_only_y, 1] = r_y.uniform_(*self._full_ranges_lin_vel_y)
            self.vel_command_b[ids_only_y, 2] = 0.0
            self.is_standing_env[ids_only_y] = False
            self.is_heading_env[ids_only_y] = False

        # 应用纯 lin_x：vx 从完整范围采样，vy=0，wz=0
        if len(ids_only_x) > 0:
            r_x = torch.empty(len(ids_only_x), device=self.device)
            self.vel_command_b[ids_only_x, 0] = r_x.uniform_(*self._full_ranges_lin_vel_x)
            self.vel_command_b[ids_only_x, 1] = 0.0
            self.vel_command_b[ids_only_x, 2] = 0.0
            self.is_standing_env[ids_only_x] = False
            self.is_heading_env[ids_only_x] = False

        # 应用纯 ang_z：vx=0，vy=0，偏航命令从完整范围采样。
        if len(ids_only_ang) > 0:
            r_h = torch.empty(len(ids_only_ang), device=self.device)
            self.vel_command_b[ids_only_ang, 0] = 0.0
            self.vel_command_b[ids_only_ang, 1] = 0.0
            self.is_standing_env[ids_only_ang] = False
            if self.cfg.heading_command:
                self.is_heading_env[ids_only_ang] = True
                self.heading_target[ids_only_ang] = r_h.uniform_(*self.cfg.ranges.heading)
            else:
                # Q/E 部署接口直接发送角速度。这里显式地重新采样，
                # 因为父类采样器可能在覆盖前已将该环境设为站立样本。
                self.is_heading_env[ids_only_ang] = False
                self.vel_command_b[ids_only_ang, 2] = r_h.uniform_(*self._full_ranges_ang_vel_z)


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """带阈值均匀速度命令生成器的配置。"""

    class_type: type = UniformThresholdVelocityCommand

    # 仅当 class_type 为 Lite3FlatVelocityCommand 时使用。
    # 其他机器人仍使用原采样器及其现有默认值。
    flat_mode_proportions: tuple[float, ...] = (0.08, 0.18, 0.27, 0.25, 0.20)
    """停止、前进、后退、横移、偏航；剩余 2% 为混合运动。"""
    flat_minimum_speeds: tuple[float, ...] = (0.15, 0.10, 0.15)
    """纯方向命令的最小绝对值：x/y 单位为 m/s，偏航单位为 rad/s。"""

    rel_zero_vel_envs: float = 0.07
    """始终接收零速度命令的环境比例。

    这些样本用于防止遗忘静止站立能力，不受命令课程学习影响。
    """

    rel_only_lin_y_envs: float = 0.07
    """仅接收横向（y）速度命令的环境比例。

    vx 和 wz 强制为零，vy 从原始完整范围采样，不受命令课程学习影响。
    """

    rel_only_lin_x_envs: float = 0.07
    """仅接收前进/后退（x）速度命令的环境比例。

    vy 和 wz 强制为零，vx 从原始完整范围采样，不受命令课程学习影响。
    """

    rel_only_ang_z_envs: float = 0.07
    """仅接收角速度（航向）命令的环境比例。

    vx 和 vy 强制为零，航向从完整范围采样，不受命令课程学习影响。
    """


class Lite3FlatVelocityCommand(UniformThresholdVelocityCommand):
    """平衡各种平面运动能力，并覆盖低速键盘命令。"""

    def _resample_command(self, env_ids: Sequence[int]):
        from .lite3_flat_gait import moving_mask, sample_flat_commands

        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        command = sample_flat_commands(
            len(env_ids), self.device,
            (self.cfg.ranges.lin_vel_x, self.cfg.ranges.lin_vel_y, self.cfg.ranges.ang_vel_z),
            self.cfg.flat_mode_proportions, self.cfg.flat_minimum_speeds,
        )
        self.vel_command_b[env_ids] = command
        self.is_standing_env[env_ids] = ~moving_mask(command)
        self.is_heading_env[env_ids] = False


class DiscreteCommandController(CommandTerm):
    """
    为环境分配离散命令的命令生成器。

    命令以预定义整数列表的形式存储。
    控制器通过索引映射命令（例如索引 0 -> 10，索引 1 -> 20）。
    """

    cfg: DiscreteCommandControllerCfg
    """命令控制器的配置。"""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        初始化命令控制器。

        参数：
            cfg：命令控制器配置。
            env：环境对象。
        """
        # 初始化父类
        super().__init__(cfg, env)

        # 检查 available_commands 是否非空
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # 确保所有元素均为整数
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # 保存可用命令
        self.available_commands = self.cfg.available_commands

        # 创建用于保存命令的缓冲区
        # -- 命令缓冲区：保存每个环境的离散动作索引
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands：保存当前命令的快照（整数形式）
        self.current_commands = [self.available_commands[0]] * self.num_envs  # 默认使用第一个命令

    def __str__(self) -> str:
        """返回命令控制器的字符串表示。"""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    属性
    """

    @property
    def command(self) -> torch.Tensor:
        """返回当前命令缓冲区，形状为 (num_envs, 1)。"""
        return self.command_buffer

    """
    实现相关函数。
    """

    def _update_metrics(self):
        """更新命令控制器的指标。"""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """为指定环境重新采样命令。"""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """更新并保存当前命令。"""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """离散命令控制器的配置。"""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    可用离散命令列表，其中每个元素均为整数。
    示例：[10, 20, 30, 40, 50]
    """
