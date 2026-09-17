"""Stair-specific gait rewards for Lite3.

This module is intentionally independent from Isaac Sim at import time so the
clearance math can be unit-tested on a CPU-only machine.
"""

from __future__ import annotations

import torch


def moving_mask(command: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """Return environments whose planar or yaw command is non-zero."""
    return (command[:, :2].norm(dim=-1) > threshold) | (command[:, 2].abs() > threshold)


def desired_trot_contact(
    time: torch.Tensor,
    cycle_time: float = 0.60,
    stance_fraction: float = 0.55,
) -> torch.Tensor:
    """Return desired contacts in FL, FR, HL, HR order for diagonal trot."""
    if cycle_time <= 0 or not 0.0 < stance_fraction < 1.0:
        raise ValueError("cycle_time must be positive and stance_fraction must be in (0, 1).")
    offsets = time.new_tensor((0.0, 0.5, 0.5, 0.0))
    phase = torch.remainder(time[:, None] / cycle_time + offsets, 1.0)
    return phase < stance_fraction


def trot_swing_profile(
    time: torch.Tensor,
    cycle_time: float = 0.60,
    stance_fraction: float = 0.55,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return diagonal-trot contacts and a smooth unit swing-height profile.

    The lift profile and its first two derivatives are zero at lift-off and
    touch-down.  This avoids the discontinuous clearance target used by the
    first stair experiment, which encouraged abrupt leg motion.
    """
    contact = desired_trot_contact(time, cycle_time, stance_fraction)
    offsets = time.new_tensor((0.0, 0.5, 0.5, 0.0))
    phase = torch.remainder(time[:, None] / cycle_time + offsets, 1.0)
    progress = ((phase - stance_fraction) / (1.0 - stance_fraction)).clamp(0.0, 1.0)
    lift = 64.0 * progress.pow(3) * (1.0 - progress).pow(3)
    return contact, torch.where(contact, torch.zeros_like(lift), lift)


def terrain_clearance_error_from_state(
    foot_z: torch.Tensor,
    actual_contact: torch.Tensor,
    desired_contact: torch.Tensor,
    swing_profile: torch.Tensor,
    command: torch.Tensor,
    peak_clearance: float = 0.12,
    clearance_std: float = 0.04,
) -> torch.Tensor:
    """Penalize only swing-foot clearance shortfall above the local support.

    The lowest supporting foot is used as a conservative local tread estimate.
    Unlike a world-height target, this remains meaningful while the four feet
    occupy different stair levels.  Overshoot is not penalized here because a
    higher tread can physically hold the swing foot above the reference; body
    motion and energy penalties already prevent jumping as a shortcut.
    """
    if foot_z.ndim != 2 or foot_z.shape[1] != 4:
        raise ValueError("foot_z must have shape [num_envs, 4].")
    if swing_profile.shape != foot_z.shape or desired_contact.shape != foot_z.shape:
        raise ValueError("contact and swing profile tensors must match foot_z.")
    if peak_clearance <= 0.0 or clearance_std <= 0.0:
        raise ValueError("clearance parameters must be positive.")

    support = actual_contact & desired_contact
    any_support = support.any(dim=1)
    support_z = foot_z.masked_fill(~support, torch.inf).min(dim=1).values
    any_contact = actual_contact.any(dim=1)
    contact_z = foot_z.masked_fill(~actual_contact, torch.inf).min(dim=1).values
    fallback_z = torch.where(any_contact, contact_z, foot_z.min(dim=1).values)
    support_z = torch.where(any_support, support_z, fallback_z)

    target = peak_clearance * swing_profile
    clearance = foot_z - support_z[:, None]
    shortfall = torch.relu(target - clearance)
    swing = ~desired_contact
    error = torch.square(shortfall / clearance_std).clamp(max=4.0)
    error = (error * swing).sum(dim=1) / swing.sum(dim=1).clamp_min(1)
    return error * moving_mask(command)


def contact_pattern_error_from_state(
    actual_contact: torch.Tensor,
    desired_contact: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    """Penalize contact-pattern mismatch while requiring four contacts at rest."""
    moving = moving_mask(command)
    desired = torch.where(moving[:, None], desired_contact, torch.ones_like(desired_contact))
    return (actual_contact != desired).float().mean(dim=1)


def riser_collision_error_from_state(
    contact_forces: torch.Tensor,
    command: torch.Tensor,
    horizontal_ratio: float = 2.0,
    force_scale: float = 20.0,
) -> torch.Tensor:
    """连续惩罚足尖撞击台阶立面产生的水平接触力。"""
    if contact_forces.ndim != 3 or contact_forces.shape[1:] != (4, 3):
        raise ValueError("contact_forces must have shape [num_envs, 4, 3].")
    if horizontal_ratio <= 0.0 or force_scale <= 0.0:
        raise ValueError("collision parameters must be positive.")

    horizontal_force = torch.linalg.norm(contact_forces[..., :2], dim=-1)
    vertical_force = contact_forces[..., 2].abs()
    excess = torch.relu(horizontal_force - horizontal_ratio * vertical_force)
    error = torch.square(excess / force_scale).clamp(max=4.0).mean(dim=1)
    return error * moving_mask(command)


def landing_stability_reward_from_state(
    first_contact: torch.Tensor,
    desired_contact: torch.Tensor,
    foot_velocity: torch.Tensor,
    command: torch.Tensor,
    horizontal_speed_std: float = 0.25,
    downward_speed_std: float = 0.35,
) -> torch.Tensor:
    """奖励在目标支撑相内发生的低水平速度、低冲击落脚。"""
    if first_contact.shape != desired_contact.shape or first_contact.ndim != 2:
        raise ValueError("contact tensors must have matching [num_envs, 4] shapes.")
    if foot_velocity.shape != (*first_contact.shape, 3):
        raise ValueError("foot_velocity must have shape [num_envs, 4, 3].")
    if horizontal_speed_std <= 0.0 or downward_speed_std <= 0.0:
        raise ValueError("landing speed scales must be positive.")

    horizontal_speed = torch.linalg.norm(foot_velocity[..., :2], dim=-1)
    downward_speed = torch.relu(-foot_velocity[..., 2])
    quality = torch.exp(
        -torch.square(horizontal_speed / horizontal_speed_std)
        -torch.square(downward_speed / downward_speed_std)
    )
    valid_landing = first_contact & desired_contact
    reward = (quality * valid_landing).sum(dim=1) / valid_landing.sum(dim=1).clamp_min(1)
    return reward * moving_mask(command)


def support_slip_error_from_state(
    actual_contact: torch.Tensor,
    desired_contact: torch.Tensor,
    foot_velocity_xy: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    """只惩罚目标支撑脚在世界坐标系中的水平滑动。"""
    if actual_contact.shape != desired_contact.shape or actual_contact.ndim != 2:
        raise ValueError("contact tensors must have matching [num_envs, 4] shapes.")
    if foot_velocity_xy.shape != (*actual_contact.shape, 2):
        raise ValueError("foot_velocity_xy must have shape [num_envs, 4, 2].")

    support = actual_contact & desired_contact
    speed = torch.linalg.norm(foot_velocity_xy, dim=-1)
    error = (speed * support).sum(dim=1) / support.sum(dim=1).clamp_min(1)
    return error * moving_mask(command)


def stair_trot_clearance_error(
    env,
    command_name: str,
    sensor_cfg,
    asset_cfg,
    cycle_time: float = 0.60,
    stance_fraction: float = 0.55,
    force_threshold: float = 1.0,
    peak_clearance: float = 0.12,
    clearance_std: float = 0.04,
) -> torch.Tensor:
    """Isaac Lab wrapper for smooth, stair-relative trot clearance."""
    command = env.command_manager.get_command(command_name)[:, :3]
    time = env.episode_length_buf.float() * env.step_dt
    desired_contact, swing_profile = trot_swing_profile(time, cycle_time, stance_fraction)
    foot_z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    actual_contact = forces.norm(dim=-1) > force_threshold
    return terrain_clearance_error_from_state(
        foot_z,
        actual_contact,
        desired_contact,
        swing_profile,
        command,
        peak_clearance,
        clearance_std,
    )


def stair_trot_contact_error(
    env,
    command_name: str,
    sensor_cfg,
    cycle_time: float = 0.60,
    stance_fraction: float = 0.55,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize deviation from the deployable policy's diagonal trot."""
    command = env.command_manager.get_command(command_name)[:, :3]
    time = env.episode_length_buf.float() * env.step_dt
    desired_contact = desired_trot_contact(time, cycle_time, stance_fraction)
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    actual_contact = forces.norm(dim=-1) > force_threshold
    return contact_pattern_error_from_state(actual_contact, desired_contact, command)


def stair_riser_collision_error(
    env,
    command_name: str,
    sensor_cfg,
    horizontal_ratio: float = 2.0,
    force_scale: float = 20.0,
) -> torch.Tensor:
    """Isaac Lab 包装：惩罚足尖撞击台阶立面。"""
    command = env.command_manager.get_command(command_name)[:, :3]
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    return riser_collision_error_from_state(forces, command, horizontal_ratio, force_scale)


def stair_stable_landing_reward(
    env,
    command_name: str,
    sensor_cfg,
    asset_cfg,
    cycle_time: float = 0.60,
    stance_fraction: float = 0.55,
    horizontal_speed_std: float = 0.25,
    downward_speed_std: float = 0.35,
) -> torch.Tensor:
    """Isaac Lab 包装：奖励目标支撑相内的稳定落脚。"""
    command = env.command_manager.get_command(command_name)[:, :3]
    time = env.episode_length_buf.float() * env.step_dt
    desired_contact = desired_trot_contact(time, cycle_time, stance_fraction)
    sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    foot_velocity = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids]
    return landing_stability_reward_from_state(
        first_contact,
        desired_contact,
        foot_velocity,
        command,
        horizontal_speed_std,
        downward_speed_std,
    )


def stair_support_slip_error(
    env,
    command_name: str,
    sensor_cfg,
    asset_cfg,
    cycle_time: float = 0.60,
    stance_fraction: float = 0.55,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Isaac Lab 包装：惩罚实际接触且处于目标支撑相的足端滑动。"""
    command = env.command_manager.get_command(command_name)[:, :3]
    time = env.episode_length_buf.float() * env.step_dt
    desired_contact = desired_trot_contact(time, cycle_time, stance_fraction)
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    actual_contact = forces.norm(dim=-1) > force_threshold
    foot_velocity_xy = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return support_slip_error_from_state(actual_contact, desired_contact, foot_velocity_xy, command)
