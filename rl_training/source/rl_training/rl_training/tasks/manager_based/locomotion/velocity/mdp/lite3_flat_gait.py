"""Lite3 flat-ground gait targets and rewards (FL, FR, HL, HR order).

Tensor helpers intentionally have no Isaac Sim dependency, so the gait math can
be regression-tested on the deployment computer. These rewards are flat-only:
foot clearance is measured above the environment's ground plane.
"""

from __future__ import annotations

import torch


def moving_mask(command, threshold=0.05):
    return (command[:, :2].norm(dim=-1) > threshold) | (command[:, 2].abs() > threshold)


def difficult_command(command):
    """Backward, sideways and yaw motion need extra footwork supervision."""
    return (command[:, 0] < -0.05) | (command[:, 1].abs() > 0.05) | (command[:, 2].abs() > 0.05)


def sample_flat_commands(count, device, ranges, proportions, minimum_speeds):
    """Sample stop/forward/backward/lateral/yaw/mixed, with no tiny pure commands.

    Proportions specify the first five modes; the remainder is mixed motion.
    Bounds are current curriculum bounds, not a bypass to the full-speed range.
    """
    if len(proportions) != 5 or min(proportions) < 0 or sum(proportions) > 1.0 + 1e-6:
        raise ValueError("Expected five nonnegative mode proportions summing to <= 1.")
    bounds = torch.tensor(ranges, device=device)
    if bounds.shape != (3, 2) or not bool(((bounds[:, 0] < 0) & (bounds[:, 1] > 0)).all()):
        raise ValueError("Lite3 flat commands require three ranges spanning zero.")
    command = bounds[:, 0] + torch.rand(count, 3, device=device) * (bounds[:, 1] - bounds[:, 0])
    slot = torch.rand(count, device=device)
    edges = torch.tensor(proportions, device=device).cumsum(0)
    mode = torch.bucketize(slot, edges)
    command[mode == 0] = 0
    for mode_id, axis, fixed_sign in ((1, 0, 1), (2, 0, -1), (3, 1, 0), (4, 2, 0)):
        # Generate full batches to avoid device synchronization for mask sizes.
        sign = torch.where(torch.rand(count, device=device) < 0.5, -1.0, 1.0)
        if fixed_sign:
            sign.fill_(fixed_sign)
        maximum = torch.where(sign > 0, bounds[axis, 1], -bounds[axis, 0])
        minimum = torch.minimum(torch.full_like(maximum, minimum_speeds[axis]), maximum)
        speed = sign * (minimum + torch.rand(count, device=device) * (maximum - minimum))
        mask = mode == mode_id
        command[mask] = 0
        command[mask, axis] = speed[mask]
    # Mixed samples use the same deadband as the gait rewards.
    command[~moving_mask(command)] = 0
    return command


def trot_reference(time, command, nominal_xy, cycle_time=0.60, stance_fraction=0.55,
                   max_half_stride=0.18, forward_clearance=0.06, other_clearance=0.08):
    """Return XY position/velocity, vertical lift, and desired contact per foot.

    XY swing is cubic Hermite with the *same* velocity as stance at both ends.
    Vertical lift has zero velocity and acceleration at lift-off/touchdown.
    """
    if cycle_time <= 0 or not 0 < stance_fraction < 1:
        raise ValueError("cycle_time must be positive and stance_fraction must be in (0, 1).")
    nominal = torch.as_tensor(nominal_xy, device=command.device, dtype=command.dtype)
    if nominal.shape != (4, 2):
        raise ValueError("nominal_xy must contain FL, FR, HL, HR coordinates.")
    offsets = command.new_tensor([0.0, 0.5, 0.5, 0.0])
    phase = torch.remainder(time[:, None] / cycle_time + offsets, 1.0)
    contact = phase < stance_fraction
    twist = torch.stack((command[:, 0, None] - command[:, 2, None] * nominal[None, :, 1],
                         command[:, 1, None] + command[:, 2, None] * nominal[None, :, 0]), dim=-1)
    half = twist * (cycle_time * stance_fraction / 2)
    half *= (max_half_stride / half.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1)
    stance_xy = half * (1 - 2 * phase / stance_fraction)[..., None]
    stance_vel = -2 * half / (cycle_time * stance_fraction)
    t = ((phase - stance_fraction) / (1 - stance_fraction)).clamp(0, 1)
    slope = -2 * (1 - stance_fraction) / stance_fraction
    smooth = 3 * t.square() - 2 * t.pow(3)
    swing_xy = half * (-1 + slope * t + (2 - slope) * smooth)[..., None]
    swing_vel = half * (slope + (2 - slope) * 6 * t * (1 - t))[..., None] / (cycle_time * (1 - stance_fraction))
    height = torch.where(difficult_command(command), other_clearance, forward_clearance)
    lift = height[:, None] * 64 * t.pow(3) * (1 - t).pow(3)
    xy = nominal[None] + torch.where(contact[..., None], stance_xy, swing_xy)
    velocity = torch.where(contact[..., None], stance_vel, swing_vel)
    return xy, velocity, torch.where(contact, 0, lift), contact


def rotate_inverse(quaternion, vector):
    """Apply inverse unit quaternion (wxyz), with matching leading dimensions."""
    xyz = quaternion[..., 1:]
    uv = torch.cross(xyz, vector, dim=-1)
    return vector - 2 * quaternion[..., :1] * uv + 2 * torch.cross(xyz, uv, dim=-1)


def body_foot_state(position_w, velocity_w, root_position, root_velocity, root_quat, root_omega_b):
    quaternion = root_quat[:, None, :].expand(-1, position_w.shape[1], -1)
    position = rotate_inverse(quaternion, position_w - root_position[:, None])
    velocity = rotate_inverse(quaternion, velocity_w - root_velocity[:, None])
    # d(R^T (p_foot - p_base))/dt includes the rotating-frame term.
    velocity -= torch.cross(root_omega_b[:, None].expand_as(position), position, dim=-1)
    return position, velocity


def action_second_difference(action, previous, previous_previous, episode_step):
    error = (action - 2 * previous + previous_previous).square().sum(-1)
    # A zero-valued action is valid history, not a reset indicator.
    return error * (episode_step >= 3)


def smooth_actions(env):
    action = env.action_manager.action
    previous = env.action_manager.prev_action
    old = getattr(env, "_lite3_flat_previous_previous_action", None)
    valid = old is not None
    if not valid:
        old = torch.zeros_like(action)
    reward = action_second_difference(action, previous, old, env.episode_length_buf)
    env._lite3_flat_previous_previous_action = previous.detach().clone()
    return reward if valid else torch.zeros_like(reward)


def _targets(env, command_name, nominal_xy, cycle_time, stance_fraction, max_half_stride,
             forward_clearance, other_clearance):
    command = env.command_manager.get_command(command_name)[:, :3]
    reference = trot_reference(env.episode_length_buf * env.step_dt, command, nominal_xy,
                               cycle_time=cycle_time, stance_fraction=stance_fraction,
                               max_half_stride=max_half_stride, forward_clearance=forward_clearance,
                               other_clearance=other_clearance)
    return command, reference


def horizontal_foot_tracking(env, command_name, asset_cfg, nominal_xy,
                             cycle_time=0.60, stance_fraction=0.55, position_std=0.05,
                             velocity_std=0.6, max_half_stride=0.18,
                             forward_clearance=0.06, other_clearance=0.08):
    command, (xy, velocity, _, _) = _targets(env, command_name, nominal_xy, cycle_time, stance_fraction,
                                           max_half_stride, forward_clearance, other_clearance)
    data = env.scene[asset_cfg.name].data
    position, actual_velocity = body_foot_state(
        data.body_pos_w[:, asset_cfg.body_ids], data.body_lin_vel_w[:, asset_cfg.body_ids],
        data.root_pos_w, data.root_lin_vel_w, data.root_quat_w, data.root_ang_vel_b)
    error = ((position[..., :2] - xy) / position_std).square().sum(-1)
    error += 0.1 * ((actual_velocity[..., :2] - velocity) / velocity_std).square().sum(-1)
    return torch.exp(-error).mean(-1) * moving_mask(command)


def foot_clearance_error(env, command_name, asset_cfg, nominal_xy, cycle_time=0.60,
                         stance_fraction=0.55, foot_radius=0.022, height_std=0.035,
                         max_half_stride=0.18, forward_clearance=0.06, other_clearance=0.08):
    """Penalize height error in swing AND stance; torso bobbing cannot lift all feet for free."""
    command, (_, _, lift, _) = _targets(env, command_name, nominal_xy, cycle_time, stance_fraction,
                                       max_half_stride, forward_clearance, other_clearance)
    actual = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    target = env.scene.env_origins[:, 2, None] + foot_radius + lift
    error = ((actual - target) / height_std).square().clamp(max=4).mean(-1)
    return error * moving_mask(command) * (1 + 0.5 * difficult_command(command))


def diagonal_contact_error(env, command_name, sensor_cfg, nominal_xy, cycle_time=0.60,
                           stance_fraction=0.55, force_threshold=1.0, max_half_stride=0.18,
                           forward_clearance=0.06, other_clearance=0.08):
    command, (_, _, _, desired) = _targets(env, command_name, nominal_xy, cycle_time, stance_fraction,
                                          max_half_stride, forward_clearance, other_clearance)
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    contact = forces.norm(dim=-1) > force_threshold
    # Signed penalty: all feet grounded must not earn a positive gait reward.
    return (contact != desired).float().mean(-1) * moving_mask(command)


def stance_slip(env, command_name, sensor_cfg, asset_cfg, force_threshold=1.0):
    sensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > force_threshold
    # Current contact, not stale history that penalizes a just-lifted swing foot.
    speed = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    command = env.command_manager.get_command(command_name)
    return (speed * contact).sum(-1) * (1 + 0.5 * difficult_command(command))


def standing_contact(env, command_name, sensor_cfg, force_threshold=1.0):
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    contact = forces.norm(dim=-1) > force_threshold
    return contact.float().mean(-1) * ~moving_mask(env.command_manager.get_command(command_name))
