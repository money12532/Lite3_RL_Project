"""Headless, isolated flat-ground ONNX evaluation; does not use UDP or change the GUI.

Matches the current C++ policy runner's observation order, phase, PD, and raw
actions. This is a policy/physics regression test, not a keyboard/UDP timing test.
Prints JSON to stdout; redirect it to a new result file for comparisons.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort


def evaluate(model, session, command, duration, warmup, sequence=None):
    data = mujoco.MjData(model)
    default = np.tile([0., -.65, 1.3], 4)
    scale = np.tile([.125, .25, .25], 4)
    data.qpos[:3] = [0, 0, .35]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = default
    mujoco.mj_forward(model, data)
    feet = [model.geom(name + "_FOOT_collision").id for name in ("FL", "FR", "HL", "HR")]
    geom_to_foot = {geom: foot for foot, geom in enumerate(feet)}
    last_action = np.zeros(12)
    target = default.copy()
    records, targets = [], []
    force = np.zeros(6)
    geom_velocity = np.zeros(6)
    start = None
    obs_dim = session.get_inputs()[0].shape[1]
    if obs_dim not in (45, 47):
        raise ValueError(f"Expected 45/47 observations, got {obs_dim}")
    warm_steps = round(warmup / model.opt.timestep)
    total_steps = round((warmup + duration) / model.opt.timestep)
    for step in range(total_steps):
        if step == warm_steps:
            start = data.qpos[:3].copy()
        if step % 20 == 0:
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
            rotation = rotation.reshape(3, 3)
            if step < warm_steps:
                cmd = np.zeros(3)
            elif sequence is not None:
                elapsed = (step - warm_steps) * model.opt.timestep
                segment = min(int(elapsed / (duration / len(sequence))), len(sequence) - 1)
                cmd = np.asarray(sequence[segment], dtype=float)
            else:
                cmd = command
            obs = np.concatenate((data.qvel[3:6] * .25, rotation.T @ [0., 0., -1.], cmd,
                                  data.qpos[7:] - default, data.qvel[6:] * .05, last_action))
            if obs_dim == 47:
                phase = 2 * np.pi * step * model.opt.timestep / .60
                obs = np.r_[obs, np.sin(phase), np.cos(phase)]
            action = session.run(None, {session.get_inputs()[0].name: obs.astype(np.float32)[None]})[0].reshape(12)
            if not np.isfinite(action).all():
                raise RuntimeError("Nonfinite ONNX actions")
            last_action = action.copy()  # raw action, matching current C++ runner
            target = default + scale * action
            if step >= warm_steps:
                targets.append(target.copy())
                contact = np.zeros(4, dtype=bool)
                for index in range(data.ncon):
                    c = data.contact[index]
                    mujoco.mj_contactForce(model, data, index, force)
                    if force[0] > 2.0:
                        for geom in (c.geom1, c.geom2):
                            if geom in geom_to_foot:
                                contact[geom_to_foot[geom]] = True
                speed = []
                for geom in feet:
                    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom, geom_velocity, 0)
                    speed.append(np.linalg.norm(geom_velocity[3:5]))
                yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
                tilt = np.arccos(np.clip(rotation[2, 2], -1, 1))
                records.append((rotation.T @ data.qvel[:3], data.qvel[5], yaw, data.qpos[2], tilt,
                                data.geom_xpos[feet, 2].copy() - model.geom_size[feet, 0],
                                contact, np.asarray(speed), cmd.copy(), step * model.opt.timestep))
        data.ctrl[:] = 30 * (target - data.qpos[7:]) - data.qvel[6:]
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError("Nonfinite MuJoCo state")
    velocity = np.stack([row[0] for row in records])
    yaw_rate = np.array([row[1] for row in records])
    clearance = np.stack([row[5] for row in records])
    contact = np.stack([row[6] for row in records])
    slip = np.stack([row[7] for row in records])
    targets = np.stack(targets)
    commands = np.stack([row[8] for row in records])
    times = np.array([row[9] for row in records])
    phase = (times[:, None] / .60 + [0, .5, .5, 0]) % 1
    desired_contact = phase < .55
    moving = (np.linalg.norm(commands[:, :2], axis=1) > .05) | (np.abs(commands[:, 2]) > .05)
    result = {
        "command_m_s_rad_s": command.tolist() if sequence is None else None,
        "mean_body_velocity_m_s": velocity[:, :2].mean(0).tolist(),
        "mean_yaw_rate_rad_s": float(yaw_rate.mean()),
        "xy_tracking_rmse_m_s": float(np.sqrt(np.mean(np.sum((velocity[:, :2] - commands[:, :2])**2, axis=1)))),
        "yaw_tracking_rmse_rad_s": float(np.sqrt(np.mean((yaw_rate - commands[:, 2])**2))),
        "displacement_world_m": (data.qpos[:3] - start).tolist(),
        "yaw_change_rad": float(np.diff(np.unwrap([row[2] for row in records])).sum()),
        "min_base_height_m": float(min(row[3] for row in records)),
        "mean_base_height_m": float(np.mean([row[3] for row in records])),
        "max_tilt_rad": float(max(row[4] for row in records)),
        "foot_clearance_p95_m_FL_FR_HL_HR": np.percentile(clearance, 95, axis=0).tolist(),
        "foot_vertical_range_m_FL_FR_HL_HR": np.ptp(clearance, axis=0).tolist(),
        "toe_contact_fraction_FL_FR_HL_HR": contact.mean(0).tolist(),
        "stance_slip_mean_m_s": float((slip * contact).sum() / max(contact.sum(), 1)),
        "target_second_difference_rms_rad": float(np.sqrt(np.mean(np.diff(targets, n=2, axis=0)**2))),
        "max_target_offset_rad": float(np.abs(targets - default).max()),
        "diagonal_contact_match_fraction": float((contact[moving] == desired_contact[moving]).mean()) if moving.any() else None,
        "diagonal_pair_sync_fraction": float(np.stack((contact[moving, 0] == contact[moving, 3],
                                                       contact[moving, 1] == contact[moving, 2]), axis=1).mean()) if moving.any() else None,
        "exclusive_diagonal_support_fraction": float(((contact[moving] == [1, 0, 0, 1]).all(1)
                                                        | (contact[moving] == [0, 1, 1, 0]).all(1)).mean()) if moving.any() else None,
    }
    if sequence is not None:
        result["segments"] = []
        for index, cmd in enumerate(sequence):
            # Summarize the final second of each segment, after its transition.
            end = warmup + (index + 1) * duration / len(sequence)
            mask = (times >= end - 1) & (times < end)
            result["segments"].append({"command": list(cmd),
                "final_second_velocity_xy": velocity[mask, :2].mean(0).tolist(),
                "final_second_yaw_rate": float(yaw_rate[mask].mean()),
                "final_second_mean_foot_clearance_m": float(clearance[mask].mean()),
                "final_second_foot_vertical_range_m": np.ptp(clearance[mask], axis=0).tolist()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-root", type=Path, default=Path(__file__).resolve().parents[3] / "Lite3_rl_deploy")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=6.)
    parser.add_argument("--warmup", type=float, default=2.)
    parser.add_argument("--levels", type=float, nargs="+", default=[.25, .5, 1.])
    parser.add_argument("--sequence", action="store_true", help="Run W/S/A/D/Q/E/stop continuously, 4 seconds each.")
    parser.add_argument("--stop-segments", type=int, default=1, help="Number of final four-second stop segments in sequence mode.")
    args = parser.parse_args()
    if args.duration < .1 or args.warmup < 0 or any(not 0 < level <= 1 for level in args.levels):
        parser.error("duration >= 0.1, warmup >= 0, and levels in (0, 1] required")
    if args.stop_segments < 1:
        parser.error("stop-segments must be >= 1")
    # Load only robot + ground, not Lite3_stair.xml: stairs can confound flat tests.
    xml = args.deploy_root / "third_party/deep_robotics_model/Lite3/Lite3_mjcf/mjcf/Lite3.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    model.opt.timestep = .001
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.policy), sess_options=options, providers=["CPUExecutionProvider"])
    directions = {"W": [1, 0, 0], "S": [-1, 0, 0], "A": [0, .4, 0],
                  "D": [0, -.4, 0], "Q": [0, 0, .6], "E": [0, 0, -.6]}
    if args.sequence:
        commands = [[.5, 0, 0], [-.5, 0, 0], [0, .2, 0], [0, -.2, 0], [0, 0, .3], [0, 0, -.3], [0, 0, 0]]
        commands += [[0, 0, 0]] * (args.stop_segments - 1)
        result = {"policy": str(args.policy.resolve()), "sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                  "scene": str(xml), "segment_duration_s": 4, "warmup_s": args.warmup,
                  "continuous_sequence": evaluate(model, session, np.zeros(3), 4. * len(commands), args.warmup, sequence=commands)}
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    result = {"policy": str(args.policy.resolve()), "sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
              "scene": str(xml), "duration_s": args.duration, "warmup_s": args.warmup,
              "limitations": "Deterministic flat reset; no UDP, GUI, latency randomization or robustness certification.",
              "tests": {"stop": evaluate(model, session, np.zeros(3), args.duration, args.warmup)}}
    for level in args.levels:
        for key, velocity in directions.items():
            result["tests"][f"{key}_{level:g}"] = evaluate(model, session, np.array(velocity) * level, args.duration, args.warmup)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
