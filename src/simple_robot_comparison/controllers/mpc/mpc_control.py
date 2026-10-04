"""do-mpc path tracking with a nominal rolling model and wheel-speed feedback."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from simple_robot_comparison.control_input import ControlInput
from simple_robot_comparison.controllers.speed_reference import (
    DEFAULT_MAX_BRAKING_ACCELERATION,
    DEFAULT_MAX_LATERAL_ACCELERATION,
    DEFAULT_MAX_SPEED,
    WHEEL_RADIUS,
    WHEEL_SEPARATION,
    build_speed_profile,
)
from simple_robot_comparison.observation import Observation

if TYPE_CHECKING:
    import do_mpc

# Nominal geometry and joint limits from models/car.xml (SI units).
AXLE_OFFSET = 0.07  # The tracked body origin is ahead of the wheel axle.
WHEEL_DAMPING = 0.03
MAX_WHEEL_TORQUE = 0.5
MAX_WHEEL_SPEED = MAX_WHEEL_TORQUE / WHEEL_DAMPING
WHEEL_SPEED_KP = 0.02


@dataclass(frozen=True)
class MPCConfig:
    """Prediction timing, nominal motion limits, and tracking cost weights."""

    horizon: int = 15
    time_step: float = 0.1
    speed: float = DEFAULT_MAX_SPEED
    max_lateral_acceleration: float = DEFAULT_MAX_LATERAL_ACCELERATION
    max_braking_acceleration: float = DEFAULT_MAX_BRAKING_ACCELERATION
    max_yaw_rate: float = 4.0
    wheel_acceleration: float = 15.0
    position_weight: float = 30.0
    heading_weight: float = 1.0
    speed_weight: float = 5.0
    input_weight: float = 0.1

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("MPC horizon must be a positive integer")
        for field in fields(self):
            value = getattr(self, field.name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"MPC {field.name} must be finite and positive")
        if self.speed > WHEEL_RADIUS * MAX_WHEEL_SPEED:
            raise ValueError(
                "MPC speed must not exceed the nominal wheel limit (0.5 m/s)"
            )


class MPCController:
    """Optimize a speed/turning ramp, then track it with wheel motor efforts.

    The five prediction states are body x/y/yaw and *commanded* speed/yaw rate.
    The model assumes rolling without slip and ideal wheel-speed tracking;
    MuJoCo still determines the actual motion. Feedback uses the observation's
    ideal pose and encoders, never the ground's hidden friction map.

    Solve every ``time_step`` seconds; ramp wheel references and apply feedback
    every simulation step. Create a fresh instance when the episode resets.
    """

    def __init__(self, config: MPCConfig | None = None) -> None:
        self.config = config or MPCConfig()
        self.speed_reference = 0.0
        self.wheel_reference = np.zeros(2)
        self._target_wheels = np.zeros(2)
        self._steps_remaining = 0
        self._dt: float | None = None
        self._yaw: float | None = None
        self._initialized = False
        self._mpc = self._build_mpc()

    def _build_mpc(self) -> do_mpc.controller.MPC:
        # Keep solver/plotting imports out of manual and classic startup.
        import casadi as ca
        import do_mpc

        config = self.config
        model = do_mpc.model.Model("discrete")
        pose = model.set_variable("_x", "pose", (3, 1))
        motion = model.set_variable("_x", "motion", (2, 1))
        command = model.set_variable("_u", "command", (2, 1))
        target = model.set_variable("_tvp", "target", (4, 1))

        # Midpoint integration of a linear speed/yaw-rate ramp over one MPC step.
        speed = (motion[0] + command[0]) / 2
        yaw_rate = (motion[1] + command[1]) / 2
        heading = pose[2] + config.time_step * yaw_rate / 2
        velocity = ca.vertcat(
            speed * ca.cos(heading) - AXLE_OFFSET * yaw_rate * ca.sin(heading),
            speed * ca.sin(heading) + AXLE_OFFSET * yaw_rate * ca.cos(heading),
            yaw_rate,
        )
        model.set_rhs("pose", pose + config.time_step * velocity)
        model.set_rhs("motion", command)
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.settings.n_horizon = config.horizon
        mpc.settings.t_step = config.time_step
        mpc.settings.store_full_solution = False
        mpc.settings.store_lagr_multiplier = False
        mpc.settings.nlpsol_opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 100,
            "ipopt.tol": 1e-5,
            "print_time": False,
        }
        # A periodic heading cost has no discontinuity at +/-pi.
        cost = config.position_weight * ca.sumsqr(pose[:2] - target[:2])
        cost += config.heading_weight * 2 * (1 - ca.cos(pose[2] - target[2]))
        cost += config.speed_weight * (motion[0] - target[3]) ** 2
        mpc.set_objective(lterm=cost, mterm=cost)
        mpc.set_rterm(command=config.input_weight)
        mpc.bounds["lower", "_u", "command"] = [0.0, -config.max_yaw_rate]
        mpc.bounds["upper", "_u", "command"] = [config.speed, config.max_yaw_rate]
        for name, sign in (("left", -1), ("right", 1)):
            wheel = (
                command[0] + sign * WHEEL_SEPARATION * command[1] / 2
            ) / WHEEL_RADIUS
            previous = (
                motion[0] + sign * WHEEL_SEPARATION * motion[1] / 2
            ) / WHEEL_RADIUS
            mpc.set_nl_cons(f"{name}_speed", wheel**2, ub=MAX_WHEEL_SPEED**2)
            mpc.set_nl_cons(
                f"{name}_acceleration",
                (wheel - previous) ** 2,
                ub=(config.wheel_acceleration * config.time_step) ** 2,
            )
        self._tvp = mpc.get_tvp_template()
        mpc.set_tvp_fun(lambda _time: self._tvp)
        mpc.setup()
        return mpc

    def __call__(
        self, observation: Observation, reference: np.ndarray, dt: float
    ) -> ControlInput:
        """Replan when due, and return bounded forward/turn motor efforts."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Controller timestep must be finite and positive")
        steps = self.config.time_step / dt
        if steps < 1 or not np.isclose(steps, round(steps), rtol=0, atol=1e-9):
            raise ValueError(
                "MPC time_step must be an integer multiple of simulation dt"
            )
        if self._dt is not None and not np.isclose(dt, self._dt, rtol=0, atol=1e-12):
            raise ValueError(
                "Simulation timestep changed; create a fresh MPC controller"
            )
        self._dt = dt
        measured_wheels = np.array(
            [observation.left_wheel_speed, observation.right_wheel_speed]
        )
        pose = np.array([observation.x, observation.y, observation.yaw])
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(measured_wheels)):
            raise ValueError("MPC requires a finite pose and wheel speeds")
        if self._yaw is None:
            self._yaw = observation.yaw
        else:
            difference = observation.yaw - self._yaw
            self._yaw += float(np.arctan2(np.sin(difference), np.cos(difference)))
        pose[2] = self._yaw

        if self._steps_remaining == 0:
            targets = reference_horizon(pose[:2], reference, self.config)
            self.speed_reference = float(targets[0, 3])
            for step, target in enumerate(targets):
                self._tvp["_tvp", step, "target"] = target
            left, right = self.wheel_reference
            motion = [
                WHEEL_RADIUS * (left + right) / 2,
                WHEEL_RADIUS * (right - left) / WHEEL_SEPARATION,
            ]
            state = np.r_[pose, motion].reshape(-1, 1)
            if not self._initialized:
                self._mpc.x0 = state
                self._mpc.u0 = np.array(motion).reshape(-1, 1)
                self._mpc.set_initial_guess()
                self._initialized = True
            try:
                command = self._mpc.make_step(state).ravel()
            except RuntimeError as error:
                raise RuntimeError(
                    "MPC solver failed to compute a motor plan"
                ) from error
            if not np.all(np.isfinite(command)):
                raise RuntimeError("MPC solve failed: non-finite result")
            if not self._mpc.solver_stats["success"]:
                status = self._mpc.solver_stats.get("return_status", "unknown status")
                raise RuntimeError(f"MPC solve failed: {status}")
            speed, yaw_rate = command
            self._target_wheels = (
                np.array(
                    [
                        speed - WHEEL_SEPARATION * yaw_rate / 2,
                        speed + WHEEL_SEPARATION * yaw_rate / 2,
                    ]
                )
                / WHEEL_RADIUS
            )
            self._steps_remaining = round(steps)

        # Match the ramp used by the prediction model, including between solves.
        self.wheel_reference += (
            self._target_wheels - self.wheel_reference
        ) / self._steps_remaining
        self._steps_remaining -= 1
        torque = WHEEL_DAMPING * self.wheel_reference + WHEEL_SPEED_KP * (
            self.wheel_reference - measured_wheels
        )
        left, right = np.clip(torque, -MAX_WHEEL_TORQUE, MAX_WHEEL_TORQUE)
        return ControlInput(forward=float(left + right), turn=float(right - left))


def reference_horizon(
    position: np.ndarray, reference: np.ndarray, config: MPCConfig
) -> np.ndarray:
    """Sample x/y/heading/speed, advancing along the shared variable-speed profile.

    Integrate ds/dt = v_ref(s) with midpoint steps, so bends and their braking
    zones occupy more of the time horizon than equally long straight sections.
    Closed paths wrap; open paths hold their last point with zero future speed.
    """
    profile = build_speed_profile(
        reference,
        config.speed,
        config.max_lateral_acceleration,
        config.max_braking_acceleration,
    )
    distance = profile.project(position)
    samples = np.empty(config.horizon + 1)
    speeds = np.empty(config.horizon + 1)
    for step in range(config.horizon + 1):
        samples[step] = distance
        speeds[step] = profile.speed_at(distance)
        midpoint = distance + 0.5 * config.time_step * speeds[step]
        distance += config.time_step * profile.speed_at(midpoint)
    if profile.closed:
        samples %= profile.distance[-1]
    else:
        # A clamped waypoint has stopped moving; do not ask MPC to drive past it.
        speeds[samples >= profile.distance[-1]] = 0.0
        samples = np.minimum(samples, profile.distance[-1])
    segments = np.diff(profile.points, axis=0)
    indices = np.clip(
        np.searchsorted(profile.distance, samples, side="right") - 1,
        0,
        len(segments) - 1,
    )
    headings = np.arctan2(segments[indices, 1], segments[indices, 0])
    return np.column_stack(
        [
            np.interp(samples, profile.distance, profile.points[:, axis])
            for axis in range(2)
        ]
        + [headings, speeds]
    )
