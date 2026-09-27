"""Line-of-sight guidance, heading PD, and wheel-speed feedback."""

import numpy as np

from simple_robot_comparison.control_input import ControlInput
from simple_robot_comparison.observation import Observation

# Nominal dimensions and motor properties from models/car.xml (SI units).
WHEEL_RADIUS = 0.03
WHEEL_SEPARATION = 0.12
WHEEL_DAMPING = 0.03
MAX_WHEEL_TORQUE = 0.5

# Conservative reference limits for this simulation, not hardware ratings.
MAX_YAW_RATE = 10.0  # rad/s
MAX_WHEEL_SPEED = 12.0  # rad/s
MAX_WHEEL_ACCELERATION = 15.0  # rad/s^2


class ClassicController:
    """A callable controller holding the previous PD error and speed commands.

    Create a fresh instance for each episode. The interface stays
    ``controller(observation, reference, dt) -> ControlInput``.
    """

    def __init__(self) -> None:
        self.lookahead = 0.18  # metres along the path
        self.cruise_speed = 0.75  # m/s
        self.heading_kp = 4.0  # (rad/s) / rad
        self.heading_kd = 0.2
        self.derivative_filter_time = 0.1  # seconds
        self.wheel_speed_kp = 0.02  # N m / (rad/s)
        self.previous_error: float | None = None
        self.filtered_derivative = 0.0
        self.wheel_reference = np.zeros(2)  # left, right; rad/s

    def __call__(
        self, observation: Observation, reference: np.ndarray, dt: float
    ) -> ControlInput:
        """Convert a lookahead heading into achievable wheel motor efforts."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Controller timestep must be finite and positive")

        position = np.array([observation.x, observation.y])
        target = lookahead_target(position, reference, self.lookahead)
        offset = target - position
        desired_heading = np.arctan2(offset[1], offset[0])
        error = wrap_angle(desired_heading - observation.yaw)

        # No derivative kick on the first step. Wrapping the difference also
        # prevents a false spike when the error crosses +/-pi.
        derivative = 0.0
        if self.previous_error is not None:
            derivative = wrap_angle(error - self.previous_error) / dt
        alpha = dt / (self.derivative_filter_time + dt)
        self.filtered_derivative += alpha * (derivative - self.filtered_derivative)
        self.previous_error = error
        yaw_rate = np.clip(
            self.heading_kp * error + self.heading_kd * self.filtered_derivative,
            -MAX_YAW_RATE,
            MAX_YAW_RATE,
        )

        # Differential-drive kinematics: v_L = v - b*w/2, v_R = v + b*w/2.
        desired_wheels = (
            np.array(
                [
                    self.cruise_speed - WHEEL_SEPARATION * yaw_rate / 2,
                    self.cruise_speed + WHEEL_SEPARATION * yaw_rate / 2,
                ]
            )
            / WHEEL_RADIUS
        )
        # Scale both wheels together to preserve the requested curvature.
        desired_wheels /= max(1.0, np.max(np.abs(desired_wheels)) / MAX_WHEEL_SPEED)
        self.wheel_reference += np.clip(
            desired_wheels - self.wheel_reference,
            -MAX_WHEEL_ACCELERATION * dt,
            MAX_WHEEL_ACCELERATION * dt,
        )

        measured_wheels = np.array(
            [observation.left_wheel_speed, observation.right_wheel_speed]
        )
        # Compensate nominal viscous joint damping, then correct speed error.
        # This commands finite torques; contact physics still determines slip.
        torque = WHEEL_DAMPING * self.wheel_reference + self.wheel_speed_kp * (
            self.wheel_reference - measured_wheels
        )
        left, right = np.clip(torque, -MAX_WHEEL_TORQUE, MAX_WHEEL_TORQUE)

        # car.xml mixes commands as tau_L=(forward-turn)/2 and
        # tau_R=(forward+turn)/2. Invert that mapping for the two wheel torques.
        return ControlInput(forward=float(left + right), turn=float(right - left))


def wrap_angle(angle: float) -> float:
    """Return the equivalent angle in [-pi, pi)."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def lookahead_target(
    position: np.ndarray, reference: np.ndarray, lookahead: float
) -> np.ndarray:
    """Project onto the closest segment, then move lookahead metres along it.

    Interpolation uses distance along the path, not a waypoint count or a
    straight-line distance. Closed loops wrap; open paths stop at their end.
    Repeated waypoints are allowed, provided the path has some length.
    """
    segments = np.diff(reference, axis=0)
    squared_lengths = np.sum(segments**2, axis=1)
    lengths = np.sqrt(squared_lengths)
    distance = np.r_[0.0, np.cumsum(lengths)]
    if len(reference) < 2 or not np.isfinite(distance[-1]) or distance[-1] <= 0:
        raise ValueError("Reference must contain a finite path with positive length")
    fractions = np.divide(
        np.sum((position - reference[:-1]) * segments, axis=1),
        squared_lengths,
        out=np.zeros_like(lengths),
        where=squared_lengths > 0.0,
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = reference[:-1] + fractions[:, None] * segments
    closest = np.sum((projections - position) ** 2, axis=1).argmin()
    target_distance = (
        distance[closest] + fractions[closest] * lengths[closest] + lookahead
    )
    if np.array_equal(reference[0], reference[-1]):
        target_distance %= distance[-1]
    else:
        target_distance = min(target_distance, distance[-1])
    return np.array(
        [np.interp(target_distance, distance, reference[:, axis]) for axis in range(2)]
    )
