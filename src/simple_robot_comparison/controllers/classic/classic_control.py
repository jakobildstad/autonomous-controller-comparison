"""Curvature-based speed, line-of-sight guidance, PD, and wheel-speed feedback."""

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
MAX_WHEEL_SPEED = 30.0  # rad/s
MAX_WHEEL_ACCELERATION = 15.0  # rad/s^2


class ClassicController:
    """A callable controller holding the previous PD error and speed commands.

    Create a fresh instance for each episode. The interface stays
    ``controller(observation, reference, dt) -> ControlInput``.
    """

    def __init__(self) -> None:
        self.lookahead = 0.18  # metres along the path
        self.max_speed = WHEEL_RADIUS * MAX_WHEEL_SPEED  # 0.36 m/s straight ahead
        self.max_lateral_acceleration = 0.3  # m/s^2; assumed grip, not measured
        self.max_braking_acceleration = 0.25  # m/s^2; planned deceleration
        self.speed_reference = 0.0  # current v_ref in m/s
        self.heading_kp = 4.0  # (rad/s) / rad
        self.heading_kd = 0.1
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
        self.speed_reference = speed_reference(
            position,
            reference,
            self.max_speed,
            self.max_lateral_acceleration,
            self.max_braking_acceleration,
        )
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
                    self.speed_reference - WHEEL_SEPARATION * yaw_rate / 2,
                    self.speed_reference + WHEEL_SEPARATION * yaw_rate / 2,
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


def speed_reference(
    position: np.ndarray,
    reference: np.ndarray,
    max_speed: float,
    max_lateral_acceleration: float,
    max_braking_acceleration: float,
) -> float:
    """Return the largest local speed allowed by the planned path constraints.

    Estimate curvature from three neighbouring points, cap speed by lateral
    acceleration and wheel/yaw limits, then propagate braking limits backwards.
    Closed loops include braking across the start line. Open paths do not impose
    a terminal stop. Consecutive duplicate points are ignored.

    Limits are nominal assumptions, not knowledge of the terrain: this plans a
    reference, not a guaranteed minimum lap time or a tire-slip safety bound.
    """
    limits = np.array([max_speed, max_lateral_acceleration, max_braking_acceleration])
    if not np.all(np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError("Speed and acceleration limits must be finite and positive")
    points = np.asarray(reference, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 2
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("Reference must contain at least two finite (x, y) points")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, lengths > 1e-9]]
    if len(points) < 2:
        raise ValueError("Reference must have positive length")
    closed = np.array_equal(points[0], points[-1])
    vertices = points[:-1] if closed else points
    if closed and len(vertices) < 3:
        raise ValueError("A closed reference needs at least three distinct vertices")

    # Circumcircle curvature: |kappa| = 2*|cross(a,b)| / (|a|*|b|*|a+b|).
    # Unlike differences by waypoint index, this handles unequal point spacing.
    incoming = vertices - np.roll(vertices, 1, axis=0)
    outgoing = np.roll(vertices, -1, axis=0) - vertices
    denominator = (
        np.linalg.norm(incoming, axis=1)
        * np.linalg.norm(outgoing, axis=1)
        * np.linalg.norm(incoming + outgoing, axis=1)
    )
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    curvature = np.divide(
        2 * np.abs(cross),
        denominator,
        out=np.zeros(len(vertices)),
        where=denominator > 1e-12,
    )
    if not closed:
        curvature[[0, -1]] = curvature[[1, -2]] if len(vertices) > 2 else 0.0
    else:
        curvature = np.r_[curvature, curvature[0]]

    # On the path, yaw rate is v*kappa and the outer wheel travels faster.
    # Reserve that wheel-speed headroom before the downstream motor limiter.
    safe_curvature = np.maximum(curvature, 1e-6)
    wheel_limit = (
        WHEEL_RADIUS * MAX_WHEEL_SPEED / (1 + WHEEL_SEPARATION * curvature / 2)
    )
    speed_squared = np.minimum(max_speed, wheel_limit) ** 2
    speed_squared = np.minimum(speed_squared, max_lateral_acceleration / safe_curvature)
    speed_squared = np.minimum(speed_squared, (MAX_YAW_RATE / safe_curvature) ** 2)

    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    # v_i^2 <= v_(i+1)^2 + 2*a_brake*ds. A second lap carries a limit near
    # the start backwards through the finish; further laps cannot tighten it.
    for _ in range(2 if closed else 1):
        for i in range(len(lengths) - 1, -1, -1):
            speed_squared[i] = min(
                speed_squared[i],
                speed_squared[i + 1] + 2 * max_braking_acceleration * lengths[i],
            )
        if closed:
            speed_squared[-1] = speed_squared[0]

    # Sample at the car's projection, not the LOS target: braking is already
    # built into the profile. Interpolate v^2, which is linear under constant a.
    fractions = np.clip(
        np.sum((position - points[:-1]) * segments, axis=1) / lengths**2, 0.0, 1.0
    )
    projections = points[:-1] + fractions[:, None] * segments
    closest = np.sum((projections - position) ** 2, axis=1).argmin()
    fraction = fractions[closest]
    return float(
        np.sqrt(
            (1 - fraction) * speed_squared[closest]
            + fraction * speed_squared[closest + 1]
        )
    )


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
