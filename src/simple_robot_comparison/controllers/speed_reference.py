"""Shared path-speed planning for classic and MPC controllers."""

from dataclasses import dataclass

import numpy as np

DEFAULT_MAX_SPEED = 0.5
DEFAULT_MAX_LATERAL_ACCELERATION = 0.3
DEFAULT_MAX_BRAKING_ACCELERATION = 0.25

# Keep the existing classic planner's geometry and wheel/yaw reference limits.
WHEEL_RADIUS = 0.03
WHEEL_SEPARATION = 0.12
MAX_WHEEL_SPEED = 30.0
MAX_YAW_RATE = 10.0


@dataclass(frozen=True)
class SpeedProfile:
    """Path vertices and squared speeds indexed by distance along the path."""

    points: np.ndarray
    distance: np.ndarray
    speed_squared: np.ndarray

    @property
    def closed(self) -> bool:
        return bool(np.array_equal(self.points[0], self.points[-1]))

    def project(self, position: np.ndarray) -> float:
        """Return arc length at the closest point on the polyline."""
        segments = np.diff(self.points, axis=0)
        lengths = np.diff(self.distance)
        fractions = np.clip(
            np.sum((position - self.points[:-1]) * segments, axis=1) / lengths**2,
            0.0,
            1.0,
        )
        projections = self.points[:-1] + fractions[:, None] * segments
        closest = np.sum((projections - position) ** 2, axis=1).argmin()
        return float(self.distance[closest] + fractions[closest] * lengths[closest])

    def speed_at(self, distance: float) -> float:
        """Interpolate v squared, which is linear under constant acceleration."""
        if self.closed:
            distance %= self.distance[-1]
        return float(np.sqrt(np.interp(distance, self.distance, self.speed_squared)))


def speed_reference(
    position: np.ndarray,
    reference: np.ndarray,
    max_speed: float,
    max_lateral_acceleration: float,
    max_braking_acceleration: float,
) -> float:
    """Return the shared planned speed at the car's projection onto the path."""
    profile = build_speed_profile(
        reference, max_speed, max_lateral_acceleration, max_braking_acceleration
    )
    return profile.speed_at(profile.project(position))


def build_speed_profile(
    reference: np.ndarray,
    max_speed: float,
    max_lateral_acceleration: float,
    max_braking_acceleration: float,
) -> SpeedProfile:
    """Build the shared curvature-limited profile with braking before bends.

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

    return SpeedProfile(points, np.r_[0.0, np.cumsum(lengths)], speed_squared)
