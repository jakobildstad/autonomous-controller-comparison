"""The state available to every autonomous controller."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    """Ideal planar state, read directly from the simulation without noise.

    Position (x, y) and velocity (vx, vy) use world coordinates, in metres and
    metres/second. Yaw is in radians, counter-clockwise from +x. Yaw rate is
    world-z angular velocity in radians/second (planar yaw rate).
    Wheel speeds are ideal encoder readings in rad/s, positive rolling forward.
    Terrain friction is deliberately absent: a controller must infer grip.
    """

    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    yaw_rate: float
    left_wheel_speed: float
    right_wheel_speed: float
