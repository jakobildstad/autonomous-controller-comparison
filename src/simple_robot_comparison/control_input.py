"""The motor command shared by manual and autonomous control."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlInput:
    """Motor efforts: positive forward drives ahead; positive turn turns left.

    Each value uses [-1, 1]; Simulation.step clips commands to these limits.
    Zero effort switches the motors off without actively braking. These are
    effort commands, not target speed or steering angle.
    """

    forward: float = 0.0
    turn: float = 0.0
