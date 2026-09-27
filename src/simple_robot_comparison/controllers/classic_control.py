"""Start implementing your controller in this file."""

import numpy as np

from simple_robot_comparison.observation import Observation


def control(
    observation: Observation, reference: np.ndarray, dt: float
) -> tuple[float, float]:
    """Return (forward, turn) motor efforts for the next dt seconds.

    ``observation`` describes the car now. ``reference`` contains ordered
    (x, y) points in world metres; its last point repeats the first.
    Both outputs use [-1, 1]: positive forward drives ahead, positive turn
    turns left. These are efforts, not speed or steering-angle commands.

    Replace the return below with your logic. Zero effort switches the motors
    off; it does not actively brake. This placeholder does not follow the path.
    """

    
    
    return 0.0, 0.0
