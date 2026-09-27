"""Register factories that create a fresh controller for each episode."""

from collections.abc import Callable

import numpy as np

from simple_robot_comparison.control_input import ControlInput
from simple_robot_comparison.controllers.classic_control import ClassicController
from simple_robot_comparison.observation import Observation

type Controller = Callable[[Observation, np.ndarray, float], ControlInput]

# A controller can be a callable object or a function returned by a factory.
CONTROLLERS: dict[str, Callable[[], Controller]] = {
    "classic": ClassicController,
}
