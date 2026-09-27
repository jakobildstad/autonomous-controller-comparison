"""Import each controller function here and give it a CLI mode name."""

from collections.abc import Callable

import numpy as np

from simple_robot_comparison.controllers.classic_control import control as classic_control
from simple_robot_comparison.observation import Observation

type Controller = Callable[[Observation, np.ndarray, float], tuple[float, float]]

# Add another module with the same signature, then register its function here.
CONTROLLERS: dict[str, Controller] = {
    "pd": classic_control,
    
    
    
    
    }
