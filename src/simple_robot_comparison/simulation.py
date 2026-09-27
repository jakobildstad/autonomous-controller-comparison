"""Load the car and let MuJoCo handle all motion and contact physics."""

from pathlib import Path

import mujoco
import numpy as np

from simple_robot_comparison.terrain import add_ground, randomize_grip


class Simulation:
    """A small interface shared by manual driving and future controllers.

    ``model`` holds the car/ground parameters; ``data`` holds the current state.
    The two control inputs are motor efforts, not target speeds or angles.
    """

    def __init__(self) -> None:
        self.rng = np.random.default_rng()
        spec = mujoco.MjSpec.from_file(
            str(Path(__file__).parent / "models" / "car.xml")
        )
        add_ground(spec, self.rng)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.substeps = 10
        self.dt = self.substeps * self.model.opt.timestep
        self.reset()

    def reset(self) -> None:
        """Reset the car and randomize patch grip, keeping the flat geometry."""
        randomize_grip(self.model, self.rng)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, forward: float, turn: float) -> None:
        """Advance 20 ms: positive forward drives ahead; positive turn goes left."""
        self.data.ctrl[:] = np.clip([forward, turn], -1.0, 1.0)
        # The physics substeps run inside MuJoCo's compiled engine.
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        mujoco.mj_forward(self.model, self.data)
