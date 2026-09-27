"""Load the car and let MuJoCo handle all motion and contact physics."""

from pathlib import Path

import mujoco
import numpy as np

from simple_robot_comparison.control_input import ControlInput
from simple_robot_comparison.observation import Observation
from simple_robot_comparison.terrain import FRICTION_MODES, add_ground, set_grip


class Simulation:
    """A small interface shared by manual driving and future controllers.

    ``model`` holds the car/ground parameters; ``data`` holds the current state.
    The two control inputs are motor efforts, not target speeds or angles.
    """

    def __init__(self, friction: str = "high") -> None:
        if friction not in FRICTION_MODES:
            raise ValueError(
                f"Unknown friction mode: {friction!r}; choose {FRICTION_MODES}"
            )
        self.friction = friction
        self.rng = np.random.default_rng()
        spec = mujoco.MjSpec.from_file(
            str(Path(__file__).parent / "models" / "car.xml")
        )
        add_ground(spec, self.rng, friction)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.car_id = self.model.body("car").id
        self.substeps = 10
        self.dt = self.substeps * self.model.opt.timestep
        self.reset()

    def reset(self) -> None:
        """Reset the car, preserving the preset or resampling random patch grip."""
        set_grip(self.model, self.rng, self.friction)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def observe(self) -> Observation:
        """Copy the car's planar state into a controller-friendly snapshot."""
        position = self.data.xpos[self.car_id]
        rotation = self.data.xmat[self.car_id].reshape(3, 3)
        velocity = np.zeros(6)
        # XBODY measures at the body origin; 0 requests world-oriented axes.
        # MuJoCo returns angular velocity first, then linear velocity.
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.car_id, velocity, 0
        )
        return Observation(
            x=float(position[0]),
            y=float(position[1]),
            yaw=float(np.arctan2(rotation[1, 0], rotation[0, 0])),
            vx=float(velocity[3]),
            vy=float(velocity[4]),
            yaw_rate=float(velocity[2]),
            left_wheel_speed=float(self.data.joint("left").qvel[0]),
            right_wheel_speed=float(self.data.joint("right").qvel[0]),
        )

    def step(self, control_input: ControlInput) -> None:
        """Advance 20 ms, clipping the named motor efforts to [-1, 1]."""
        self.data.ctrl[:] = np.clip(
            [control_input.forward, control_input.turn], -1.0, 1.0
        )
        # The physics substeps run inside MuJoCo's compiled engine.
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        mujoco.mj_forward(self.model, self.data)
