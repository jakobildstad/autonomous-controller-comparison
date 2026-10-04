"""Load the car and let MuJoCo handle all motion and contact physics."""

from pathlib import Path

import mujoco
import numpy as np

from simple_robot_comparison.control_input import ControlInput
from simple_robot_comparison.observation import Observation
from simple_robot_comparison.terrain import (
    FRICTION_MODES,
    FRICTION_PRESETS,
    FrictionMap,
    add_ground,
)


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
        add_ground(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.car_id = self.model.body("car").id
        self._wheel_body_ids = [
            self.model.body(f"{side} wheel").id for side in ("left", "right")
        ]
        self._wheel_pair_ids = [
            self.model.pair(f"{side}_ground").id for side in ("left", "right")
        ]
        self.substeps = 10
        self.dt = self.substeps * self.model.opt.timestep
        self.reset()

    def reset(self, grip: FrictionMap | None = None) -> None:
        """Reset with a supplied shared map, or generate a new episode's map."""
        self.grip = grip if grip is not None else FrictionMap(self.rng, self.friction)
        self.grip.write_texture(self.model)
        self.model.geom("ground").friction[0] = FRICTION_PRESETS.get(self.friction, 1.0)
        mujoco.mj_resetData(self.model, self.data)
        self._update_wheel_grip()
        mujoco.mj_forward(self.model, self.data)

    def _update_wheel_grip(self) -> None:
        """Sample beneath each wheel centre and set its ground contact friction."""
        # Refresh poses after integration before looking up the next contact's
        # grip. Wheel-centre projection approximates the small contact footprint.
        mujoco.mj_kinematics(self.model, self.data)
        positions = self.data.xpos[self._wheel_body_ids, :2]
        coefficients = self.grip.sample(positions)
        self.model.pair_friction[self._wheel_pair_ids, :2] = coefficients[:, None]

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
        if self.friction == "random":
            # Contact physics still runs in MuJoCo; Python updates the spatial
            # coefficient before each 2 ms step, not just once per control input.
            for _ in range(self.substeps):
                self._update_wheel_grip()
                mujoco.mj_step(self.model, self.data)
            self._update_wheel_grip()
        else:
            mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        mujoco.mj_forward(self.model, self.data)
