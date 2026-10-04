"""MuJoCo rendering and held-key input; no vehicle physics lives here."""

from collections import deque
from collections.abc import Mapping
from itertools import pairwise

import glfw
import mujoco
import numpy as np

from simple_robot_comparison.control_input import ControlInput
from simple_robot_comparison.simulation import Simulation
from simple_robot_comparison.terrain import TEXTURE_NAME

CAR_COLORS = {
    "classic": np.array([0.2, 0.6, 1.0, 1.0]),
    "mpc": np.array([1.0, 0.45, 0.1, 1.0]),
}
PATH_HEIGHT = 0.01
MAX_TRAIL_POINTS = 3000


class Viewer:
    """A GLFW window using MuJoCo's renderer, with a fixed overview camera."""

    def __init__(
        self, model: mujoco.MjModel, mode: str = "manual", friction: str = "high"
    ) -> None:
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW; a desktop display is needed")
        self.window = glfw.create_window(1100, 750, "MuJoCo car", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not open the MuJoCo window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)  # The main loop sets the simulation's real-time pace.
        self.trails: dict[str, deque[np.ndarray]] = {}
        self.trail_colors = (
            CAR_COLORS
            if mode == "compare"
            else {mode: model.geom("chasis").rgba.copy()}
        )
        self.scene = mujoco.MjvScene(
            model, maxgeom=1000 + len(self.trail_colors) * MAX_TRAIL_POINTS
        )
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.options = mujoco.MjvOption()
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.0, 0.75, 0.0]
        self.camera.distance = 3.8
        self.camera.azimuth = 90
        self.camera.elevation = -65
        self.reset_requested = False
        self.instructions = f"Mode: {mode}   Friction: {friction}\n"
        if mode == "manual":
            self.instructions += (
                "W/S or Up/Down: drive\nA/D or Left/Right: turn\n"
                "Release keys: motors off\n"
            )
        self.instructions += (
            "R: reset + new loop   Esc: quit\nBright green line: reference\n"
            "Faint colored lines: driven paths\n"
            "Ground: grey concrete (high), green grass (medium), brown mud (low)"
        )
        if mode == "compare":
            self.instructions += (
                "\nBlue: CLASSIC   Orange: MPC\n"
                "Shared track and start; cars do not collide"
            )
        glfw.set_key_callback(self.window, self._on_key)

    def _on_key(self, window, key, scancode, action, mods) -> None:
        """Handle one-shot commands; driving uses held keys in controls()."""
        if action == glfw.PRESS:
            if key == glfw.KEY_R:
                self.reset_requested = True
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

    def poll(self) -> bool:
        """Process input and report whether the window is still open."""
        glfw.poll_events()
        return not glfw.window_should_close(self.window)

    def refresh_ground_texture(self, model: mujoco.MjModel) -> None:
        """Upload the new grip colors after a simulation reset."""
        mujoco.mjr_uploadTexture(model, self.context, model.texture(TEXTURE_NAME).id)

    def reset_trails(self, cars: Mapping[str, Simulation]) -> None:
        """Start fresh trails at each car's initial tracking position."""
        self.trails.clear()
        self.record_trails(cars)

    def record_trails(self, cars: Mapping[str, Simulation]) -> None:
        """Save body-origin x/y, exactly as exposed by Simulation.observe()."""
        for name, car in cars.items():
            # Copy MuJoCo's mutable position; only project its height to the path.
            position = car.data.xpos[car.car_id].copy()
            position[2] = PATH_HEIGHT
            if name not in self.trails:
                self.trails[name] = deque(maxlen=MAX_TRAIL_POINTS)
            trail = self.trails[name]
            if not trail or np.linalg.norm(position - trail[-1]) >= 0.001:
                trail.append(position)

    def controls(self) -> ControlInput:
        """Return moderate motor efforts while W/S and A/D (or arrows) are held."""
        if not glfw.get_window_attrib(self.window, glfw.FOCUSED):
            return ControlInput()

        def held(*keys: int) -> int:
            return int(
                any(glfw.get_key(self.window, key) == glfw.PRESS for key in keys)
            )

        forward = held(glfw.KEY_W, glfw.KEY_UP) - held(glfw.KEY_S, glfw.KEY_DOWN)
        turn = held(glfw.KEY_A, glfw.KEY_LEFT) - held(glfw.KEY_D, glfw.KEY_RIGHT)
        return ControlInput(forward=0.7 * forward, turn=0.3 * turn)

    def draw(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        reference: np.ndarray,
        *,
        cars: Mapping[str, Simulation] | None = None,
    ) -> None:
        """Draw a shared track and either one car or labeled comparison cars."""
        mujoco.mjv_updateScene(
            model,
            data,
            self.options,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_STATIC if cars else mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        if cars:
            self._add_comparison_cars(cars)
        for start, end in pairwise(reference):
            geom = self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                np.array([0.3, 1.0, 0.35, 1.0]),
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                0.008,
                [*start, PATH_HEIGHT],
                [*end, PATH_HEIGHT],
            )
            self.scene.ngeom += 1
        self._add_trails()

        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            self.instructions,
            "",
            self.context,
        )
        glfw.swap_buffers(self.window)

    def _add_trails(self) -> None:
        """Draw thin, translucent segments on the reference path's plane."""
        for name, trail in self.trails.items():
            color = self.trail_colors[name].copy()
            color[3] = 0.55
            for start, end in pairwise(trail):
                if self.scene.ngeom >= self.scene.maxgeom:
                    return
                geom = self.scene.geoms[self.scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).ravel(),
                    color,
                )
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, start, end)
                self.scene.ngeom += 1

    def _add_comparison_cars(self, cars: Mapping[str, Simulation]) -> None:
        """Append independently simulated cars to the same render-only scene."""
        # All cars use car.xml, so mesh/material IDs match the viewer's context.
        perturb = mujoco.MjvPerturb()
        for index, (name, car) in enumerate(cars.items()):
            color = CAR_COLORS[name]
            first_geom = self.scene.ngeom
            mujoco.mjv_addGeoms(
                car.model,
                car.data,
                self.options,
                perturb,
                mujoco.mjtCatBit.mjCAT_DYNAMIC,
                self.scene,
            )
            chassis_id = car.model.geom("chasis").id
            for geom in self.scene.geoms[first_geom : self.scene.ngeom]:
                if geom.objtype == mujoco.mjtObj.mjOBJ_SITE or (
                    geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
                    and geom.objid == chassis_id
                ):
                    geom.rgba[:] = color
                else:
                    geom.rgba[:] = [0.15, 0.16, 0.18, 1.0]

            position = car.data.xpos[car.car_id]
            # Stagger label heights so both remain readable at the shared start.
            label_position = position + [0.0, 0.0, 0.16 + index * 0.14]
            leader = self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(
                leader,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                color,
            )
            mujoco.mjv_connector(
                leader,
                mujoco.mjtGeom.mjGEOM_LINE,
                2.0,
                position + [0.0, 0.0, 0.04],
                label_position,
            )
            self.scene.ngeom += 1
            label = self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(
                label,
                mujoco.mjtGeom.mjGEOM_LABEL,
                np.zeros(3),
                label_position,
                np.eye(3).ravel(),
                color,
            )
            label.label = name.upper()
            self.scene.ngeom += 1

    def close(self) -> None:
        """Release graphics resources while the OpenGL context is still active."""
        self.context.free()
        glfw.destroy_window(self.window)
        glfw.terminate()
