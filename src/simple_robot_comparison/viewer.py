"""MuJoCo rendering and held-key input; no vehicle physics lives here."""

from itertools import pairwise

import glfw
import mujoco
import numpy as np

from simple_robot_comparison.control_input import ControlInput
from simple_robot_comparison.terrain import TEXTURE_NAME


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
        self.scene = mujoco.MjvScene(model, maxgeom=1000)
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
        self.instructions += "R: reset + new loop   Esc: quit\nGreen: reference"
        if friction == "random":
            self.instructions += "   Lighter blue: more slippery"
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
        self, model: mujoco.MjModel, data: mujoco.MjData, reference: np.ndarray
    ) -> None:
        """Draw the current state and a green reference line with no collision."""
        mujoco.mjv_updateScene(
            model,
            data,
            self.options,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
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
                [*start, 0.01],
                [*end, 0.01],
            )
            self.scene.ngeom += 1

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

    def close(self) -> None:
        """Release graphics resources while the OpenGL context is still active."""
        self.context.free()
        glfw.destroy_window(self.window)
        glfw.terminate()
