"""Flat ground with uniform friction or randomly sized slippery patches."""

from itertools import pairwise

import mujoco
import numpy as np

FRICTION_PRESETS = {"high": 1.0, "medium": 0.3, "low": 0.08}
FRICTION_MODES = (*FRICTION_PRESETS, "random")


def add_ground(spec: mujoco.MjSpec, rng: np.random.Generator, friction: str) -> None:
    """Cover the same 6 x 4 metre area with one surface or adjoining patches.

    Every top surface is at z=0. Uniform presets use one box, so there are no
    patch seams. MuJoCo handles contact physics in either case.
    """
    if friction != "random":
        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[1.0, 0.0, -0.05],
            size=[3.0, 2.0, 0.05],
            priority=1,
        )
        return

    heights = rng.uniform(0.5, 1.5, 4)
    y_edges = np.r_[-2.0, -2.0 + np.cumsum(heights) / heights.sum() * 4.0]
    for row, (bottom, top) in enumerate(pairwise(y_edges)):
        widths = rng.uniform(0.5, 1.5, 6)
        x_edges = np.r_[-2.0, -2.0 + np.cumsum(widths) / widths.sum() * 6.0]
        for column, (left, right) in enumerate(pairwise(x_edges)):
            spec.worldbody.add_geom(
                name=f"ground_{row}_{column}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[(left + right) / 2, (bottom + top) / 2, -0.05],
                size=[(right - left) / 2, (top - bottom) / 2, 0.05],
                priority=1,  # Ground friction overrides the wheel's default.
            )


def set_grip(model: mujoco.MjModel, rng: np.random.Generator, friction: str) -> None:
    """Apply a uniform preset, or give about 70% of patches lower grip.

    Random grip uses log-uniform sampling between 0.03 and 0.6; the remaining
    patches have grip 1.0. Pale blue means more slippery. Geometry stays fixed.
    """
    ground_ids = np.flatnonzero(model.geom_bodyid == 0)
    if friction != "random":
        model.geom_friction[ground_ids] = [FRICTION_PRESETS[friction], 0.005, 0.0001]
        model.geom_rgba[ground_ids] = [0.18, 0.22, 0.25, 1.0]
        return

    for geom_id in ground_ids:
        if rng.random() < 0.7:
            shade = rng.uniform(0.0, 1.0)
            coefficient = np.exp(np.log(0.03) + shade * np.log(0.6 / 0.03))
            color = (1 - shade) * np.array([0.7, 0.9, 1.0]) + shade * np.array(
                [0.1, 0.3, 0.5]
            )
        else:
            coefficient = 1.0
            color = [0.18, 0.22, 0.25]
        model.geom_friction[geom_id] = [coefficient, 0.005, 0.0001]
        model.geom_rgba[geom_id] = [*color, 1.0]
