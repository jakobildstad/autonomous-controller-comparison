"""Flat patches with random dimensions and independently varying grip."""

from itertools import pairwise

import mujoco
import numpy as np


def add_ground(spec: mujoco.MjSpec, rng: np.random.Generator) -> None:
    """Tile a 6 x 4 metre area without gaps, overlaps, or changes in height.

    Row heights and patch widths vary. The car's starting point is inside the
    area; every patch has its top at z=0. MuJoCo handles their contact physics.
    """
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


def randomize_grip(model: mujoco.MjModel, rng: np.random.Generator) -> None:
    """Give about 70% of patches low grip; pale blue means more slippery.

    Log-uniform sampling includes plenty of very slippery patches, rather than
    concentrating samples near the upper limit. Geometry stays fixed on reset.
    """
    for geom_id in np.flatnonzero(model.geom_bodyid == 0):
        if rng.random() < 0.7:
            shade = rng.uniform(0.0, 1.0)
            friction = np.exp(np.log(0.03) + shade * np.log(0.6 / 0.03))
            color = (1 - shade) * np.array([0.7, 0.9, 1.0]) + shade * np.array(
                [0.1, 0.3, 0.5]
            )
        else:
            friction = 1.0
            color = [0.18, 0.22, 0.25]
        model.geom_friction[geom_id] = [friction, 0.005, 0.0001]
        model.geom_rgba[geom_id] = [*color, 1.0]
