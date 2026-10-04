"""One flat surface with a continuous, spatially correlated friction map."""

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.special import expit

FRICTION_PRESETS = {"high": 1.0, "medium": 0.3, "low": 0.08}
FRICTION_MODES = (*FRICTION_PRESETS, "random")
GROUND_MIN = np.array([-2.0, -2.0])
GROUND_SIZE = np.array([6.0, 4.0])
MAP_SHAPE = (256, 384)  # Rows along y, columns along x; about 1.6 cm per pixel.
MIN_FRICTION = 0.03
MAX_FRICTION = 1.0
TEXTURE_NAME = "ground_grip"


def add_ground(spec: mujoco.MjSpec) -> None:
    """Add one 6 x 4 metre collider, its texture, and wheel contact pairs."""
    spec.add_texture(
        name=TEXTURE_NAME,
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        width=MAP_SHAPE[1],
        height=MAP_SHAPE[0],
        rgb1=[1.0, 1.0, 1.0],
    )
    material = spec.add_material(
        name="ground_material", texrepeat=[1, 1], texuniform=False, specular=0.0
    )
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = TEXTURE_NAME
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[1.0, 0.0, -0.05],
        size=[3.0, 2.0, 0.05],
        material="ground_material",
        priority=1,
    )
    for side in ("left", "right"):
        # Explicit pairs override geom friction mixing, so low grip is not
        # replaced by the wheel or ground's larger default coefficient.
        spec.add_pair(
            name=f"{side}_ground",
            geomname1=f"{side} tire",
            geomname2="ground",
            condim=3,
            friction=[1.0, 1.0, 0.005, 0.0001, 0.0001],
        )


class FrictionMap:
    """A fixed grip field for one episode, shared by physics and visualization.

    Gaussian-filtered noise creates broad regions plus smaller irregular spots.
    Bilinear interpolation makes grip continuous between grid samples. This is
    a varying contact coefficient, not a deformable-soil or tire model.
    """

    def __init__(self, rng: np.random.Generator, friction: str) -> None:
        if friction != "random":
            self.values = np.full(MAP_SHAPE, FRICTION_PRESETS[friction])
            return

        field = np.zeros(MAP_SHAPE)
        pixel_size = GROUND_SIZE[::-1] / MAP_SHAPE
        for weight, minimum, maximum in [(0.75, 0.25, 0.65), (0.25, 0.05, 0.14)]:
            # Random x/y smoothing lengths make elongated as well as round
            # regions. Lengths are in metres, independent of grid resolution.
            sigma = rng.uniform(minimum, maximum, 2) / pixel_size
            noise = gaussian_filter(rng.standard_normal(MAP_SHAPE), sigma=sigma)
            field += weight * (noise - noise.mean()) / max(noise.std(), 1e-12)

        # A smooth bounded mapping avoids hard clipping into flat plateaus.
        wetness = expit(2.0 * (field + rng.uniform(-0.3, 0.3)))
        self.values = np.exp(
            np.log(MAX_FRICTION) + wetness * np.log(MIN_FRICTION / MAX_FRICTION)
        )

    def sample(self, positions: np.ndarray) -> np.ndarray:
        """Return grip at an (N, 2) array of world positions, clamping at edges."""
        # Values lie at texture-pixel centres. Use the same coordinates as the
        # ground's 2D texture, rather than assigning a constant value per cell.
        pixels = (positions - GROUND_MIN) / GROUND_SIZE * np.array(
            MAP_SHAPE[::-1]
        ) - 0.5
        return map_coordinates(
            self.values, pixels[:, ::-1].T, order=1, mode="nearest", prefilter=False
        )

    def write_texture(self, model: mujoco.MjModel) -> None:
        """Paint low grip as mud, medium as grass, and high as concrete."""
        friction_levels = np.log(
            [FRICTION_PRESETS[mode] for mode in ("low", "medium", "high")]
        )
        palette = np.array(
            [
                [0.38, 0.26, 0.16],  # Brown mud.
                [0.24, 0.38, 0.16],  # Green grass.
                [0.58, 0.58, 0.58],  # Grey concrete.
            ]
        )
        # Anchor colors at the presets; blend random terrain on the log grip scale.
        log_friction = np.log(self.values)
        colors = np.stack(
            [
                np.interp(log_friction, friction_levels, channel)
                for channel in palette.T
            ],
            axis=-1,
        )
        texture_id = model.texture(TEXTURE_NAME).id
        start = model.tex_adr[texture_id]
        # MuJoCo's box-top texture runs from +y to -y, while map rows run
        # from -y to +y. Flip only the display data to keep physics aligned.
        model.tex_data[start : start + colors.size] = (
            np.round(255 * colors[::-1]).astype(np.uint8).ravel()
        )
