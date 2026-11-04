"""Path-tracking accuracy measured independently of controller and rendering."""

from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass
class TrackingScore:
    """RMS distance in metres, sampled once per fixed simulation timestep."""

    squared_error_sum: float = 0.0
    samples: int = 0

    def update(self, position: tuple[float, float], reference: np.ndarray) -> None:
        """Accumulate distance to the nearest point on the reference polyline."""
        point = np.asarray(position, dtype=float)
        points = np.asarray(reference, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("Tracking position must contain two finite coordinates")
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 2
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(
                "Tracking reference must contain at least two finite points"
            )
        segments = np.diff(points, axis=0)
        lengths_squared = np.sum(segments**2, axis=1)
        # Zero-length segments from duplicate waypoints reduce to their endpoint.
        fractions = np.divide(
            np.sum((point - points[:-1]) * segments, axis=1),
            lengths_squared,
            out=np.zeros_like(lengths_squared),
            where=lengths_squared > 0,
        )
        closest = points[:-1] + np.clip(fractions, 0, 1)[:, None] * segments
        self.squared_error_sum += float(np.min(np.sum((point - closest) ** 2, axis=1)))
        self.samples += 1

    @property
    def rms_error(self) -> float:
        """Return zero before the first sample, otherwise the cumulative RMS."""
        return sqrt(self.squared_error_sum / self.samples) if self.samples else 0.0
