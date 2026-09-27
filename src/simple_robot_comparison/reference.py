"""Generate small, varied closed circuits for the car to follow."""

from itertools import pairwise

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


def random_reference(seed: int | None = None) -> np.ndarray:
    """Return 350 approximately evenly spaced points, including loop closure.

    Random points define a convex hull. Displaced edge midpoints add inward
    bends, then Bezier curves round the corners. Reject cramped or crossing
    layouts. The start/finish remains at (0, 0), facing +x along a straight.
    """
    rng = np.random.default_rng(seed)
    for _ in range(200):
        corners = _random_corners(rng)
        if np.any(corners < [-1.6, 0.0]) or np.any(corners > [1.6, 1.7]):
            continue
        incoming = corners - np.roll(corners, 1, axis=0)
        lengths = np.linalg.norm(incoming, axis=1)
        if lengths.min() < 0.3:
            continue
        directions = incoming / lengths[:, None]
        outgoing = np.roll(directions, -1, axis=0)
        turn_cosines = np.sum(directions * outgoing, axis=1)
        # Avoid nearly reversing direction at a corner (turns above 140 degrees).
        if turn_cosines.min() < np.cos(np.deg2rad(140)):
            continue
        turns = directions[:, 0] * outgoing[:, 1] - directions[:, 1] * outgoing[:, 0]
        if np.count_nonzero(turns < -0.1) < 2:
            continue  # Require at least two inward bends instead of an oval.

        path = _round_corners(corners, rng)
        if _has_crossing(path):
            continue
        distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        separation = np.abs(distance[:, None] - distance[None, :])
        separation = np.minimum(separation, distance[-1] - separation)
        # Neighbours along the line are naturally close. Separate distant parts
        # of the lap by at least 22 cm, slightly more than the car's length.
        if np.any((separation > 0.65) & (cdist(path, path) < 0.22)):
            continue
        return path
    raise RuntimeError("Could not generate a well-spaced track after 200 attempts")


def _random_corners(rng: np.random.Generator) -> np.ndarray:
    """Build a new polygon, with the longest hull edge reserved for the start."""
    points = rng.uniform(-1.0, 1.0, (rng.integers(12, 21), 2))
    hull = points[ConvexHull(points).vertices]  # Counter-clockwise order.
    edges = np.roll(hull, -1, axis=0) - hull
    start_edge = np.linalg.norm(edges, axis=1).argmax()
    hull = np.roll(hull, -start_edge - 1, axis=0)

    # Rotate the starting edge onto +x, centred at the car. The hull is above it.
    forward = hull[0] - hull[-1]
    forward /= np.linalg.norm(forward)
    rotation = np.array([[forward[0], -forward[1]], [forward[1], forward[0]]])
    hull = (hull - (hull[0] + hull[-1]) / 2) @ rotation
    hull[:, 0] *= 1.6 / np.max(np.abs(hull[:, 0]))
    hull[:, 1] *= 1.7 / hull[:, 1].max()
    hull[[0, -1], 1] = 0.0

    corners = []
    for start, end in pairwise(hull):
        corners.append(start)
        edge = end - start
        length = np.linalg.norm(edge)
        if length > 0.65 and rng.random() < 0.85:
            normal = np.array([-edge[1], edge[0]]) / length
            midpoint = start + rng.uniform(0.35, 0.65) * edge
            # Vary indentation depth and position. Bounds and clearance are
            # checked afterwards because deep offsets can cross another edge.
            depth = rng.uniform(0.2, 0.65) * min(length, 1.2)
            corners.append(midpoint + depth * normal)
    corners.append(hull[-1])
    return np.array(corners)


def _round_corners(corners: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Join straight sections with tangent-matched quadratic Bezier curves."""
    incoming = corners - np.roll(corners, 1, axis=0)
    outgoing = np.roll(corners, -1, axis=0) - corners
    incoming_length = np.linalg.norm(incoming, axis=1)
    outgoing_length = np.linalg.norm(outgoing, axis=1)
    trim = np.minimum(
        rng.uniform(0.18, 0.35, len(corners)),
        0.4 * np.minimum(incoming_length, outgoing_length),
    )
    entries = corners - incoming * (trim / incoming_length)[:, None]
    exits = corners + outgoing * (trim / outgoing_length)[:, None]

    path = [np.zeros(2)]
    for entry, corner, exit_point in zip(entries, corners, exits, strict=True):
        for t in np.linspace(0.0, 1.0, 16):
            path.append(
                (1 - t) ** 2 * entry + 2 * t * (1 - t) * corner + t**2 * exit_point
            )
    path.append(np.zeros(2))
    path = np.array(path)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    samples = np.linspace(0.0, distance[-1], 350)
    return np.column_stack(
        [np.interp(samples, distance, path[:, axis]) for axis in range(2)]
    )


def _has_crossing(path: np.ndarray) -> bool:
    """Check all non-neighbouring segments, including the closing segment."""
    starts, ends = path[:-1], path[1:]
    edges = ends - starts
    offsets = starts[None, :, :] - starts[:, None, :]

    def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    side_start = cross(edges[:, None, :], offsets)
    side_end = cross(edges[:, None, :], ends[None, :, :] - starts[:, None, :])
    straddles = side_start * side_end <= 0.0
    lower = np.maximum(np.minimum(starts, ends)[:, None], np.minimum(starts, ends))
    upper = np.minimum(np.maximum(starts, ends)[:, None], np.maximum(starts, ends))
    intersects = straddles & straddles.T & np.all(lower <= upper, axis=-1)
    non_neighbours = np.triu(np.ones(intersects.shape, dtype=bool), k=2)
    non_neighbours[0, -1] = False  # First and last segments share the start line.
    return bool(np.any(intersects & non_neighbours))
