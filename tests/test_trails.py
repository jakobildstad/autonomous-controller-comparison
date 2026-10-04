"""Trajectory trails use controller positions and stay bounded across resets."""

import unittest

import mujoco
import numpy as np

from simple_robot_comparison.simulation import Simulation
from simple_robot_comparison.viewer import (
    CAR_COLORS,
    MAX_TRAIL_POINTS,
    PATH_HEIGHT,
    Viewer,
)


class TrailTests(unittest.TestCase):
    def setUp(self):
        self.cars = {name: Simulation() for name in CAR_COLORS}
        self.viewer = Viewer.__new__(Viewer)
        self.viewer.trails = {}
        self.viewer.trail_colors = CAR_COLORS
        self.viewer.scene = mujoco.MjvScene(self.cars["classic"].model, maxgeom=100)
        self.viewer.reset_trails(self.cars)

    def test_trail_matches_observed_body_origin_and_reference_plane(self):
        for index, (name, car) in enumerate(self.cars.items()):
            start = self.viewer.trails[name][0].copy()
            car.data.qpos[:3] = [0.2 + index, 0.3, 0.04]
            car.data.qpos[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
            mujoco.mj_forward(car.model, car.data)
            observed = car.observe()
            self.viewer.record_trails({name: car})
            endpoint = np.array([observed.x, observed.y, PATH_HEIGHT])
            np.testing.assert_array_equal(self.viewer.trails[name][-1], endpoint)
            np.testing.assert_array_equal(self.viewer.trails[name][0], start)

        self.viewer._add_trails()
        self.assertEqual(self.viewer.scene.ngeom, 2)
        for index, name in enumerate(self.cars):
            geom = self.viewer.scene.geoms[index]
            axis = geom.mat.reshape(3, 3)[:, 2]
            np.testing.assert_allclose(
                geom.pos,
                self.viewer.trails[name][0],
                atol=1e-7,
            )
            np.testing.assert_allclose(
                geom.pos + geom.size[2] * axis,
                self.viewer.trails[name][-1],
                atol=1e-7,
            )
            np.testing.assert_allclose(geom.rgba[:3], CAR_COLORS[name][:3])
            self.assertLess(geom.rgba[3], 1)
            self.assertEqual(geom.type, mujoco.mjtGeom.mjGEOM_LINE)
            self.assertLessEqual(geom.size[0], 2)

    def test_stationary_samples_are_skipped_and_history_is_bounded(self):
        for _ in range(10):
            self.viewer.record_trails(self.cars)
        self.assertTrue(all(len(trail) == 1 for trail in self.viewer.trails.values()))
        car = self.cars["classic"]
        for step in range(MAX_TRAIL_POINTS + 10):
            car.data.xpos[car.car_id, 0] = step * 0.01
            self.viewer.record_trails({"classic": car})
        self.assertEqual(len(self.viewer.trails["classic"]), MAX_TRAIL_POINTS)
        self.assertGreater(self.viewer.trails["classic"][0][0], 0)
        self.viewer._add_trails()
        self.assertEqual(self.viewer.scene.ngeom, self.viewer.scene.maxgeom)

    def test_reset_removes_old_paths_and_seeds_the_new_start(self):
        for car in self.cars.values():
            car.data.qpos[0] = 0.5
            mujoco.mj_forward(car.model, car.data)
        self.viewer.record_trails(self.cars)
        for car in self.cars.values():
            car.reset()
        self.viewer.reset_trails(self.cars)
        for name, car in self.cars.items():
            self.assertEqual(len(self.viewer.trails[name]), 1)
            np.testing.assert_array_equal(
                self.viewer.trails[name][0],
                [car.observe().x, car.observe().y, PATH_HEIGHT],
            )


if __name__ == "__main__":
    unittest.main()
