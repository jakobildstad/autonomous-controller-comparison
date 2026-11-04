"""Tracking distance, RMS accumulation, reset lifecycle, and score overlay."""

import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from simple_robot_comparison.control_input import ControlInput
from simple_robot_comparison.main import main
from simple_robot_comparison.metrics import TrackingScore
from simple_robot_comparison.simulation import Simulation
from simple_robot_comparison.viewer import Viewer


class TrackingScoreTests(unittest.TestCase):
    def test_uses_line_segments_and_clamps_to_endpoints(self):
        reference = np.array([[0, 0], [2, 0], [2, 2]])
        for position, distance in (((1, 0), 0), ((1, 0.3), 0.3), ((3, 3), np.sqrt(2))):
            with self.subTest(position=position):
                score = TrackingScore()
                score.update(position, reference)
                self.assertAlmostEqual(score.rms_error, distance)

    def test_rms_counts_every_sample_including_stationary_positions(self):
        score = TrackingScore()
        self.assertEqual(score.rms_error, 0)
        reference = np.array([[0, 0], [1, 0]])
        for error in (0.03, 0.04, 0.04):
            score.update((0.5, error), reference)
        self.assertEqual(score.samples, 3)
        self.assertAlmostEqual(score.rms_error, np.sqrt((0.03**2 + 2 * 0.04**2) / 3))

    def test_duplicate_waypoints_do_not_change_distance(self):
        score = TrackingScore()
        score.update((0.5, 0.3), np.array([[0, 0], [0, 0], [1, 0]]))
        self.assertAlmostEqual(score.rms_error, 0.3)
        score = TrackingScore()
        score.update((0, 0.3), np.zeros((2, 2)))
        self.assertAlmostEqual(score.rms_error, 0.3)

    def test_invalid_samples_are_rejected_without_affecting_score(self):
        score = TrackingScore()
        reference = np.array([[0, 0], [1, 0]])
        score.update((0, 0.2), reference)
        for point, path in (
            ((np.nan, 0), reference),
            ((0, np.inf), reference),
            ((0, 0), []),
            ((0, 0), [[0, 0]]),
            ((0, 0), [[0, 0], [np.nan, 1]]),
        ):
            with self.subTest(point=point, path=path), self.assertRaises(ValueError):
                score.update(point, np.asarray(path))
        self.assertEqual(score.samples, 1)
        self.assertAlmostEqual(score.rms_error, 0.2)


class ScoreDisplayTests(unittest.TestCase):
    def test_main_resets_score_when_reference_changes(self):
        references = [np.array([[0, 1], [1, 1]]), np.array([[0, 2], [1, 2]])]
        with (
            patch("sys.argv", ["sim", "--mode", "manual"]),
            patch("simple_robot_comparison.main.Viewer") as factory,
            patch(
                "simple_robot_comparison.main.random_reference", side_effect=references
            ),
            patch("simple_robot_comparison.main.time.sleep"),
        ):
            viewer = factory.return_value
            viewer.controls.return_value = ControlInput()
            resets = iter((False, False, True))

            def poll():
                try:
                    viewer.reset_requested = next(resets)
                except StopIteration:
                    return False
                return True

            viewer.poll.side_effect = poll
            main()
            scores = [
                call.kwargs["scores"]["manual"] for call in viewer.draw.call_args_list
            ]
            # Allow the resting car's tiny motion while contacts settle.
            np.testing.assert_allclose(scores, [1, 1, 2], atol=1e-6)
            viewer.close.assert_called_once()

    def test_overlay_shows_each_car_in_centimetres_at_top_right(self):
        car = Simulation()
        viewer = Viewer.__new__(Viewer)
        viewer.scene = mujoco.MjvScene(car.model, maxgeom=100)
        viewer.options = mujoco.MjvOption()
        viewer.camera = mujoco.MjvCamera()
        viewer.context = object()
        viewer.window = object()
        viewer.instructions = "Instructions"
        viewer.trails = {}
        with (
            patch(
                "simple_robot_comparison.viewer.glfw.get_framebuffer_size",
                return_value=(1100, 750),
            ),
            patch("simple_robot_comparison.viewer.glfw.swap_buffers"),
            patch("simple_robot_comparison.viewer.mujoco.mjr_render"),
            patch("simple_robot_comparison.viewer.mujoco.mjr_overlay") as overlay,
        ):
            viewer.draw(
                car.model,
                car.data,
                np.array([[0, 0], [1, 0]]),
                scores={"classic": 0.03, "mpc": 0.042},
            )
            self.assertEqual(overlay.call_count, 2)
            args = overlay.call_args.args
            self.assertEqual(args[1], mujoco.mjtGridPos.mjGRID_TOPRIGHT)
            self.assertIn("RMS cm", args[3])
            self.assertIn("Lower is better", args[3])
            self.assertIn("CLASSIC: 3.00", args[3])
            self.assertIn("MPC: 4.20", args[3])


if __name__ == "__main__":
    unittest.main()
