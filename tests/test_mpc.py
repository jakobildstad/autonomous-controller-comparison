"""MPC path sampling, solver integration, motor limits, and CLI lifecycle."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from simple_robot_comparison.control_input import ControlInput
from simple_robot_comparison.controllers.mpc.mpc_control import (
    MAX_WHEEL_SPEED,
    MPCConfig,
    MPCController,
    reference_horizon,
)
from simple_robot_comparison.main import build_parser, main
from simple_robot_comparison.reference import random_reference
from simple_robot_comparison.simulation import Simulation


class ReferenceHorizonTests(unittest.TestCase):
    def test_projection_and_arc_length_ignore_duplicate_waypoints(self):
        reference = np.array([[0, 0], [0, 0], [0.3, 0], [2, 0]])
        config = MPCConfig(horizon=4, speed=0.35)
        targets = reference_horizon(np.array([0.2, 0.5]), reference, config)
        np.testing.assert_allclose(targets[:, 0], 0.2 + np.arange(5) * 0.035)
        np.testing.assert_allclose(targets[:, 1:3], 0)
        np.testing.assert_allclose(targets[:, 3], config.speed)

    def test_closed_path_wraps_across_finish(self):
        reference = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        targets = reference_horizon(
            np.array([0, 0.05]), reference, MPCConfig(horizon=2, speed=0.35)
        )
        np.testing.assert_allclose(targets[:, :2], [[0, 0.05], [0, 0.015], [0.02, 0]])
        np.testing.assert_allclose(targets[:, 2], [-np.pi / 2, -np.pi / 2, 0])

    def test_open_path_clamps_at_end(self):
        targets = reference_horizon(
            np.array([0.99, 0]), np.array([[0, 0], [1, 0]]), MPCConfig(horizon=3)
        )
        np.testing.assert_allclose(targets[:, 0], [0.99, 1, 1, 1])
        np.testing.assert_allclose(targets[:, 1:3], 0)
        np.testing.assert_allclose(targets[1:, 3], 0)

    def test_invalid_paths_are_rejected(self):
        for reference in (
            [],
            [[0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [np.nan, 0]],
            [0, 1],
        ):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                reference_horizon(np.zeros(2), np.array(reference), MPCConfig())


class MPCConfigTests(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self):
        for options in (
            {"horizon": 0},
            {"horizon": 1.5},
            {"horizon": True},
            {"time_step": float("nan")},
            {"speed": 0},
            {"speed": 0.51},
            {"max_yaw_rate": -1},
            {"wheel_acceleration": float("inf")},
            {"position_weight": 0},
            {"heading_weight": -1},
            {"input_weight": -1},
            {"speed_weight": 0},
            {"max_lateral_acceleration": 0},
            {"max_braking_acceleration": -1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                MPCConfig(**options)

    def test_cli_exposes_mpc_and_tuning_flags(self):
        args = build_parser().parse_args(
            [
                "--mode",
                "mpc",
                "--mpc-horizon",
                "8",
                "--mpc-time-step",
                "0.04",
                "--mpc-speed",
                "0.2",
                "--mpc-max-yaw-rate",
                "3",
                "--mpc-wheel-acceleration",
                "10",
                "--mpc-position-weight",
                "40",
                "--mpc-heading-weight",
                "2",
                "--mpc-input-weight",
                "0.3",
                "--mpc-speed-weight",
                "7",
            ]
        )
        self.assertEqual(args.mode, "mpc")
        self.assertEqual(args.mpc_horizon, 8)
        self.assertEqual(args.mpc_time_step, 0.04)
        self.assertEqual(args.mpc_speed, 0.2)
        self.assertEqual(args.mpc_max_yaw_rate, 3)
        self.assertEqual(args.mpc_wheel_acceleration, 10)
        self.assertEqual(args.mpc_position_weight, 40)
        self.assertEqual(args.mpc_heading_weight, 2)
        self.assertEqual(args.mpc_input_weight, 0.3)
        self.assertEqual(args.mpc_speed_weight, 7)
        self.assertEqual(build_parser().parse_args([]).mode, "manual")
        self.assertEqual(
            build_parser().parse_args(["--mode", "classic"]).mode, "classic"
        )

    def test_cli_preserves_configuration_on_reset(self):
        with (
            patch("sys.argv", ["sim", "--mode", "mpc", "--mpc-horizon", "8"]),
            patch("simple_robot_comparison.main.MPCController") as factory,
            patch("simple_robot_comparison.main.Simulation") as simulation,
            patch("simple_robot_comparison.main.Viewer") as viewer,
            patch(
                "simple_robot_comparison.main.random_reference",
                return_value=np.zeros((2, 2)),
            ),
            patch("simple_robot_comparison.main.time.sleep"),
        ):
            simulation.return_value.dt = 0.02
            viewer.return_value.poll.side_effect = [True, False]
            viewer.return_value.reset_requested = True
            factory.return_value.return_value = ControlInput()
            main()
            self.assertEqual(factory.call_count, 2)
            self.assertEqual(factory.call_args_list[0], factory.call_args_list[1])
            self.assertEqual(factory.call_args.kwargs["config"].horizon, 8)
            simulation.return_value.reset.assert_called_once()
            viewer.return_value.close.assert_called_once()


class MPCIntegrationTests(unittest.TestCase):
    def test_tracks_a_bend_with_bounded_effort_and_wheel_acceleration(self):
        simulation = Simulation()
        reference = random_reference(7)
        controller = MPCController()
        positions = []
        errors = []
        previous_wheels = np.zeros(2)
        with patch.object(
            controller._mpc, "make_step", wraps=controller._mpc.make_step
        ) as solve:
            for _ in range(400):
                observation = simulation.observe()
                command = controller(observation, reference, simulation.dt)
                self.assertTrue(np.isfinite([command.forward, command.turn]).all())
                self.assertLessEqual(max(abs(command.forward), abs(command.turn)), 1)
                self.assertLessEqual(
                    np.max(np.abs(controller.wheel_reference)), MAX_WHEEL_SPEED + 1e-5
                )
                self.assertLessEqual(
                    np.max(np.abs(controller.wheel_reference - previous_wheels)),
                    controller.config.wheel_acceleration * simulation.dt + 1e-5,
                )
                previous_wheels = controller.wheel_reference.copy()
                positions.append([observation.x, observation.y])
                errors.append(np.min(np.linalg.norm(reference - positions[-1], axis=1)))
                simulation.step(command)
            self.assertEqual(solve.call_count, 80)
        self.assertGreater(np.max(np.array(positions)[:, 1]), 0.2)
        self.assertLess(np.sqrt(np.mean(np.square(errors))), 0.08)

    def test_invalid_timing_and_observations_are_rejected(self):
        controller = MPCController(MPCConfig(horizon=3))
        observation = Simulation().observe()
        reference = np.array([[0, 0], [1, 0]])
        for dt in (0, -0.02, float("nan"), 0.03, 0.2):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                controller(observation, reference, dt)
        with self.assertRaisesRegex(ValueError, "finite pose"):
            controller(replace(observation, yaw=float("nan")), reference, 0.02)

    def test_failed_solve_is_reported_before_emitting_effort(self):
        controller = MPCController(MPCConfig(horizon=3))
        observation = Simulation().observe()
        reference = np.array([[0, 0], [1, 0]])
        controller._mpc.solver_stats = {"success": False, "return_status": "failed"}
        with (
            patch.object(controller._mpc, "make_step", return_value=np.zeros((2, 1))),
            self.assertRaisesRegex(RuntimeError, "MPC solve failed: failed"),
        ):
            controller(observation, reference, 0.02)
        np.testing.assert_array_equal(controller.wheel_reference, 0)
        controller._mpc.solver_stats = {
            "success": True,
            "return_status": "Solve_Succeeded",
        }
        with (
            patch.object(
                controller._mpc, "make_step", return_value=np.full((2, 1), np.nan)
            ),
            self.assertRaisesRegex(RuntimeError, "non-finite result"),
        ):
            controller(observation, reference, 0.02)

    def test_yaw_feedback_is_continuous_across_angle_wrap(self):
        controller = MPCController(MPCConfig(horizon=3))
        observation = Simulation().observe()
        reference = np.array([[0, 0], [-1, 0]])
        controller._mpc.solver_stats = {"success": True}
        with patch.object(controller._mpc, "make_step", return_value=np.zeros((2, 1))):
            controller(replace(observation, yaw=np.pi - 0.01), reference, 0.02)
            controller(replace(observation, yaw=-np.pi + 0.01), reference, 0.02)
        self.assertAlmostEqual(controller._yaw, np.pi + 0.01)


if __name__ == "__main__":
    unittest.main()
