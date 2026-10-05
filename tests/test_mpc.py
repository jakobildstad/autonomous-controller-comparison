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
        config = MPCConfig(horizon=4, time_step=0.1, speed=0.35)
        targets = reference_horizon(np.array([0.2, 0.5]), reference, config)
        np.testing.assert_allclose(targets[:, 0], 0.2 + np.arange(5) * 0.035)
        np.testing.assert_allclose(targets[:, 1:3], 0)
        np.testing.assert_allclose(targets[:, 3], config.speed)

    def test_closed_path_wraps_across_finish(self):
        reference = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        targets = reference_horizon(
            np.array([0, 0.05]),
            reference,
            MPCConfig(horizon=2, time_step=0.1, speed=0.35),
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
            {"speed_time_constant": 0},
            {"speed_time_constant": float("nan")},
            {"yaw_time_constant": -0.1},
            {"yaw_time_constant": float("inf")},
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
                "--mpc-speed-time-constant",
                "0.08",
                "--mpc-yaw-time-constant",
                "0.12",
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
        self.assertEqual(args.mpc_speed_time_constant, 0.08)
        self.assertEqual(args.mpc_yaw_time_constant, 0.12)
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
            patch(
                "sys.argv",
                [
                    "sim",
                    "--mode",
                    "mpc",
                    "--mpc-horizon",
                    "8",
                    "--mpc-speed-time-constant",
                    "0.08",
                    "--mpc-yaw-time-constant",
                    "0.12",
                ],
            ),
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
            self.assertEqual(
                factory.call_args.kwargs["config"].speed_time_constant, 0.08
            )
            self.assertEqual(factory.call_args.kwargs["config"].yaw_time_constant, 0.12)
            simulation.return_value.reset.assert_called_once()
            viewer.return_value.close.assert_called_once()


class MPCIntegrationTests(unittest.TestCase):
    def test_measured_motion_and_command_history_reach_solver_separately(self):
        controller = MPCController(MPCConfig(horizon=3, time_step=0.02))
        observation = Simulation().observe()
        reference = np.array([[0, 0], [1, 0]])
        controller._mpc.solver_stats = {"success": True}
        # Same body-frame velocity rotated into different world headings;
        # sideways motion must not inflate the signed forward-speed estimate.
        for yaw, vx, vy, forward in (
            (0, 0.2, 0.7, 0.2),
            (np.pi / 2, -0.7, 0.2, 0.2),
            (np.pi / 2, -0.7, -0.1, -0.1),
        ):
            with (
                self.subTest(yaw=yaw, forward=forward),
                patch.object(
                    controller._mpc, "make_step", return_value=np.zeros((2, 1))
                ) as solve,
            ):
                controller.wheel_reference[:] = [4, 6]
                controller(
                    replace(
                        observation,
                        yaw=yaw,
                        vx=vx,
                        vy=vy,
                        yaw_rate=-0.3,
                        left_wheel_speed=14,
                        right_wheel_speed=15,
                    ),
                    reference,
                    0.02,
                )
                state = solve.call_args.args[0].ravel()
                np.testing.assert_allclose(state[3:5], [forward, -0.3])
                np.testing.assert_allclose(state[5:], [0.15, 0.5])

    def test_response_model_matches_continuous_ramp_dynamics(self):
        import casadi as ca
        from scipy.integrate import solve_ivp

        config = MPCConfig(horizon=3, speed_time_constant=0.2, yaw_time_constant=0.3)
        controller = MPCController(config)
        model = controller._mpc.model
        predict = ca.Function("predict", [model.x.cat, model.u.cat], [model._rhs.cat])
        # Check against independently integrated continuous dynamics, including
        # a reversing yaw response and a ramp rather than a held input.
        initial_motion = np.array([0.1, -0.4])
        initial_command = np.array([0.2, 0.6])
        final_command = np.array([0.35, 0.8])

        def dynamics(time, state):
            command = (
                initial_command
                + (final_command - initial_command) * time / config.time_step
            )
            acceleration = (command - state[:2]) / [
                config.speed_time_constant,
                config.yaw_time_constant,
            ]
            return np.r_[acceleration, state[:2]]

        expected = solve_ivp(
            dynamics,
            (0, config.time_step),
            np.r_[initial_motion, 0, 0],
            rtol=1e-10,
            atol=1e-12,
        ).y[:, -1]
        state = np.r_[0, 0, 0, initial_motion, initial_command]
        predicted = np.array(predict(state, final_command)).ravel()
        np.testing.assert_allclose(predicted[3:5], expected[:2], atol=1e-10)
        self.assertAlmostEqual(predicted[2], expected[3], places=10)
        np.testing.assert_allclose(predicted[5:], final_command)
        # Recover the predicted forward displacement from the midpoint heading.
        heading = predicted[2] / 2
        self.assertAlmostEqual(
            predicted[0] * np.cos(heading) + predicted[1] * np.sin(heading),
            expected[2],
            places=10,
        )

    def test_large_motion_mismatch_preserves_wheel_command_limits(self):
        controller = MPCController(MPCConfig(horizon=5))
        observation = replace(Simulation().observe(), vx=-0.3, yaw_rate=6.0)
        reference = np.array([[0, 0], [1, 0]])
        previous_wheels = np.zeros(2)
        for _ in range(10):
            command = controller(observation, reference, 0.02)
            self.assertLessEqual(max(abs(command.forward), abs(command.turn)), 1)
            self.assertLessEqual(
                np.max(np.abs(controller.wheel_reference)), MAX_WHEEL_SPEED + 1e-5
            )
            self.assertLessEqual(
                np.max(np.abs(controller.wheel_reference - previous_wheels)),
                controller.config.wheel_acceleration * 0.02 + 1e-5,
            )
            previous_wheels = controller.wheel_reference.copy()

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
            steps_per_solve = round(controller.config.time_step / simulation.dt)
            self.assertEqual(
                solve.call_count, (400 + steps_per_solve - 1) // steps_per_solve
            )
        self.assertGreater(np.max(np.array(positions)[:, 1]), 0.2)
        self.assertLess(np.sqrt(np.mean(np.square(errors))), 0.08)

    def test_invalid_timing_and_observations_are_rejected(self):
        controller = MPCController(MPCConfig(horizon=3, time_step=0.1))
        observation = Simulation().observe()
        reference = np.array([[0, 0], [1, 0]])
        for dt in (0, -0.02, float("nan"), 0.03, 0.2):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                controller(observation, reference, dt)
        for field in ("yaw", "vx", "vy", "yaw_rate", "left_wheel_speed"):
            for value in (float("nan"), float("inf")):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaisesRegex(
                        ValueError, "finite pose, velocity and wheel speeds"
                    ),
                ):
                    controller(replace(observation, **{field: value}), reference, 0.02)

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
