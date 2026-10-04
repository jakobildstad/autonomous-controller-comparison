"""Shared speed limits, curvature/braking constraints, and MPC reference use."""

import io
import unittest
from contextlib import redirect_stderr
from dataclasses import replace

import numpy as np

from simple_robot_comparison.controllers.classic.classic_control import (
    ClassicController,
)
from simple_robot_comparison.controllers.mpc.mpc_control import (
    MPCConfig,
    MPCController,
    reference_horizon,
)
from simple_robot_comparison.controllers.speed_reference import build_speed_profile
from simple_robot_comparison.main import build_parser
from simple_robot_comparison.reference import random_reference
from simple_robot_comparison.simulation import Simulation


class SharedSpeedReferenceTests(unittest.TestCase):
    def test_straight_and_circle_have_expected_speeds(self):
        straight = build_speed_profile(
            np.array([[0, 0], [1, 0], [2, 0]]), 0.5, 0.3, 0.25
        )
        np.testing.assert_allclose(np.sqrt(straight.speed_squared), 0.5)
        angle = np.linspace(0, 2 * np.pi, 101)
        circle = 0.25 * np.column_stack([np.cos(angle), np.sin(angle)])
        circle[-1] = circle[0]
        profile = build_speed_profile(circle, 0.5, 0.3, 0.25)
        np.testing.assert_allclose(np.sqrt(profile.speed_squared), np.sqrt(0.3 * 0.25))

    def test_braking_before_bend_and_across_closed_seam(self):
        reference = random_reference(7)
        profile = build_speed_profile(reference, 0.5, 0.3, 0.25)
        segments = np.diff(profile.distance)
        self.assertLess(np.sqrt(profile.speed_squared).min(), 0.3)
        self.assertAlmostEqual(np.sqrt(profile.speed_squared).max(), 0.5)
        self.assertTrue(
            np.all(
                profile.speed_squared[:-1]
                <= profile.speed_squared[1:] + 2 * 0.25 * segments + 1e-12
            )
        )
        self.assertEqual(profile.speed_squared[0], profile.speed_squared[-1])
        self.assertAlmostEqual(
            profile.speed_at(0.1), profile.speed_at(profile.distance[-1] + 0.1)
        )
        # Rotating the closed path's start must not change the physical profile.
        rotated = np.roll(reference[:-1], 77, axis=0)
        rotated = np.vstack([rotated, rotated[0]])
        rotated_profile = build_speed_profile(rotated, 0.5, 0.3, 0.25)
        np.testing.assert_allclose(
            rotated_profile.speed_squared[:-1], np.roll(profile.speed_squared[:-1], 77)
        )

    def test_same_position_has_same_reference_for_both_controllers(self):
        reference = random_reference(7)
        classic = ClassicController()
        config = MPCConfig()
        self.assertEqual(classic.max_speed, config.speed)
        self.assertEqual(
            classic.max_lateral_acceleration, config.max_lateral_acceleration
        )
        self.assertEqual(
            classic.max_braking_acceleration, config.max_braking_acceleration
        )
        observation = Simulation().observe()
        speeds = []
        for point in reference[::10]:
            with self.subTest(point=point):
                classic(replace(observation, x=point[0], y=point[1]), reference, 0.02)
                targets = reference_horizon(point, reference, config)
                self.assertAlmostEqual(classic.speed_reference, targets[0, 3])
                speeds.append(targets[0, 3])
        self.assertGreater(np.ptp(speeds), 0.15)

    def test_mpc_future_targets_follow_variable_profile(self):
        reference = random_reference(7)
        config = MPCConfig(horizon=50)
        profile = build_speed_profile(
            reference,
            config.speed,
            config.max_lateral_acceleration,
            config.max_braking_acceleration,
        )
        targets = reference_horizon(np.zeros(2), reference, config)
        for target in targets:
            self.assertAlmostEqual(
                target[3], profile.speed_at(profile.project(target[:2]))
            )
        self.assertGreater(np.ptp(targets[:, 3]), 0.15)
        distances = np.linalg.norm(np.diff(targets[:, :2], axis=0), axis=1)
        self.assertLess(distances.min(), config.speed * config.time_step * 0.9)
        self.assertGreater(np.ptp(distances), 0.01)

    def test_variable_reference_reaches_the_solver(self):
        reference = random_reference(7)
        simulation = Simulation()
        classic = ClassicController()
        mpc = MPCController(MPCConfig(horizon=30))
        observation = simulation.observe()
        classic(observation, reference, simulation.dt)
        mpc(observation, reference, simulation.dt)
        self.assertAlmostEqual(classic.speed_reference, mpc.speed_reference)
        speeds = [float(mpc._tvp["_tvp", step, "target", 3]) for step in range(31)]
        self.assertGreater(np.ptp(speeds), 0.15)
        self.assertTrue(mpc._mpc.solver_stats["success"])

    def test_shared_flags_and_legacy_speed_alias(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--mode",
                "compare",
                "--max-speed",
                "0.4",
                "--max-lateral-acceleration",
                "0.2",
                "--max-braking-acceleration",
                "0.15",
            ]
        )
        self.assertEqual(args.mpc_speed, 0.4)
        self.assertEqual(args.max_lateral_acceleration, 0.2)
        self.assertEqual(args.max_braking_acceleration, 0.15)
        self.assertEqual(parser.parse_args(["--mpc-speed", "0.4"]).mpc_speed, 0.4)
        for flag, value in (
            ("--max-speed", "nan"),
            ("--max-lateral-acceleration", "0"),
            ("--max-braking-acceleration", "-1"),
        ):
            with (
                self.subTest(flag=flag),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args([flag, value])


if __name__ == "__main__":
    unittest.main()
