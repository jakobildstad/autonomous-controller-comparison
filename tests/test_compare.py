"""Shared terrain, independent cars, comparison labels, and reset routing."""

import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from simple_robot_comparison.control_input import ControlInput
from simple_robot_comparison.controllers import CONTROLLERS
from simple_robot_comparison.main import build_parser, main
from simple_robot_comparison.simulation import Simulation
from simple_robot_comparison.terrain import FRICTION_MODES
from simple_robot_comparison.viewer import CAR_COLORS, Viewer


class ComparisonTests(unittest.TestCase):
    def test_shared_map_and_equal_start_in_every_friction_mode(self):
        for friction in FRICTION_MODES:
            with self.subTest(friction=friction):
                classic = Simulation(friction)
                mpc = Simulation(friction)
                mpc.reset(grip=classic.grip)
                self.assertIs(classic.grip, mpc.grip)
                np.testing.assert_array_equal(classic.data.qpos, mpc.data.qpos)
                np.testing.assert_array_equal(
                    classic.model.tex_data, mpc.model.tex_data
                )
                self.assertIsNot(classic.model, mpc.model)
                self.assertFalse(
                    np.shares_memory(
                        classic.model.pair_friction, mpc.model.pair_friction
                    )
                )

    def test_independent_physics_reproduces_the_same_inputs(self):
        first = Simulation("random")
        second = Simulation("random")
        second.reset(grip=first.grip)
        command = ControlInput(forward=0.25, turn=0.03)
        for _ in range(25):
            before = second.data.qpos.copy()
            first.step(command)
            np.testing.assert_array_equal(second.data.qpos, before)
            second.step(command)
            np.testing.assert_array_equal(first.data.qpos, second.data.qpos)
            self.assertEqual(first.data.time, second.data.time)
        first.step(ControlInput(forward=-0.25))
        second.step(command)
        self.assertGreater(np.linalg.norm(first.data.qpos - second.data.qpos), 1e-5)

    def test_reset_replaces_random_map_for_both_cars(self):
        first = Simulation("random")
        second = Simulation("random")
        second.reset(grip=first.grip)
        old_values = first.grip.values.copy()
        first.step(ControlInput(forward=0.2))
        second.step(ControlInput(turn=0.2))
        first.reset()
        second.reset(grip=first.grip)
        self.assertFalse(np.array_equal(first.grip.values, old_values))
        self.assertIs(first.grip, second.grip)
        np.testing.assert_array_equal(first.data.qpos, second.data.qpos)
        self.assertEqual(first.data.time, 0)
        self.assertEqual(second.data.time, 0)

    def test_scene_has_one_ground_and_two_distinct_labeled_cars(self):
        cars = {name: Simulation() for name in ("classic", "mpc")}
        primary = cars["classic"]
        # Scene construction works without a window or an OpenGL context.
        viewer = Viewer.__new__(Viewer)
        viewer.options = mujoco.MjvOption()
        viewer.scene = mujoco.MjvScene(primary.model, maxgeom=100)
        mujoco.mjv_updateScene(
            primary.model,
            primary.data,
            viewer.options,
            None,
            mujoco.MjvCamera(),
            mujoco.mjtCatBit.mjCAT_STATIC,
            viewer.scene,
        )
        viewer._add_comparison_cars(cars)
        geoms = viewer.scene.geoms[: viewer.scene.ngeom]
        labels = [geom for geom in geoms if geom.type == mujoco.mjtGeom.mjGEOM_LABEL]
        self.assertEqual([geom.label for geom in labels], ["CLASSIC", "MPC"])
        self.assertGreater(abs(labels[0].pos[2] - labels[1].pos[2]), 0.1)
        for geom, name in zip(labels, cars, strict=True):
            np.testing.assert_allclose(geom.rgba, CAR_COLORS[name])
        ground_id = primary.model.geom("ground").id
        chassis_id = primary.model.geom("chasis").id
        physical_geoms = [
            geom for geom in geoms if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
        ]
        self.assertEqual(sum(geom.objid == ground_id for geom in physical_geoms), 1)
        chassis = [geom for geom in physical_geoms if geom.objid == chassis_id]
        self.assertEqual(len(chassis), 2)
        for geom, name in zip(chassis, cars, strict=True):
            np.testing.assert_allclose(geom.rgba, CAR_COLORS[name])

    def test_compare_cli_runs_and_resets_both_controllers_with_mpc_flags(self):
        self.assertEqual(
            build_parser().parse_args(["--mode", "compare"]).mode, "compare"
        )
        classic_factory = Mock(
            side_effect=[Mock(), Mock(return_value=ControlInput(forward=0.2))]
        )
        mpc_factory = Mock(
            side_effect=[Mock(), Mock(return_value=ControlInput(turn=0.1))]
        )
        with (
            patch(
                "sys.argv",
                [
                    "sim",
                    "--mode",
                    "compare",
                    "--friction",
                    "random",
                    "--mpc-horizon",
                    "8",
                    "--max-speed",
                    "0.4",
                    "--max-lateral-acceleration",
                    "0.2",
                    "--max-braking-acceleration",
                    "0.15",
                ],
            ),
            patch.dict(CONTROLLERS, {"classic": classic_factory}),
            patch("simple_robot_comparison.main.MPCController", mpc_factory),
            patch("simple_robot_comparison.main.Viewer") as viewer,
            patch("simple_robot_comparison.main.time.sleep"),
        ):
            viewer.return_value.poll.side_effect = [True, False]
            viewer.return_value.reset_requested = True
            main()
            self.assertEqual(classic_factory.call_count, 2)
            self.assertEqual(mpc_factory.call_count, 2)
            self.assertEqual(
                mpc_factory.call_args_list[0], mpc_factory.call_args_list[1]
            )
            self.assertEqual(mpc_factory.call_args.kwargs["config"].horizon, 8)
            config = mpc_factory.call_args.kwargs["config"]
            self.assertEqual(
                classic_factory.call_args_list[0], classic_factory.call_args_list[1]
            )
            self.assertEqual(
                classic_factory.call_args.kwargs,
                {
                    "max_speed": config.speed,
                    "max_lateral_acceleration": config.max_lateral_acceleration,
                    "max_braking_acceleration": config.max_braking_acceleration,
                },
            )
            self.assertEqual(config.speed, 0.4)
            self.assertEqual(config.max_lateral_acceleration, 0.2)
            self.assertEqual(config.max_braking_acceleration, 0.15)
            cars = viewer.return_value.draw.call_args.kwargs["cars"]
            self.assertEqual(tuple(cars), ("classic", "mpc"))
            self.assertIs(cars["classic"].grip, cars["mpc"].grip)
            self.assertAlmostEqual(cars["classic"].data.time, 0.02)
            self.assertEqual(cars["classic"].data.time, cars["mpc"].data.time)
            np.testing.assert_allclose(cars["classic"].data.ctrl, [0.2, 0])
            np.testing.assert_allclose(cars["mpc"].data.ctrl, [0, 0.1])
            viewer.return_value.controls.assert_not_called()
            viewer.return_value.refresh_ground_texture.assert_called_once()
            self.assertEqual(viewer.return_value.reset_trails.call_count, 2)
            viewer.return_value.record_trails.assert_called_once_with(cars)
            viewer.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
