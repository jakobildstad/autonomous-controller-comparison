"""Choose a controller and ground friction, then run the simulation."""

import argparse
import time
from functools import partial
from math import isfinite

from simple_robot_comparison.controllers import CONTROLLERS
from simple_robot_comparison.controllers.mpc.mpc_control import MPCConfig, MPCController
from simple_robot_comparison.reference import random_reference
from simple_robot_comparison.simulation import Simulation
from simple_robot_comparison.terrain import FRICTION_MODES
from simple_robot_comparison.viewer import Viewer


def positive_float(value: str) -> float:
    """Parse a finite, positive motion limit before opening the viewer."""
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Describe controller selection and the MPC tuning flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["manual", *CONTROLLERS, "compare"],
        default="manual",
        help="control method; compare runs classic and MPC together (default: manual)",
    )
    parser.add_argument(
        "--friction",
        choices=FRICTION_MODES,
        default="high",
        help="uniform high=1.0, medium=0.3, low=0.08, or a smooth random map (default: high)",
    )
    defaults = MPCConfig()
    speed_options = parser.add_argument_group(
        "Shared speed reference (classic and MPC)"
    )
    speed_options.add_argument(
        "--max-speed",
        "--mpc-speed",
        dest="mpc_speed",
        type=positive_float,
        default=defaults.speed,
        help=f"maximum reference speed in m/s; MPC supports up to 0.5 (default: {defaults.speed})",
    )
    for name, help_text in (
        ("max_lateral_acceleration", "cornering acceleration limit in m/s^2"),
        ("max_braking_acceleration", "planned braking acceleration in m/s^2"),
    ):
        default = getattr(defaults, name)
        speed_options.add_argument(
            f"--{name.replace('_', '-')}",
            type=positive_float,
            default=default,
            help=f"{help_text} (default: {default})",
        )
    mpc_options = parser.add_argument_group(
        "MPC tuning (used with --mode mpc or compare)"
    )
    for name, value_type, help_text in (
        ("horizon", int, "number of prediction steps"),
        ("time_step", float, "seconds between solves; a multiple of 0.02 s"),
        ("max_yaw_rate", float, "maximum turning rate in rad/s"),
        (
            "wheel_acceleration",
            float,
            "maximum wheel-reference acceleration in rad/s^2",
        ),
        ("position_weight", float, "position tracking cost weight"),
        ("heading_weight", float, "heading tracking cost weight"),
        ("speed_weight", float, "variable speed tracking cost weight"),
        ("input_weight", float, "penalty on changes in speed and turning commands"),
    ):
        default = getattr(defaults, name)
        mpc_options.add_argument(
            f"--mpc-{name.replace('_', '-')}",
            type=value_type,
            default=default,
            help=f"{help_text} (default: {default})",
        )
    return parser


def main() -> None:
    """Run manual or autonomous control at a 20 ms simulation timestep."""
    parser = build_parser()
    args = parser.parse_args()
    controller_factory = CONTROLLERS.get(args.mode)
    speed_limits = {
        "max_speed": args.mpc_speed,
        "max_lateral_acceleration": args.max_lateral_acceleration,
        "max_braking_acceleration": args.max_braking_acceleration,
    }
    classic_factory = partial(CONTROLLERS["classic"], **speed_limits)
    if args.mode == "classic":
        controller_factory = classic_factory
    simulation = Simulation(friction=args.friction)
    if args.mode in ("mpc", "compare"):
        try:
            config = MPCConfig(
                max_lateral_acceleration=args.max_lateral_acceleration,
                max_braking_acceleration=args.max_braking_acceleration,
                **{
                    name.removeprefix("mpc_"): value
                    for name, value in vars(args).items()
                    if name.startswith("mpc_")
                },
            )
            steps = config.time_step / simulation.dt
            if steps < 1 or abs(steps - round(steps)) > 1e-9:
                raise ValueError(
                    "--mpc-time-step must be a positive multiple of 0.02 s"
                )
        except ValueError as error:
            parser.error(str(error))
        controller_factory = partial(MPCController, config=config)
    simulations = {args.mode: simulation}
    factories = {args.mode: controller_factory}
    if args.mode == "compare":
        # Separate physics keeps overlapping cars from influencing one another.
        # Each model needs its own mutable per-wheel contact coefficients.
        simulations = {
            "classic": simulation,
            "mpc": Simulation(friction=args.friction),
        }
        simulations["mpc"].reset(grip=simulation.grip)
        factories = {"classic": classic_factory, "mpc": controller_factory}
    controllers = {
        name: factory() if factory is not None else None
        for name, factory in factories.items()
    }
    reference = random_reference()
    viewer = Viewer(simulation.model, mode=args.mode, friction=args.friction)
    try:
        viewer.reset_trails(simulations)
        while viewer.poll():
            started = time.perf_counter()
            if viewer.reset_requested:
                simulation.reset()
                for other in simulations.values():
                    if other is not simulation:
                        other.reset(grip=simulation.grip)
                viewer.refresh_ground_texture(simulation.model)
                viewer.reset_trails(simulations)
                reference = random_reference()
                controllers = {
                    name: factory() if factory is not None else None
                    for name, factory in factories.items()
                }
                viewer.reset_requested = False

            for name, car in simulations.items():
                controller = controllers[name]
                if controller is None:
                    control_input = viewer.controls()
                else:
                    control_input = controller(car.observe(), reference, car.dt)
                car.step(control_input)
            viewer.record_trails(simulations)
            viewer.draw(
                simulation.model,
                simulation.data,
                reference,
                cars=simulations if args.mode == "compare" else None,
            )
            time.sleep(max(0.0, simulation.dt - (time.perf_counter() - started)))
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
