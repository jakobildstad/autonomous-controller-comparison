"""Choose a controller and ground friction, then run the simulation."""

import argparse
import time

from simple_robot_comparison.controllers import CONTROLLERS
from simple_robot_comparison.reference import random_reference
from simple_robot_comparison.simulation import Simulation
from simple_robot_comparison.terrain import FRICTION_MODES
from simple_robot_comparison.viewer import Viewer


def main() -> None:
    """Run manual or autonomous control at a 20 ms simulation timestep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["manual", *CONTROLLERS],
        default="manual",
        help="control method (default: manual)",
    )
    parser.add_argument(
        "--friction",
        choices=FRICTION_MODES,
        default="high",
        help="uniform high=1.0, medium=0.3, low=0.08, or a smooth random map (default: high)",
    )
    args = parser.parse_args()
    controller_factory = CONTROLLERS.get(args.mode)
    controller = controller_factory() if controller_factory is not None else None
    simulation = Simulation(friction=args.friction)
    reference = random_reference()
    viewer = Viewer(simulation.model, mode=args.mode, friction=args.friction)
    try:
        while viewer.poll():
            started = time.perf_counter()
            if viewer.reset_requested:
                simulation.reset()
                viewer.refresh_ground_texture(simulation.model)
                reference = random_reference()
                controller = (
                    controller_factory() if controller_factory is not None else None
                )
                viewer.reset_requested = False

            if controller is None:
                control_input = viewer.controls()
            else:
                control_input = controller(
                    simulation.observe(), reference, simulation.dt
                )
            simulation.step(control_input)
            viewer.draw(simulation.model, simulation.data, reference)
            time.sleep(max(0.0, simulation.dt - (time.perf_counter() - started)))
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
