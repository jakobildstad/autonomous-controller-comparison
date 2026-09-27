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
        help="control method (default: manual; custom is your controller placeholder)",
    )
    parser.add_argument(
        "--friction",
        choices=FRICTION_MODES,
        default="high",
        help="uniform high=1.0, medium=0.3, low=0.08, or random patches (default: high)",
    )
    args = parser.parse_args()
    controller = CONTROLLERS.get(args.mode)
    simulation = Simulation(friction=args.friction)
    reference = random_reference()
    viewer = Viewer(simulation.model, mode=args.mode, friction=args.friction)
    try:
        while viewer.poll():
            started = time.perf_counter()
            if viewer.reset_requested:
                simulation.reset()
                reference = random_reference()
                viewer.reset_requested = False

            if controller is None:
                forward, turn = viewer.controls()
            else:
                forward, turn = controller(
                    simulation.observe(), reference, simulation.dt
                )
            simulation.step(forward, turn)
            viewer.draw(simulation.model, simulation.data, reference)
            time.sleep(max(0.0, simulation.dt - (time.perf_counter() - started)))
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
