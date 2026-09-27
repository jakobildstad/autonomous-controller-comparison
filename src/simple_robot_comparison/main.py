"""Manual driving now; replace the control line here to add autonomy later."""

import time

from simple_robot_comparison.reference import random_reference
from simple_robot_comparison.simulation import Simulation
from simple_robot_comparison.viewer import Viewer


def main() -> None:
    """Run one car and one reference path at real-time speed."""
    simulation = Simulation()
    reference = random_reference()
    viewer = Viewer(simulation.model)
    try:
        while viewer.poll():
            started = time.perf_counter()
            if viewer.reset_requested:
                simulation.reset()
                reference = random_reference()
                viewer.reset_requested = False

            # A future controller takes simulation.data and reference here.
            forward, turn = viewer.controls()
            simulation.step(forward, turn)
            viewer.draw(simulation.model, simulation.data, reference)
            time.sleep(max(0.0, simulation.dt - (time.perf_counter() - started)))
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
