# Simple Robot Comparison

Manually drive DeepMind's MuJoCo car along a random green closed-loop reference.
The ground has patches of varying size: about 70% have friction between 0.005 and
0.6; the rest have friction 1.0. Lighter blue means more slippery.
This car uses differential drive, with two driven wheels and a front support.

```sh
uv sync
uv run simple-robot-comparison
```

Hold **W/S** (or up/down) to drive and **A/D** (or left/right) to turn.
Release the keys to switch off motor effort. **R** resets the car and generates
a new loop and grip distribution. Patch sizes change when the app restarts.
**Esc** closes the window. Stay on the finite ground area.

- `main.py`: the short control loop; replace `viewer.controls()` to add autonomy.
- `simulation.py`: load, reset, and step the compiled MuJoCo physics.
- `reference.py`: generate a smooth closed loop in metres.
- `terrain.py`: create flat patches and randomize their grip.
- `viewer.py`: keyboard input and MuJoCo rendering.
- `models/car.xml`: the car model.

The reference is visual only. Slip emerges from contact physics; this is not a
deformable-mud model. The simulation can run without the viewer for future RL.

Car adapted from [Google DeepMind's example](https://github.com/google-deepmind/mujoco/blob/main/model/car/car.xml)
under Apache-2.0; see `src/simple_robot_comparison/models/LICENSE`.
