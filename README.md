# Simple Robot Comparison

Drive DeepMind's MuJoCo car along a random green closed-loop reference, manually
or with your own controller. The car uses differential drive, with two driven
wheels and a front support.

```sh
uv sync
uv run sim
uv run sim --mode manual --friction random
uv run sim --mode custom --friction medium
```

The default is `--mode manual --friction high`. Use `--help` for all options.

| `--friction` | Ground |
| --- | --- |
| `high` | Uniform friction 1.0; no patches (default) |
| `medium` | Uniform friction 0.3; no patches |
| `low` | Uniform friction 0.08; no patches |
| `random` | About 70% of patches have friction 0.03–0.6; the rest have 1.0 |

In manual mode, hold **W/S** (or up/down) to drive and **A/D** (or left/right)
to turn. Releasing the keys switches off motor effort; it does not brake.
**R** resets the car and generates a new loop. Uniform friction stays unchanged;
random patch grip is resampled. Patch sizes change when the app restarts.
Lighter blue patches are more slippery. **Esc** closes the window.

To implement control, edit
[`controllers/custom.py`](src/simple_robot_comparison/controllers/custom.py):

```python
forward, turn = control(observation, reference, dt)
```

The function is called every 0.02 simulation seconds. `observation` contains
world position, heading, and velocities; `reference` contains ordered `(x, y)`
points in metres forming a closed loop. `dt` is the timestep in seconds.
Observations are ideal simulated state and exclude terrain friction.

Return two motor efforts in `[-1, 1]`: positive `forward` drives ahead and
positive `turn` turns left. The placeholder returns zero and does not follow
the path yet. Add each future method in its own file and register its function
in [`CONTROLLERS`](src/simple_robot_comparison/controllers/__init__.py) to expose
it through `--mode`.

- `main.py`: CLI options and the short control loop.
- `controllers/`: controller functions and their mode names.
- `observation.py`: controller inputs and their units.
- `simulation.py`: load, reset, and step the compiled MuJoCo physics.
- `reference.py`: generate a smooth closed loop in metres.
- `terrain.py`: create uniform ground or random friction patches.
- `viewer.py`: keyboard input and MuJoCo rendering.
- `models/car.xml`: the car model.

The reference has no collision geometry; stay on the finite ground area.
Slip emerges from contact physics; this is not a deformable-mud model.
The simulation can run without the viewer for future RL.

Car adapted from [Google DeepMind's example](https://github.com/google-deepmind/mujoco/blob/main/model/car/car.xml)
under Apache-2.0; see `src/simple_robot_comparison/models/LICENSE`.

## Autonomous Systems

### classic_control

A simple path-following controller for a stable differential-drive rover: LOS selects the desired heading, and PD determines the turning rate.

#### Select a lookahead target

Find the closest point on the path to the rover. From that point, move a fixed distance $L$ forward **along the path** to obtain the target $(x_m,y_m)$.

For a waypoint path, accumulate segment lengths and interpolate within the segment where you reach $L$. At the path end, use the endpoint. Larger $L$ gives gentler corrections; smaller $L$ gives sharper corrections.

#### Calculate the turning command

Given rover position $(x,y)$ and heading $\theta$:

$$
\theta_{\mathrm{ref}}=\operatorname{atan2}(y_m-y,\;x_m-x)
$$

$$
e_k=\operatorname{wrap}(\theta_{\mathrm{ref}}-\theta),
\qquad
\dot e_k\approx\frac{\operatorname{wrap}(e_k-e_{k-1})}{\Delta t}
$$

$$
\omega_{\mathrm{cmd}}
=\operatorname{clip}(K_p e_k+K_d\dot e_k,\;-\omega_{\max},\;\omega_{\max})
$$

`wrap` maps angles to $[-\pi,\pi)$. Use radians and optionally filter the derivative.

#### Calculate wheel-speed references

Start with a low constant forward-speed reference $v_{\mathrm{cmd}}=v_0$. For wheel radius $r$ and wheel separation $b$:

$$
\dot\phi_{R,\mathrm{ref}}
=\frac{v_0+\frac b2\omega_{\mathrm{cmd}}}{r},
\qquad
\dot\phi_{L,\mathrm{ref}}
=\frac{v_0-\frac b2\omega_{\mathrm{cmd}}}{r}
$$

Positive wheel speeds mean forward motion; positive $\omega$ means counterclockwise rotation.