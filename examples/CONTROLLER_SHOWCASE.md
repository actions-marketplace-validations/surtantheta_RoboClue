# Same positions, different sampled path

Generate a complete local case with `policy-doctor demo /tmp/controller-incident`. The legacy source example `python examples/create_controller_showcase.py /tmp/controller-incident` uses the same packaged implementation. The destination must be new. Core NumPy/safetensors installation suffices; no ROS, simulator, network or policy weights are needed. The generated reports include JSON and escaped Markdown.

The reference native `JointTrajectory` has one joint, positions 0 at time 0 and 1 at time 1 second. With no velocity/acceleration fields, the selected `splines` rule interpolates linearly. The candidate keeps those positions/times and adds velocities 4 and 0, selecting a cubic:

`position(t) = 4t − 5t² + 2t³`

At 0.75 seconds its sampled position is 1.03125, outside the fixture's declared [0,1] reference bounds. These fixture units are not a physical calibration. The corrected artifact restores the reference message and numerical parity; valid joint-name reordering also stays equal. A separate delayed-first-point message stays unknown before that point because initial state is absent.

```sh
policy-doctor compare-trajectories /tmp/controller-incident/reference.json /tmp/controller-incident/candidate.json --times 0 0.25 0.5 0.75 1 --markdown
policy-doctor sample-trajectory /tmp/controller-incident/missing_initial_state.json --times 0 1 1.5 --markdown
```

The comparison exits 1; the missing-state example exits 3. MCP exposes the same engines as `sample_ros_trajectory` and `compare_ros_trajectories`, including optional joint-named numeric `position_limits`. Limits are supplied declarations; bounds checks apply only at sampled times. The comparison exposes per-side bounds/sampling verdicts separately from reference equality. Equal references with violated supplied bounds yield an error.

## Real published input

[Issue1298](https://github.com/ros-controls/ros2_controllers/issues/1298) supplies a full native 32-point/six-joint trajectory. The author's later [comment](https://github.com/ros-controls/ros2_controllers/issues/1298#issuecomment-2380303228) distinguishes changed trajectory derivatives from a suspected interpolator regression. Our reusable check fetches that published message and samples its supplied fields under the pinned implementation:

```sh
python scripts/validate_public_controller_case.py /tmp/public-controller-case --output public-controller-result.json
```

This optional developer check uses network access and PyYAML to read the issue's fenced native message. The MCP server exposes neither the fetch nor a YAML/code executor. The script retains the exact response digest, update date, transformation and native JSON digest; raw issue content stays in the selected local directory.

The published shoulder waypoints span 0.904417–1.24934. On 1,301 selected times, the pinned rule produces a minimum sampled reference of −0.760678308886849, consistent with the approximately −0.76 excursion described by the reporter. We did not reproduce the original hardware run, historical build or entire motion-planning pipeline. A position-only derivative-removal comparison is a numerical counterfactual, not a universal recommended fix.

## Independent numerical reference

The source pin is ros2_controllers Jazzy `8ce934536c6404750fb8aafc0b62b7d365049143`; `trajectory.cpp` SHA-256 is `a979c1bca2527a912571f5e1ed29e5887d7bf92aa226f6918a125a462821537d`.

```sh
python scripts/fetch_controller_reference.py /tmp/controller-reference/trajectory.cpp
python scripts/validate_controller_reference.py --source /tmp/controller-reference/trajectory.cpp --check-implementation --output controller-reference.json
python scripts/validate_controller_reference.py --source /tmp/controller-reference/trajectory.cpp --cases /tmp/public-controller-case/reference_cases.json --check-implementation --output controller-public-reference.json
```

These optional developer gates need a C++17 compiler. They compile an unchanged upstream interpolation function with explicit integer-nanosecond time and numeric-message stand-ins. This is a polynomial reference, not ROS/controller execution. The built-in gate compares 61 supported interior cases, explicitly excluding 39 initial/terminal/outside-segment cases from that public-API comparison. The public case supplies 93 additional interior segment samples. Every supported position/velocity/acceleration/effort coordinate is compared at rel/abs tolerance 1e-10. See reference review.

## Support boundary

The native JSON reader retains joint names and aligns comparisons by name. `splines` chooses linear/cubic/quintic degree from endpoint fields; `none` forwards the next waypoint inside a segment. At/after the final point, native terminal values are retained and absent derivatives are filled with zeros. Header stamps are preserved; sample times are relative and are not mapped to a live clock.

Position-bearing goals only; derivative integration, missing pre-first-point state, continuous-joint wrapping, runtime controller configuration, acceptance, preemption, tolerance handling, PID and physical execution are outside this implementation. A waypoint-envelope excursion is a review hint and may be intentional. It is not a safety verdict or evidence that a controller actually sent or achieved the sampled value.
