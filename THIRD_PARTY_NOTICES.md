# Third-party notices

The small target/mask fixtures under `src/policy_doctor/data/native_training_demo/`
are explicitly documented projections of genuine saved GR00T native inputs
derived from NVIDIA's public Arena G1 simulation dataset, under CC-BY-4.0.
The exact pinned dataset licence, attribution, original/projected hashes and
modifications are retained in that directory's `LICENSE-CC-BY-4.0.txt`, `NOTICE.txt`,
`provenance.json` and manifest. These fixtures contain no model weights or images.
The replay is a saved-input demonstration, not a fresh native model execution.

The ten unmodified source/configuration files retained as inert data under
`tests/fixtures/arena43/` come from isaac-sim/IsaacLab-Arena revision
`8b4a3a47fc53de23e8205089d71109a2e2348acd`. Their exact original names, URLs,
hashes and individual license evidence are recorded in `UPSTREAM.json` there.
Original copyright/SPDX notices are retained. Eight files explicitly carry
Apache-2.0 headers; the JSON metadata files have no inline license header.
The fixture's standard Apache-2.0 license text comes from this project's
`LICENSE-APACHE-2.0`; it is not represented as a fetched Arena root license.

The source-derived configuration constants in `conversion_profile_data.py`
and joint mapping in `conversion_numeric.py` retain the Apache-2.0 terms of
the cited source. The original comparison, intake and reporting implementation
is under MIT. These modules inspect a pinned conversion recipe as data; they
do not bundle an Arena runtime, dataset, model or robot controller.

The original OpenPI source and lockfile text fixtures under
`tests/fixtures/openpi_source_profile_215abfb/` come from
Physical-Intelligence/openpi revision
`215abfb217dbac7d5f1273282331b9b1866c0479`. They retain their original
headers and the exact upstream Apache-2.0 license in that directory.
Tests materialize their original names temporarily to inspect source as data;
they do not import an OpenPI runtime or include policy parameters. The
application's source-profile manifest records the nine original file hashes.
The resolver and statistics intake are original project code. Native parser,
normalization execution and whole serving behavior have separate scopes.

`src/policy_doctor/trajectory.py` adapts polynomial coefficient equations from `joint_trajectory_controller/src/trajectory.cpp` in ros-controls/ros2_controllers, revision `8ce934536c6404750fb8aafc0b62b7d365049143`.

Copyright 2017 Open Source Robotics Foundation, Inc.

That source is licensed under the Apache License, Version 2.0. The adapted trajectory module is provided under Apache-2.0; the remaining original project code is under MIT unless otherwise stated. See `LICENSE-APACHE-2.0` and `LICENSE`.

Local changes implement a Python position-bearing interpolator, bounded native JSON intake, explicit unsupported cases and uncertainty, evidence/report structures, and named-joint comparison. This is not the upstream ROS controller.

The optional developer reference script compiles an unchanged function body from a separately supplied, hash-verified upstream source. Its explicit time/message stand-ins are validation scaffolding and do not constitute a ROS runtime. The upstream source is not bundled in the application.

The original, unmodified IsaacLab source files stored as text under
`tests/fixtures/isaac_layout_sources/` are from revision
`32a1f330511130ae75ffeda96304c1bf569b5acb`. They retain their original headers
and are licensed under BSD-3-Clause. Copyright (c) 2022–2026, The Isaac Lab
Project Developers. The exact upstream license is retained in that fixture
directory and in `LICENSE-ISAACLAB-BSD-3-CLAUSE`.

These files are offline provenance/reference fixtures, not an included IsaacLab
runtime. The layout adapter checks the pinned native raw-term transformation
and semantic source identities. It does not reproduce a simulator, sensor
backend or policy execution. Original project code remains MIT unless an
individual module or this notice states otherwise. No endorsement by the
Isaac Lab Project Developers is implied.
