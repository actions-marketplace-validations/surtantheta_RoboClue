# Public command snippets: ros2_controllers issue669

Source: https://github.com/ros-controls/ros2_controllers/issues/669 . These native JSON payloads preserve the author-posted command fields, removing the shell/action envelope only. Provenance and byte hashes are in provenance.json. This is public source data, not generated waypoint values or recorded robot feedback.

```sh
policy-doctor export-trajectory-case examples/public_controller_669/message_a.json examples/public_controller_669/message_b.json ./public-669-case --times 0 0.25 0.5 0.75 1
policy-doctor reproduce-case ./public-669-case
```

A specifies a two-point1s schedule; B specifies one target at0. Different sampled references do not establish the historical physical cause, a verified repair or real-time execution. The frozen historical blind engineering adjudication is documented in docs/HELDOUT_PUBLIC_ADJUDICATION_20261018.md. Re-running this now is an unblinded regression, not a new held-out result.
