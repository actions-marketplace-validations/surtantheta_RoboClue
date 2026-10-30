# A plausible batch, wrong targets, and a verified correction

[Watch the 42-second demonstration](../assets/roboclue-native-demo.mp4).

The recorded native GR00T N1.7 experiment used NVIDIA's public Arena simulation
data. Changing the known intended preprocessing from `use_relative_action=True`
to `False` produced a plausible batch that the native trainer accepted for two
optimizer steps. The guard refused that same fault before any model forward or
optimizer step. Restoring the intended setting passed the original pre-fault
criterion and completed two steps. The fault changed 1,400 target values and two
mask dtypes. This is a controlled fault, not a naturally occurring incident or
physical robot experiment.

The later terminal sequence is actual output from the saved-capture CPU replay:

```sh
python -m roboclue advanced demo-native --output ./native-demo
```

It checks healthy, faulty, corrected and uncovered captures. It performs no new
training and does not recompute the correction from scratch: it verifies the
retained corrected capture. The installed replay binds its own runtime criterion
to the frozen intended reference before checking the fault; that criterion is
distinct from the historical native training criterion. Do not recreate a
reference from a suspect batch just to obtain a pass.

The [recorded native qualification](../assets/native-qualification.json) contains
the original counts, criterion and source hashes. The packaged
[provenance](../src/policy_doctor/data/native_training_demo/provenance.json),
[attribution](../src/policy_doctor/data/native_training_demo/NOTICE.txt) and
[CC-BY-4.0 license](../src/policy_doctor/data/native_training_demo/LICENSE-CC-BY-4.0.txt)
identify the public source and controlled intervention.
[Replay details](NATIVE_REPLAY.md) explain what the installed command establishes.
