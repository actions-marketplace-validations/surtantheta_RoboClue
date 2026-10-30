# A target error you can catch and correct in seconds

From an installed copy, run:

```sh
roboclue demo
```

This offline CPU example creates a new `roboclue-demo` directory. It uses the
same target verifier as the training guard. Its three source rows are generated
for onboarding; they are not a reported incident or a native trainer run.

The declared recipe requires changes in joint position: `next_position - position`.
The broken exporter writes absolute next positions instead. All target shapes
and dtypes are valid, so shape checking alone would accept the batch.

RoboClue retains the intended reference and its pin before checking the broken
candidate. It finds three incorrect values. The demo applies the declared
subtraction to the unchanged source, writes a separate corrected copy and checks
it against the original pin. The source and broken candidate remain unchanged.
A separate uncovered-source control stays UNKNOWN.

The default output has three lines. To see the exact replay command and scope:

```sh
roboclue demo --output ./another-demo --explain
```

Use a new output directory each time; the demo refuses to overwrite old evidence.
`--json` prints the full summary instead. Each run saves the reference, pin,
source, both candidates and the raw guard reports under the chosen directory.

The example checks its three declared target/mask rows. It does not establish
your intended recipe, training quality or physical robot behavior. It does not
automatically repair arbitrary datasets. For a prospective reference of your
own, use `roboclue guard init` and the [guided setup](GUARD_INIT.md).

The [saved native replay](NATIVE_REPLAY.md) is a separate attributed example from
public NVIDIA simulation-derived data, with its own immutable source evidence.
