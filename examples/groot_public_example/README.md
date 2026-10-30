# Try the GR00T input-contract workflow on original public data

This CPU example asks which state values and camera inputs actually reach a native GR00T adapter. It uses one original IPEC LIBERO episode and two original checkpoint metadata files, with pinned candidate and reviewed corrected upstream harness source. It runs no policy inference, simulator, robot or GPU workload, and downloads no model weights.

The preparation command fetches exact original bytes, verifies every download's size and SHA-256, copies a genuine generated version file from the original published harness wheel, and creates an isolated native runtime. It contains no diagnosis, correction or incident-packing logic. Policy Doctor performs those operations through its existing CLI or MCP tools.

## Prepare a new local folder

Prerequisites: a checkout containing this recipe, Python 3, an installed `uv` executable, network access to GitHub/Hugging Face/PyPI, and a local wheel build. Linux x86_64 with a Python 3.12 runtime is the qualification target; other platforms are unqualified.

From the checkout root, build the local wheel, then prepare the example. Building the wheel and installing its dependencies are setup work:

```sh
uv build --wheel --out-dir ./public-example-wheel
python scripts/prepare_groot_public_example.py "$HOME/groot-public-demo" \
  --uv "$(command -v uv)" \
  --wheel ./public-example-wheel/*.whl
cd "$HOME/groot-public-demo"
```

Use a new wheel directory containing exactly the wheel just built. This is an optional original-artifact workflow with its own pinned dependencies and preparation cost; it is separate from the CPU-only synthetic quick start and the native training proof.

The destination must be new. A failed attempt leaves a receipt and completed files for review; the command refuses to overwrite or silently reuse it. Use another new destination after investigating a failure. There are no automatic download retries. The explicit `uv` command installs dependencies into new environments; it may reuse its normal package/Python caches. `preparation-receipt.json` records the original download bytes, timings, package versions, wheel identity and installation logs. The pinned original downloads total 1,841,337 bytes; Python and dependency installation are additional setup costs.

If the product is already installed, omit `--wheel`. You may replace `--uv` with `--python /absolute/path/to/native-env/bin/python` when you already have the seven exact dependency versions in `manifest.json`; that interpreter is inspected but its environment is not modified. The normal path creates fresh environments with no Torch dependency.

Review `native-runtime.json` before execution: those operator-selected source trees execute native code. `input-contract.json` is an explicit caller requirement about the eight-coordinate source state and ordered image/wrist associations. It does not prove physical calibration, orientation semantics, or historical training membership.

## Capture, investigate, retain and reuse

With the optional product environment installed:

```sh
./product-env/bin/policy-doctor probe-native groot_adapter ./checkpoint ./dataset \
  --sample 0:0 --native-runtime ./native-runtime.json \
  --requirements ./input-contract.json --case ./observation-0 --json
./product-env/bin/policy-doctor investigate ./observation-0/candidate \
  --reference ./observation-0/corrected --markdown
./product-env/bin/policy-doctor create-case native-observation-source \
  ./observation-0/candidate ./input-case --corrected ./observation-0/corrected
./product-env/bin/policy-doctor reproduce-case ./input-case
./product-env/bin/policy-doctor probe-native groot_adapter ./checkpoint ./dataset \
  --sample 0:55 --native-runtime ./native-runtime.json \
  --requirements ./input-contract.json --case ./observation-55 --json
./product-env/bin/policy-doctor check-case ./input-case ./observation-55/candidate
./product-env/bin/policy-doctor check-case ./input-case ./observation-55/corrected
```

Use returned candidate/corrected paths when integrating with a client. For this pinned job, the candidate reports a completed check with a source contradiction; the correction passes. The same declared rule applies to frame 55. Exit 1 is expected for an error finding, exit 3 means insufficient evidence, and exit 2 means invalid input. An error finding does not mean the native call failed. The adapter's policy loader is explicitly substituted with a capture sink returning generated zero actions; there is no policy competence claim.

Move a retained case or observation, then repeat the checks:

```sh
mv ./input-case ./moved-input-case
./product-env/bin/policy-doctor reproduce-case ./moved-input-case
mv ./observation-0 ./moved-observation-0
./product-env/bin/policy-doctor reproduce-native-probe \
  ./moved-observation-0/corrected --native-runtime ./native-runtime.json --json
```

Recorded case reproduction checks retained evidence. Native probe reproduction executes the configured native reference route again using retained decoded arrays; it does not re-decode the original media. Source/runtime/verifier drift yields unknown. Keep the prepared runtime environments and operator source roots fixed. Python virtual environments and their launcher shebangs are not promised to be relocatable; if you move the whole setup, recreate its environments and configuration at the new location. The relocation claim above applies to retained evidence, not runtime installation. No artifact chooses executable source paths.

For an MCP client, configure this existing stdio server command (run from the prepared folder):

```sh
mkdir ./mcp-cases
./product-env/bin/python -I -m policy_doctor serve \
  --root "$PWD" --case-root "$PWD/mcp-cases" \
  --native-runtime "$PWD/native-runtime.json"
```

Use absolute paths in your client's server configuration. Use `probe_native_processing`, `investigate_run_directory`, `create_incident_case`, `check_incident_candidate`, and `reproduce_native_processing`; the installed tool catalog gives argument schemas. See the feature guide.

## Provenance and limits

`manifest.json` pins the exact checkpoint revision, dataset revision, harness revisions, GR00T revision, original file URLs, sizes and hashes. The reference tree contains unchanged candidate modules plus the original corrected `groot.py`; it is a qualified minimal import closure, not a full checkout or full benchmark environment. Downloaded source licenses and notices are retained. The helper emits only local preparation metadata and configuration, without modifying native code.

The pinned IPEC dataset publication supplies no license/card grant. This example downloads original bytes directly for local inspection; it ships no dataset payload and does not establish redistribution permission. Do not treat the checkpoint's model card or a code license as a dataset license. Selected whole files include all 110 episode observations even though this example selects frames 0 and 55.

The workflow checks conditional API input preservation. It does not establish physical camera identity, object-source flow from equal images, calibration, training membership, raw quaternion conventions, task success, full native vision processing or production robot safety. Preparing this recipe and selecting its sources are product development and setup work, not measured practitioner time savings or an independent blind trial.
