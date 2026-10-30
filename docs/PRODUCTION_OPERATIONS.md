# Operating RoboClue

Install a qualified wheel into a dedicated environment and retain its artifact
manifest, qualification report and dependency snapshot with the deployment.
Use the interpreter in that environment for the generated MCP configuration.
The launcher uses Python isolated mode so local modules and PYTHONPATH cannot
replace the installed server. A source checkout alone does not install it.

The qualified profiles are Linux x86-64 with Python3.11 at the declared direct
dependency floors and Python3.12 with the recorded current dependencies. The
release report names exact versions and source revisions. These profiles do not
promise every future dependency or robotics framework release works. Core
installation supports metadata/config diagnostics; install the dataset,
recovery and MCP extras for those workflows.

Use the final packet's portable `full-dependencies.txt` for its installed
profile. Historical files under `constraints/release-py312-linux.txt` and raw
`pip freeze` snapshots are not the current production installation recipe;
snapshots can contain a build-machine URL for RoboClue itself. Install the
packet's local wheel separately and run `python -m pip check`.

For CI that must reject incomplete evidence, use:

```sh
roboclue check /absolute/path/to/input --json --fail-on unknown
```

Keep stdout as the machine-readable result and retain the exit status. Exit0
means the selected gate did not fail; it is not a claim of general compatibility.
`--fail-on error` intentionally permits WARNING and many UNKNOWN results. Use
`--fail-on warning` when published-recipe differences should also block a run.
Input refusals exit2; an incomplete explicitly requested assessment may exit3.
A downstream reader closing stdout cancels the CLI with exit141. Never accept
an incomplete JSON stream as a successful assessment.

MCP reads are restricted to configured read roots. Generate configuration with
`roboclue mcp setup --help`, select only required roots and retain the resulting
configuration. Commands and source passed for inspection are parsed, never
executed. A guard or demonstration explicitly launched by the operator is a
separate execution workflow. Hub requests require explicit network permission;
without that permission or available linked evidence, instruction coverage
remains UNKNOWN. Pass the actual `launch_command` MCP parameter when inspecting
a command. See [MCP](MCP.md) and [instruction coverage](LANGUAGE_VOCABULARY.md).

The register compares declared observations. Lockfiles do not attest the live
environment or downloaded wheel/source identity. A selected direct URL, marker,
editable dependency, conflicting pin or exceeded budget remains unresolved.
Only the packaged, reviewed catalogue is loaded. Exact checkpoint identity,
complete stack selection and typed resolved configuration are required for
publisher-recipe comparisons. See [coverage](REGISTER_CATALOGUE_EVIDENCE.md).

For damaged recordings, preserve the original directory and run preflight and
salvage against those bytes. A proven malformed Parquet envelope is an ERROR;
unsupported encryption and a structurally possible footer above the reader
budget remain UNKNOWN. Neither an unreadable file nor a missing predecessor
proves a readable later episode is wrong. A partial reading cannot authorize
keep/drop decisions or an automatic repair. Review the specific plan and copy
only into a new destination when an explicitly supported plan is available.

Plaintext PAR1 footers can also contain encrypted columns. A bounded
Compact-Thrift inspection rejects their encryption fields before native Arrow
column-metadata access. Unsupported or ambiguous metadata stays unassessed;
opaque user-metadata bytes are not interpreted as encryption markers. This
check does not attempt decryption or certify arbitrary Parquet extensions.

For native guards, retain the original criterion before faults, corrections
and replay. An accepted startup witness does not measure policy competence or
physical task success. Follow the source/profile restrictions in
[guard setup](GUARD_INIT.md) and [scope](SCOPE.md).

If an upgrade changes a qualified result or cannot inspect a required input,
stop the dependent training/deployment step and save the report, command,
artifact hashes and dependency snapshot. Reproduce in the retained environment.
Rollback by selecting the previous complete environment and its matching wheel
and criterion; do not regenerate a criterion to make a changed pipeline pass.
The diagnostic does not edit the source dataset. Run a new qualified assessment
after any deliberate data, source, dependency or criterion change.

The production qualification uses pinned public artifacts and actual native
software workflows. It does not stand in for independent practitioner feedback,
a physical robot trial, or full coverage of every proposed catalogue item.
