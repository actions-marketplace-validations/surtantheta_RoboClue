# First-use acceptance

The public workflow should let a researcher decide what was checked, what is
wrong or missing, and what to do next without interpreting a raw report.

This is a technical acceptance plan. Agent simulations and automated tests are
not evidence that a real researcher completed the workflow, adopted RoboClue,
or would recommend it.

| Question | Required behavior |
| --- | --- |
| Which command do I use? | The README begins with `roboclue`; the main help presents a small set of verbs. Specialized commands remain available through `roboclue advanced`. Existing scripts retain compatibility. |
| Which agent tool do I use? | The canonical MCP server exposes eight read-only question tools, including dataset/configuration, simulator and recovery checks. The remaining specialist catalog requires `--advanced`; existing authorization behavior stays intact. |
| What happened? | A normal assessment prints a short verdict, a concrete problem or missing evidence, and the next command. Raw JSON is opt-in. |
| Did missing evidence mean my dataset is broken? | The result distinguishes observed contradictions from incomplete coverage. A partial assessment never silently becomes a passing certificate. The JSON report and exit status retain their diagnostic meaning. |
| Is this actually a supported artifact? | Recognition uses guarded content and supported layout, rather than a filename alone. Unsupported checkpoints and arbitrary configurations are not reported as compatible or healthy. |
| Will a wrong path or unusual file hang it? | Missing paths, symlinks, FIFOs, invalid UTF-8 and malformed structured input produce an input error without a traceback or blocking read. |
| Will the first command download data? | Network access requires an explicit fetch option. Detection does not implicitly open remote resources or execute a training/configuration file. |
| Can I see the advertised loop quickly? | The CPU demo detects a real controlled fault, corrects a separate candidate, and checks the correction against the same frozen criterion. It reports its scope and elapsed time. |
| Can I preserve intent for my own trainer? | Guard setup explains or captures a prospective reviewed reference, retains its independent pin, and gives an attachment recipe. It does not manufacture historical intent from a suspect run. |
| Did an existing integration change? | Existing CLI, raw reports, JSON schemas, numerical checks and MCP authorization tests continue to pass. The public presentation layer cannot turn an unknown into raw pass. |

`tests/test_product_onboarding.py` exercises the public entry point and refusal
boundaries, including a real MCP stdio connection, its tool catalog, a local
dataset assessment and an outside-root refusal. The tests retain raw unknown
results, replay the printed next command, and compare healthy, faulty and
uncovered native captures against one unchanged reference and pin.

Existing engine and installed package suites remain necessary: a pleasant
terminal result alone cannot qualify the underlying diagnostic or integration.

Human testing remains outside this technical acceptance. Platform and framework
support are limited to the separately documented, actually tested profiles.
