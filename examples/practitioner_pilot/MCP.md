# Connect an MCP client

First install the wheel's `mcp` extra as shown in README.md. Configure your client with absolute paths:

```json
{
  "mcpServers": {
    "roboclue-preview": {
      "command": "/ABS/PREVIEW/.venv/bin/policy-doctor-mcp",
      "args": ["--root", "/ABS/YOUR/ARTIFACTS"]
    }
  }
}
```

The root must exist. Repeat `--root` for another read directory. Default tools inspect local artifacts; they do not write cases, execute native runtimes or fetch Hub data. A client can still send report contents to its own model provider: use your normal data-sharing rules.

Try: “Investigate `/ABS/YOUR/ARTIFACTS`. Show supported findings, source hashes and pointers. Explain what is unresolved. Do not guess which pair to compare if several exist.”

For a trajectory, ask the assistant to use `sample_ros_trajectory` on the explicit file and sample times, then explain the narrow result. The CLI in README.md can export the portable case without granting the MCP server write access. A moved case must be under a configured read root before `reproduce_trajectory_case` can inspect it.

For public Hub checks, the operator must explicitly add `--allow-hub` to the server arguments. This enables one additional tool, `check_policy_metadata`. Ask for `scope="metadata"`, `response_format="concise"`, a supported public repository pair and source evidence. The concise report marks omitted findings and provides `detailed_request`; following it makes a new read at resolved commits, not a cached response. Detailed remains the default. All local inputs still require configured roots.

Record your MCP client/version, model/provider and any assistant mistakes separately from verifier findings. A successful protocol test is not evidence that every assistant will use the tools correctly.
