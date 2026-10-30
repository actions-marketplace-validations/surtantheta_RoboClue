# Security and data handling

RoboClue's eight default MCP tools read explicitly authorized artifacts and do
not execute arbitrary training code or dispatch robot commands. Configure narrow
read roots. Public Hub reads require explicit operator and request opt-ins.

Local reports can contain artifact paths, configuration and selected values.
Review them before sharing. File hashes bind observed bytes; they are not
signatures or historical provenance authentication. Optional advanced case/media
tools have separate operator permissions; review their advertised schemas.

If reporting a vulnerability, avoid posting credentials, private datasets or an
exploit against a third party. After this repository is published, use GitHub's
private vulnerability reporting when enabled. If unavailable, open a minimal
issue asking the maintainer for a private reporting route without including the
sensitive details. No external reporting service is configured by this file.
