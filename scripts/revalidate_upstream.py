"""Observe upstream source drift without certifying untested framework versions."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

MAX_RESPONSE = 2 * 1024**2


class _GitHubRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        target = urllib.parse.urlparse(newurl)
        if target.scheme != "https" or target.hostname != "api.github.com":
            raise ValueError("Upstream redirect left the authorized GitHub API host")
        return super().redirect_request(request, response, code, message, headers, newurl)


def fetch_json(url):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "RoboClue-source-revalidation"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.build_opener(_GitHubRedirect()).open(request, timeout=20) as response:
        if urllib.parse.urlparse(response.url).hostname != "api.github.com":
            raise ValueError("Upstream observation left api.github.com")
        raw = response.read(MAX_RESPONSE + 1)
    if len(raw) > MAX_RESPONSE:
        raise ValueError("GitHub response exceeds the declared byte limit")
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError("Expected an upstream JSON object")
    return value


def observe(profile):
    repository = profile["repository"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid catalog repository")
    pins = profile["tested_commits"]
    if not pins or any(not re.fullmatch(r"[0-9a-f]{40}", p) for p in pins):
        raise ValueError("Catalog source pins must be full commit identities")
    result = {"framework": profile["id"], "repository": repository, "recorded_tested_commits": pins,
              "rule_ids": profile.get("rule_ids", []), "qualification": "not_rerun",
              "scope": "Upstream source identity observation only; matching a recorded pin is not renewed native execution qualification."}
    try:
        head = fetch_json("https://api.github.com/repos/" + repository + "/commits/HEAD")
        revision = head.get("sha")
        if type(revision) is not str or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("The current upstream source identity was absent or malformed")
        result["observed_head"] = revision
        result["observation_url"] = "https://github.com/" + repository + "/commit/" + revision
        result["status"] = "matches_recorded_source_pin" if revision in pins else "source_changed_requires_revalidation"
    except (OSError, ValueError, urllib.error.URLError) as error:
        result["status"] = "unknown"
        result["reason"] = type(error).__name__ + ": upstream observation unavailable or invalid"
    try:
        release = fetch_json("https://api.github.com/repos/" + repository + "/releases/latest")
        tag = release.get("tag_name")
        result["latest_release_tag"] = tag[:200] if type(tag) is str else None
        result["release_scope"] = "Release label observed only; its target commit and native compatibility were not qualified."
    except urllib.error.HTTPError as error:
        result["latest_release_tag"] = None
        result["release_scope"] = "No latest release endpoint" if error.code == 404 else "Release observation unavailable"
    except (OSError, ValueError, urllib.error.URLError):
        result["latest_release_tag"] = None
        result["release_scope"] = "Release observation unavailable"
    return result


def _destination(value):
    from policy_doctor.product_cli import _output_destination

    return _output_destination(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Explicitly allow bounded GitHub API observations")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        destination = _destination(args.output)
        if not args.fetch:
            raise ValueError("Use --fetch to allow upstream source observations")
        from policy_doctor.framework_versions import supported_framework_profiles

        profiles = supported_framework_profiles()
        results = [observe(profile) for profile in profiles["frameworks"]]
        clear = bool(results) and all(item["status"] == "matches_recorded_source_pin" for item in results)
        result = {"schema_version": 1, "kind": "upstream_revalidation_observation", "status": "recorded_pins_observed" if clear else "revalidation_required",
                  "frameworks": results, "native_framework_qualification": "not_rerun",
                  "scope": "Detects observed source drift and queues human review; portable regressions, source pins and version metadata do not qualify a new framework release."}
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError) as error:
        print("Revalidation input: " + " ".join(str(error).split()), file=sys.stderr)
        return 2
    for item in results:
        print(item["framework"] + ": " + item["status"] + "; native qualification not rerun")
    print("Review and re-run the affected framework's source/adaptor/native qualification before extending support.")
    return 0 if clear else 3


if __name__ == "__main__":
    raise SystemExit(main())
