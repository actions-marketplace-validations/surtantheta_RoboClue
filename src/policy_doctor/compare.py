"""JSON-pointer configuration comparisons; no claims of causal attribution."""

from .artifacts import read_json


def compare_configs(before: str, after: str) -> dict:
    left, right = read_json(before), read_json(after)
    return compare_values(left, right)


def compare_values(left: dict, right: dict) -> dict:
    """Compare explicit JSON-compatible objects; pointers address these objects."""
    changes = []

    def walk(a, b, pointer):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                child = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
                if key not in a:
                    changes.append({"pointer": child, "change": "added", "after": b[key]})
                elif key not in b:
                    changes.append({"pointer": child, "change": "removed", "before": a[key]})
                else:
                    walk(a[key], b[key], child)
        elif type(a) is not type(b) or a != b:
            changes.append({"pointer": pointer, "change": "modified", "before": a, "after": b})

    walk(left, right, "")
    return {"schema_version": 1, "equal": not changes, "changes": changes,
            "scope": "Configuration differences; causation requires a controlled replay."}
