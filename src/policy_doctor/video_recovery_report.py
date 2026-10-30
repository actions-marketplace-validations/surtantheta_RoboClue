"""Small human view of the complete, separately retained recovery report."""
import json
import shlex


def concise_video_recovery(result: dict) -> str:
    kind = result.get('kind', '')
    label = ('Replay' if kind.endswith('_reproduction') else
             'Later episode check' if kind.endswith('_candidate') else 'Video reference recovery')
    lines = [f"{label}: {result.get('status', 'unknown')}",
             f"Original interval: {result.get('original_status', 'unknown')}",
             'Proposed association verified: ' + ('yes' if result.get('repair_verified') is True else 'no')]
    if 'episode_index' in result:
        lines.append('Episode: ' + json.dumps(result['episode_index']))
    if 'candidate_episode_index' in result:
        lines.append('Later episode: ' + json.dumps(result['candidate_episode_index']))
    if 'distinct_episode' in result:
        lines.append('Distinct episode: ' + json.dumps(result['distinct_episode']))
    for patch in result.get('patch', [])[:6]:
        lines.append('Metadata proposal: row ' + json.dumps(patch.get('row')) + ', ' +
                     json.dumps(patch.get('column')) + ': ' + json.dumps(patch.get('old')) +
                     ' -> ' + json.dumps(patch.get('new')))
    if len(result.get('patch', [])) > 6 or result.get('patch_truncated'):
        lines.append(f"... {result.get('patch_cells', len(result['patch']))} metadata cells in all; --json lists up to 16.")
    derived = result.get('derived_metadata')
    if isinstance(derived, dict):
        lines.append('Derived metadata copy: ' + json.dumps(derived.get('dataset_path')) + ' -> '
                     + json.dumps(derived.get('path')) + ' (sha256 ' + str(derived.get('sha256')) + ')')
    if result.get('overlay_scope'):
        lines.append(str(result['overlay_scope']))
    validation = result.get('validation_scope')
    if isinstance(validation, dict):
        lines.append('Replay: ' + str(validation.get('replay', '')))
        lines.append('Native consumer validation: ' + str(validation.get('native_consumer', '')))
    if result.get('reason'):
        lines.append('Reason: ' + json.dumps(result['reason']))
    for unknown in result.get('unknowns', [])[:6]:
        lines.append('Unknown: ' + json.dumps(unknown))
    directory = result.get('case_directory')
    if directory:
        lines.append('Offline case: ' + json.dumps(directory))
        lines.append('Replay: ' + shlex.join(['policy-doctor', 'reproduce-case', str(directory)]))
    lines.append('Source metadata is unchanged. Verification covers the selected media association; full-dataset and policy behavior remain unassessed.')
    return '\n'.join(lines) + '\n'
