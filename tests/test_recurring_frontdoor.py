"""Exercise the unified front door against real offline recurring engines."""
import json
from pathlib import Path

from policy_doctor.big_bets_frontdoor import main


def test_monitor_without_fetch_publishes_current_unknown_through_unified_cli(tmp_path, capsys):
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps({'schema_version': 1, 'datasets': [
        {'id': 'public-data', 'source': 'hf://lerobot/berkeley_autolab_ur5', 'consumer': 'offline-test'}]}))
    state = tmp_path / 'state'
    code = main(['monitor', 'scan-once', '--inventory', str(inventory), '--state', str(state)])
    output = capsys.readouterr()
    assert code == 3
    report = json.loads((state / 'state.json').read_text())['current']
    assert report['status'] == 'unknown'
    assert (state / 'index.html').is_file()
    assert 'unknown' in output.out.lower()


def test_offline_upgrade_watch_writes_local_unknown_alert_through_unified_cli(tmp_path, capsys):
    recipe = tmp_path / 'recipe.json'
    recipe.write_text('{"isaaclab_version":"2.3.2"}')
    config = tmp_path / 'watch.json'
    config.write_text(json.dumps({'schema_version': 1, 'kind': 'owned_upgrade_watch',
        'watches': [{'watch_id': 'isaac', 'repository': 'isaac-sim/IsaacLab', 'kind': 'commit',
                     'ref': 'e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4'}],
        'assets': [{'asset_id': 'my-recording', 'consumer': 'isaaclab-hdf5-xyzw', 'watch_id': 'isaac',
                    'files': {'recipe': str(recipe)},
                    'version': {'path': str(recipe), 'format': 'json', 'selector': '/isaaclab_version'}}]}))
    state = tmp_path / 'watch-state'
    code = main(['upgrade-watch', 'scan-once', '--config', str(config), '--state-dir', str(state), '--offline', '--json'])
    summary = json.loads(capsys.readouterr().out)
    assert code == 3 and summary['counts'] == {'affected': 0, 'unaffected': 0, 'unknown': 1}
    report = json.loads(Path(summary['report']).read_text())
    assert report['assets'][0]['code'] == 'offline_upstream_unobserved'
    assert report['previous_results_reused'] is False
    assert report['observations'][0]['receipts'] == []


def test_catalog_is_accessible_without_running_a_watch(capsys):
    assert main(['upgrade-watch', 'catalog']) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert 'isaaclab-hdf5-xyzw' in catalog
