"""Proven container damage fails consistently without inventing episode loss."""
import json

import pytest
from test_compound_metadata_recovery import make_variant
from test_salvage import hashes

from policy_doctor import product_cli, salvage, windows
from policy_doctor.salvage import check_salvage
from policy_doctor.windows import audit_episode_windows


@pytest.mark.parametrize('damage', ['missing_tail', 'bad_tail', 'header', 'empty', 'zero_length', 'outside_length'])
@pytest.mark.parametrize('headers', [False, True])
def test_bad_middle_envelope_is_error_in_preflight_and_salvage_without_blame_or_writes(tmp_path, capsys, damage, headers):
    root = make_variant(tmp_path, corrected=True, videos=False)
    path = root / 'data/chunk-000/file-001.parquet'
    original = path.read_bytes()
    variants = {'missing_tail': original[:-4], 'bad_tail': original[:-4] + b'FAIL',
                'header': b'not a native parquet file', 'empty': b'',
                'zero_length': original[:-8] + bytes(4) + b'PAR1',
                'outside_length': original[:-8] + (len(original) + 10).to_bytes(4, 'little') + b'PAR1'}
    path.write_bytes(variants[damage])
    before = hashes(root)
    window_report = audit_episode_windows(root, video_headers=headers)
    assert window_report['status'] == 'error'
    finding = next(item for item in window_report['findings'] if item['code'] == 'episode.parquet_envelope')
    assert finding['evidence'][0]['path'] == str(path)
    assert finding['evidence'][0]['sha256'] == before['data/chunk-000/file-001.parquet']
    assert window_report['metadata_patches'] == []
    salvage = check_salvage(root, video_headers=headers)
    assert salvage['status'] == 'error' and salvage['plan']['strategy'] == 'unknown'
    assert salvage['episodes']['drop'] == []
    assert salvage.get('metadata_row_conflicts', {}).get('physical_span_conflicts', 0) == 0
    for extra in ([], ['--question', 'recovery']):
        assert product_cli.main(['check', str(root), *extra, '--json', '--fail-on', 'error',
                                 *([] if headers else ['--no-video-headers'])]) == 1
        assert json.loads(capsys.readouterr().out)['status'] == 'error'
    assert hashes(root) == before


def test_possible_but_over_budget_footer_stays_unknown_before_arrow_reads(tmp_path, monkeypatch):
    path = tmp_path / 'file.parquet'
    path.write_bytes(b'PAR1' + bytes(128) + (128).to_bytes(4, 'little') + b'PAR1')
    monkeypatch.setattr(windows, 'MAX_FOOTER_BYTES', 64)
    import pyarrow.parquet as pq
    monkeypatch.setattr(pq, 'ParquetFile', lambda *a, **k: pytest.fail('Budget refusal must precede Arrow allocation'))
    intake = windows._Intake(tmp_path, max_rows=100, max_files=8)
    with pytest.raises(windows._Unknown, match='budget exceeded') as error:
        intake.parquet(path, ['episode_index'], 'dataset_parquet')
    assert not isinstance(error.value, windows._ParquetEnvelopeConflict)


def test_encrypted_magic_is_unassessed_instead_of_claimed_damage(tmp_path, monkeypatch):
    # This checks the refusal route only; real encrypted files are an independent
    # production control, not claimed to be represented by this inert envelope.
    path = tmp_path / 'file.parquet'
    path.write_bytes(b'PARE' + bytes(128) + (128).to_bytes(4, 'little') + b'PARE')
    import pyarrow.parquet as pq
    monkeypatch.setattr(pq, 'ParquetFile', lambda *a, **k: pytest.fail('No encryption key or parsing is authorized'))
    with pytest.raises(windows._Unknown, match='Encrypted') as error:
        windows._Intake(tmp_path, 100, 8).parquet(path, ['episode_index'], 'dataset_parquet')
    assert not isinstance(error.value, windows._ParquetEnvelopeConflict)


@pytest.mark.parametrize('unsupported', ['encrypted', 'budget'])
def test_salvage_refuses_unsupported_footer_without_claiming_corruption_or_loss(tmp_path, monkeypatch, unsupported):
    root = make_variant(tmp_path, corrected=True, videos=False)
    path = root / 'data/chunk-000/file-001.parquet'
    if unsupported == 'encrypted':
        original = path.read_bytes()
        path.write_bytes(b'PARE' + original[4:-4] + b'PARE')
    else:
        monkeypatch.setattr(salvage, 'MAX_FOOTER_BYTES', 1)
    before = hashes(root)
    result = check_salvage(root, video_headers=False)
    assert result['status'] == 'unknown'
    assert result['plan']['strategy'] == 'unknown'
    assert not result['episodes'].get('drop')
    assert not any(item['status'] == 'error' for item in result['findings'])
    assert hashes(root) == before
