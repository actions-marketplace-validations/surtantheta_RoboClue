import copy

from policy_doctor import native_source_processing as source


def receipt(path, digest, role='native_metadata'):
    return {'root': 'model', 'path': path, 'role': role, 'sha256': digest * 64, 'size_bytes': 10}


def test_selected_source_identity_ignores_notice_changes_and_host_paths():
    selected = {'model': '/original/selected', 'receipts': [
        receipt('config.json', 'a'), receipt('policy_preprocessor.json', 'b'),
        receipt('README.md', 'c', 'local_notice')]}
    original = source._selected_identity(selected, [])
    relocated = copy.deepcopy(selected)
    relocated['model'] = '/colleague/new-path'
    relocated['receipts'][-1]['sha256'] = 'd' * 64
    assert source._selected_identity(relocated, []) == original


def test_selected_source_identity_binds_consumed_remote_tokenizer_bytes():
    selected = {'receipts': [receipt('config.json', 'a'), receipt('policy_preprocessor.json', 'b')]}
    tokens = [{'root': 'hf_cache', 'path': 'known/snapshot/tokenizer.json',
               'sha256': 'c' * 64, 'size_bytes': 12}]
    original = source._selected_identity(selected, tokens)
    changed = copy.deepcopy(tokens)
    changed[0]['sha256'] = 'd' * 64
    assert source._selected_identity(selected, changed) != original


def test_selected_source_identity_binds_declared_local_tokenizer_assets():
    selected = {'receipts': [receipt('config.json', 'a'),
                receipt('tokenizer/tokenizer.json', 'b', 'tokenizer_asset')]}
    original = source._selected_identity(selected, [])
    changed = copy.deepcopy(selected)
    changed['receipts'][1]['sha256'] = 'c' * 64
    assert source._selected_identity(changed, []) != original
