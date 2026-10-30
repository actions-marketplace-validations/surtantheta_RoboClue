"""Retention refuses source growth before writing beyond the accepted byte count."""
import os

import pytest

from policy_doctor import native_source_processing as api
from policy_doctor.artifacts import InputError


def test_legacy_source_copy_stops_growth_before_writing_excess(tmp_path, monkeypatch):
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.mkdir()
    destination.mkdir()
    path = source / 'payload.bin'
    path.write_bytes(b'a' * (2 * 1024**2))
    record = api._record(source, 'payload.bin')
    original_open = api._open_file
    original_read = os.read
    opened, changed = False, False
    def mark_open(*args):
        nonlocal opened
        result = original_open(*args)
        opened = True
        return result
    def growing(fd, count):
        nonlocal changed
        block = original_read(fd, count)
        if opened and not changed:
            changed = True
            with path.open('ab') as stream:
                stream.write(b'b' * (2 * 1024**2))
        return block
    monkeypatch.setattr(api, '_open_file', mark_open)
    monkeypatch.setattr(os, 'read', growing)
    with pytest.raises(InputError):
        api._copy(source, 'payload.bin', destination, 'payload.bin', record)
    assert changed
    assert (destination / 'payload.bin').stat().st_size <= record['size_bytes']
