"""Generated files through real reader and frontdoors; no public/native-run claim."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from test_conversion_fidelity_reader_proposal import Generated

from policy_doctor.cli import main
from policy_doctor.preflight import preflight


def corrupt_action(generated):
    for eid in generated.arrays:
        table = generated.table(eid)
        changed = pa.array(table['observation.state'].to_pylist(), type=pa.list_(pa.float32()))
        generated.write(eid, table.set_column(table.schema.get_field_index('action'), 'action', changed))


def test_real_reader_cli_preserves_error_witness_and_same_criterion_after_correction(tmp_path, capsys):
    generated = Generated(tmp_path)
    healthy = preflight(str(generated.export), **generated.kwargs())
    criterion = healthy['components']['conversion_fidelity']['criterion_sha256']
    assert healthy['components']['conversion_fidelity']['status'] == 'pass'
    corrupt_action(generated)
    args = ['preflight', str(generated.export), '--original-recording', str(generated.original),
            '--converter-source', str(generated.source), '--converter-config', str(generated.config_path),
            '--converter-directory', str(generated.source), '--json']
    assert main(args) == 1
    report = json.loads(capsys.readouterr().out)
    component = report['components']['conversion_fidelity']
    assert component['status'] == component['numeric_association_status'] == 'error'
    assert component['criterion_sha256'] == criterion
    assert component['coverage']['compared_numeric_rows'] == 12
    witness = next(f for f in report['findings'] if f['code'] == 'conversion.numeric_joint_association')
    assert 'source frame' in witness['message'] and 'joint' in witness['message']
    assert len(witness['evidence']) == 2 and all(len(e['sha256']) == 64 for e in witness['evidence'])
    generated.restore()
    corrected = preflight(str(generated.export), **generated.kwargs())['components']['conversion_fidelity']
    assert corrected['status'] == 'pass' and corrected['criterion_sha256'] == criterion
    assert corrected['native_consumption']['status'] == 'unknown'


def test_real_stdio_reader_keeps_concise_axes_and_rechecks_corrected_export(tmp_path):
    generated = Generated(tmp_path)
    source = str(Path(__file__).parents[1] / 'src')
    bootstrap = 'import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module("policy_doctor",run_name="__main__")'

    async def run():
        params = StdioServerParameters(command=sys.executable,
            args=['-I', '-B', '-c', bootstrap, source, 'serve', '--root', str(tmp_path)],
            cwd=str(tmp_path), env=dict(os.environ, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1'))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                schema = next(t for t in (await client.list_tools()).tools if t.name == 'preflight_check').inputSchema
                assert {'original_recording', 'converter_source', 'converter_config', 'converter_directory'} <= set(schema['properties'])
                arguments = {'source': str(generated.export), 'original_recording': str(generated.original),
                             'converter_source': str(generated.source), 'converter_config': str(generated.config_path),
                             'converter_directory': str(generated.source)}
                healthy = await client.call_tool('preflight_check', arguments)
                assert not healthy.isError
                component = healthy.structuredContent['components']['conversion_fidelity']
                criterion = component['criterion_sha256']
                assert component['numeric_association_status'] == 'pass'
                assert component['whole_numeric_source_to_export']
                assert component['native_consumption_status'] == 'unknown'
                corrupt_action(generated)
                changed = await client.call_tool('preflight_check', arguments)
                assert not changed.isError and changed.structuredContent['status'] == 'error'
                component = changed.structuredContent['components']['conversion_fidelity']
                assert component['numeric_association_status'] == 'error'
                assert component['criterion_sha256'] == criterion and component['compared_numeric_rows'] == 12
                assert not component['whole_numeric_source_to_export']
                assert len(json.dumps(changed.structuredContent).encode()) <= 8192
                detailed = await client.call_tool('preflight_check', dict(arguments, response_format='detailed'))
                assert not detailed.isError
                assert detailed.structuredContent['components']['conversion_fidelity']['criterion_sha256'] == criterion
                assert any(f['code'] == 'conversion.numeric_joint_association' for f in detailed.structuredContent['findings'])
                generated.restore()
                corrected = await client.call_tool('preflight_check', arguments)
                assert not corrected.isError
                component = corrected.structuredContent['components']['conversion_fidelity']
                assert component['numeric_association_status'] == 'pass' and component['criterion_sha256'] == criterion
                refused = await client.call_tool('preflight_check', dict(arguments, original_recording='/etc/passwd'))
                assert refused.isError

    asyncio.run(asyncio.wait_for(run(), timeout=60))
