"""Developer-only upstream execution + installed MCP integration, with explicit scope.

Trusted LeRobot code executes in a separate CPU worker. MCP only reads snapshots,
saved outputs and native recordings; it never executes artifact Python or a robot.
Requires existing pinned inputs and a local tokenizer; downloads nothing.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

LEROBOT_REVISION = '146d8474932c8b566717eb2738ba459337ab71c4'
CHECKPOINT_REVISION = 'a24f1f380461b665158d52acd140152c79e8b36d'
CHECKPOINT_FILES = {
    'config.json': '3ed3ed80ee077d94ff22942130642bb60e255443ee65da676bc88ab1684a7efe',
    'policy_preprocessor.json': '7beef33479fc22c9a45670154b3c7b909fb6d51dac3cd6db6f23376638b4ac11',
    'policy_preprocessor_step_5_normalizer_processor.safetensors': 'b0cdde6e8a6f49a8e19eefb376728e47c09d3b3cc20ce3a97c45619fe7a732d9',
    'policy_postprocessor.json': '3b51f092c70c710ce0213ee1b63bf51b4878ec67828dd2d67daf9ef51081a41a',
    'policy_postprocessor_step_0_unnormalizer_processor.safetensors': 'b0cdde6e8a6f49a8e19eefb376728e47c09d3b3cc20ce3a97c45619fe7a732d9'}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def package_identity():
    import policy_doctor

    package = Path(policy_doctor.__file__).parent
    return {'version': policy_doctor.__version__, 'package': str(package),
            'module_sha256': {p.name: digest(p) for p in sorted(package.glob('*.py'))}}


def server_identity(executable, environment, work):
    program = ('import policy_doctor,hashlib,json;from pathlib import Path;'
               'p=Path(policy_doctor.__file__).parent;'
               'print(json.dumps({"version":policy_doctor.__version__,"package":str(p),'
               '"module_sha256":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.glob("*.py"))}}))')
    result = subprocess.check_output([str(executable), '-I', '-c', program], env=environment, cwd=work, text=True, timeout=15)
    return json.loads(result)


def worker(args):
    revision = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != LEROBOT_REVISION:
        raise ValueError(f'Expected trusted LeRobot {LEROBOT_REVISION}, got {revision}')
    subprocess.run(['git', '-C', str(args.upstream), 'diff', '--quiet', 'HEAD', '--', 'src/lerobot'], check=True)
    sys.path.insert(0, str(args.upstream / 'src'))
    import numpy as np
    import torch
    from lerobot.processor.pipeline import PolicyProcessorPipeline

    from policy_doctor.capture import capture_runtime

    torch.set_num_threads(2)
    common = {'device_processor': {'device': 'cpu'},
              'tokenizer_processor': {'tokenizer_name': str(args.tokenizer)}}
    config = json.loads((args.checkpoint / 'config.json').read_text())
    config['device'] = 'cpu'
    # Controlled inputs, not outputs of a network or physical robot observations.
    raw_state = torch.zeros(8)
    supplied = torch.tensor([[0.] * 7, [.5] * 7, [-.5] * 7], dtype=torch.float32)
    outputs = {}
    for label in ('reference', 'changed', 'corrected'):
        pre_overrides, post_overrides = copy.deepcopy(common), {}
        if label == 'changed':
            baseline = PolicyProcessorPipeline.from_pretrained(str(args.checkpoint), config_filename='policy_postprocessor.json')
            stats = copy.deepcopy(baseline.steps[0].stats)
            stats['action']['mean'][0] += .25
            pre_overrides['normalizer_processor'] = {'stats': stats}
            post_overrides['unnormalizer_processor'] = {'stats': stats}
        pre = PolicyProcessorPipeline.from_pretrained(str(args.checkpoint),
            config_filename='policy_preprocessor.json', overrides=pre_overrides)
        post = PolicyProcessorPipeline.from_pretrained(str(args.checkpoint),
            config_filename='policy_postprocessor.json', overrides=post_overrides)
        batch = pre({'observation.state': raw_state.clone(),
                     'observation.images.image': torch.zeros(3, 256, 256),
                     'observation.images.image2': torch.zeros(3, 256, 256),
                     'task': 'Put the object in the container.'})
        decoded = post({'action': supplied.unsqueeze(0).clone()})['action'].squeeze(0)
        snapshot = capture_runtime(args.work / label, policy_config=config, preprocessor=pre, postprocessor=post)
        assert snapshot['capture']['integrity'] == 'verified'
        outputs[label] = {'commands': decoded.tolist(),
                          'normalized_state': batch['observation.state'].squeeze(0).tolist(),
                          'token_shape': list(batch['observation.language.tokens'].shape),
                          'processed_keys': sorted(batch), 'capture_status': snapshot['status']}
        save(args.work / f'{label}-sample.json', {'model_output': supplied.tolist(),
             'observation.state': raw_state.tolist(), 'expected_commands': decoded.tolist()})
    assert np.allclose(outputs['corrected']['commands'], outputs['reference']['commands'], rtol=0, atol=0)
    delta = np.asarray(outputs['changed']['commands']) - np.asarray(outputs['reference']['commands'])
    assert np.allclose(delta[:, 0], .25, atol=1e-7) and np.allclose(delta[:, 1:], 0, atol=0)
    loaded = {}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).is_file() and Path(path).resolve().is_relative_to(args.upstream / 'src'):
            loaded[str(Path(path).resolve().relative_to(args.upstream))] = digest(path)
    save(args.work / 'upstream-output.json', {'source_revision': revision, 'loaded_source_hashes': loaded,
         'torch': torch.__version__, 'device': 'cpu', 'outputs': outputs, 'doctor': package_identity(),
         'versions': {name: importlib.metadata.version(name) for name in
                      ('torch', 'transformers', 'huggingface_hub', 'tokenizers', 'numpy', 'safetensors')},
         'harness_sha256': digest(__file__),
         'scope': 'Actual released pre/postprocessor execution on controlled images/state and supplied action outputs. No model inference, controller or hardware.'})


async def exercise(args):
    import h5py
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from policy_doctor import __version__

    native_inputs = [args.push_t, args.toolhang, args.trajectory]
    before = {str(p): digest(p) for p in native_inputs}
    upstream = json.loads((args.work / 'upstream-output.json').read_text())
    # Derivations are explicitly injected controls. Original inputs remain unchanged.
    table = pq.read_table(args.push_t, filters=[('episode_index', '=', 0)])
    actions = table['action'].to_pylist()
    actions[5][0] += .25
    column = table.schema.field('action')
    changed = table.set_column(table.schema.get_field_index('action'), column, pa.array(actions, type=column.type))
    parquet_fault = args.work / 'injected-pusht.parquet'
    pq.write_table(changed, parquet_fault)
    h5_fault = args.work / 'injected-toolhang.hdf5'
    with h5py.File(args.toolhang, 'r') as source, h5py.File(h5_fault, 'x') as destination:
        group = destination.create_group('data/demo_0')
        for name in ('actions', 'interventions', 'policy_acting', 'states', 'user_acting'):
            values = source['data/demo_0/' + name][...]
            if name == 'actions':
                values[5, 0] += .25
            group[name] = values
    cases = json.loads(args.oracle.read_text())
    if cases['source_sha256'] != 'a979c1bca2527a912571f5e1ed29e5887d7bf92aa226f6918a125a462821537d':
        raise ValueError('Expected pinned native C++ reference receipt')
    oracle = {item['case']['sample_ns']: item['sampled'] for item in cases['results']}
    assert len(oracle) == 93
    environment = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    installed_server = server_identity(args.server_python, environment, args.work)
    caller = package_identity()
    assert installed_server['module_sha256'] == upstream['doctor']['module_sha256'] == caller['module_sha256']
    assert installed_server['version'] == upstream['doctor']['version'] == caller['version']
    roots = sorted({str(args.work), str(args.checkpoint), *(str(p.parent) for p in native_inputs)})
    arguments = ['-I', '-m', 'policy_doctor', 'serve']
    for root in roots:
        arguments.extend(['--root', root])
    params = StdioServerParameters(command=str(args.server_python), args=arguments, cwd=str(args.work), env=environment)
    calls, results = [], {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            tools = (await client.list_tools()).tools
            assert len(tools) == 29 and all(tool.annotations.readOnlyHint for tool in tools)

            async def call(label, name, arguments):
                response = await client.call_tool(name, arguments)
                assert not response.isError, (label, response.content)
                result = response.structuredContent
                assert isinstance(result, dict), label
                save(args.work / f'mcp-{label}.json', result)
                calls.append({'label': label, 'tool': name, 'status': result.get('status'),
                              'response_sha256': digest(args.work / f'mcp-{label}.json')})
                return result

            audited = await call('capture', 'audit_runtime_capture', {'capture': str(args.work / 'reference')})
            assert audited['capture']['integrity'] == 'verified'
            differences = await call('stats-change', 'compare_runtime_captures',
                {'reference': str(args.work / 'reference'), 'candidate': str(args.work / 'changed')})
            assert differences['status'] == 'different'
            assert any('/statistics/action/mean' in item['pointer'] for item in differences['changes'])
            restored = await call('restored-capture', 'compare_runtime_captures',
                {'reference': str(args.work / 'reference'), 'candidate': str(args.work / 'corrected')})
            assert restored['equal']
            for label in ('reference', 'changed', 'corrected'):
                traced = await call(label, 'trace_saved_actions', {'checkpoint': str(args.work / label / 'checkpoint'),
                                     'sample': str(args.work / f'{label}-sample.json')})
                assert traced['status'] == 'pass'
                expected = upstream['outputs'][label]
                error = float(np.max(np.abs(np.asarray(traced['commands']) - np.asarray(expected['commands']))))
                state_error = float(np.max(np.abs(np.asarray(traced['state_normalization']['normalized']) - expected['normalized_state'])))
                assert np.allclose(traced['commands'], expected['commands'], rtol=1e-6, atol=1e-8)
                assert np.allclose(traced['state_normalization']['normalized'], expected['normalized_state'], rtol=1e-6, atol=1e-6)
                results[label] = {'commands_max_absolute_error': error, 'state_max_absolute_error': state_error,
                                  'mcp_status': traced['status'], 'unsupported_preprocessors': traced['unsupported_preprocessors']}
            fault = await call('wrong-expected', 'trace_saved_actions', {'checkpoint': str(args.work / 'changed/checkpoint'),
                               'sample': str(args.work / 'reference-sample.json')})
            assert fault['status'] == 'error' and fault['findings'][0]['code'] == 'trace.expected_commands'
            recordings = {}
            for label, original, candidate, episode, alignment in [
                ('pusht-control', args.push_t, args.push_t, 0, 'frame_index'),
                ('pusht-injected', args.push_t, parquet_fault, 0, 'frame_index'),
                ('toolhang-control', args.toolhang, args.toolhang, 'demo_0', None),
                ('toolhang-injected', args.toolhang, h5_fault, 'demo_0', None)]:
                report = await call(label, 'compare_rollout_recordings', {'reference': str(original), 'candidate': str(candidate),
                    'reference_episode': episode, 'candidate_episode': episode, 'alignment_key': alignment})
                is_fault = label.endswith('injected')
                assert report['observed_difference'] == is_fault
                if is_fault:
                    first = report['first_observed_difference']
                    assert first['row'] == 5 and first['coordinate'] == [0]
                else:
                    assert all(stream['status'] == 'equal' for stream in report['streams'])
                if label.startswith('toolhang'):
                    assert report['alignment']['status'] == 'unknown'
                recordings[label] = {'status': report['status'], 'alignment': report['alignment']['status'],
                    'stream_count': len(report['streams']), 'rows': report['streams'][0]['rows_compared'],
                    'first_difference': report['first_observed_difference'] if is_fault else None}
            sampled = await call('public-ros', 'sample_ros_trajectory', {'trajectory': str(args.trajectory),
                                 'sample_times': [ns / 1e9 for ns in sorted(oracle)]})
            assert len(sampled['samples']) == len(oracle) == 93
            maximum = 0.
            for sample in sampled['samples']:
                expected = oracle[sample['time_from_start_ns']]
                for field in ('positions', 'velocities', 'accelerations', 'effort'):
                    x, y = np.asarray(sample[field]), np.asarray(expected[field])
                    assert np.allclose(x, y, rtol=1e-10, atol=1e-10)
                    maximum = max(maximum, float(np.max(np.abs(x - y))))
            assert sampled['execution_status'] == sampled['physical_cause'] == 'unknown'
    after = {str(p): digest(p) for p in native_inputs}
    assert before == after
    assert installed_server == server_identity(args.server_python, environment, args.work)
    assert all(digest(Path(path)) == sha for path, sha in args.originals.items())
    record = {'status': 'pass', 'version': __version__, 'mcp_tool_calls': len(calls), 'calls': calls,
              'installed_server': installed_server, 'upstream_worker_doctor': upstream['doctor'],
              'caller_doctor': caller, 'sdk_version': importlib.metadata.version('mcp'),
              'upstream_dependency_versions': upstream['versions'],
              'upstream_inputs_unchanged': True, 'upstream_input_hashes': args.originals,
              'harness_sha256': digest(__file__),
              'public_checkpoint': {'repository': 'lerobot/smolvla_libero', 'revision': CHECKPOINT_REVISION,
                                    'files': CHECKPOINT_FILES},
              'pipeline': {'revision': upstream['source_revision'], 'device': 'cpu', 'torch': upstream['torch'],
                           'cases': results, 'capture_audit_status': audited['status'],
                           'corrected_effective_contract_equal': restored['equal'], 'wrong_expected_status': fault['status']},
              'native_recordings': recordings, 'native_inputs_sha256': before, 'native_inputs_unchanged': True,
              'ros_reference': {'native_cpp_receipt_sha256': digest(args.oracle), 'mcp_samples': len(oracle),
                                'maximum_absolute_error': maximum, 'execution_status': 'unknown'},
              'scope': 'Actual released LeRobot processor execution on controlled inputs; real native recording analysis and actual public ROS reference arithmetic through installed MCP. No model inference, complete ROS controller, simulator, robot task, field efficacy or novelty claim.'}
    save(args.output, record)
    print(json.dumps({'status': 'pass', 'mcp_tool_calls': len(calls), 'pipeline_cases': results,
                      'recording_cases': {k: {n: v[n] for n in ('status', 'alignment', 'rows')} for k, v in recordings.items()},
                      'ros_samples': len(oracle), 'ros_max_error': maximum}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'checkpoint', 'tokenizer', 'work'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    for name in ('pipeline-python', 'server-python', 'push-t', 'toolhang', 'trajectory', 'oracle', 'output'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            # Resolving a venv Python symlink would bypass its installed packages.
            setattr(args, name, value.absolute() if name in ('pipeline_python', 'server_python') else value.resolve())
    if args.worker:
        worker(args)
        return
    for name in ('pipeline_python', 'server_python', 'push_t', 'toolhang', 'trajectory', 'oracle', 'output'):
        if getattr(args, name) is None:
            parser.error('--' + name.replace('_', '-') + ' is required')
    args.work.mkdir(parents=True, exist_ok=False)
    originals = {str(args.checkpoint / name): digest(args.checkpoint / name) for name in CHECKPOINT_FILES}
    assert all(originals[str(args.checkpoint / name)] == expected for name, expected in CHECKPOINT_FILES.items())
    tokenizer = json.loads((args.tokenizer / 'provenance.json').read_text())
    assert tokenizer['revision'] == '7b375e1b73b11138ff12fe22c8f2822d8fe03467'
    for item in tokenizer['files']:
        path = args.tokenizer / item['name']
        assert path.parent == args.tokenizer and digest(path) == item['sha256']
        originals[str(path)] = digest(path)
    args.originals = originals
    environment = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    command = [str(args.pipeline_python), '-I', str(Path(__file__).resolve()), '--worker']
    for name in ('upstream', 'checkpoint', 'tokenizer', 'work'):
        command.extend(['--' + name, str(getattr(args, name))])
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=90)
    (args.work / 'worker.stdout').write_text(result.stdout)
    (args.work / 'worker.stderr').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f'Upstream worker failed; see {args.work}/worker.stderr')
    asyncio.run(asyncio.wait_for(exercise(args), timeout=90))


if __name__ == '__main__':
    main()
