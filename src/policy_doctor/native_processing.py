"""Opt-in native CPU probes and portable numerical capsules, not policy inference."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from . import __version__
from .artifacts import InputError
from .publication import publish_capture

PROVIDERS = {'lerobot': 'native_lerobot_worker.py', 'groot_state': 'native_groot_worker.py',
             'groot_adapter': 'native_groot_adapter_worker.py'}
KIND = 'native_processing_capsule'
MAX_FILE = 128 * 1024**2
MAX_TOTAL = 256 * 1024**2
MAX_JSON = 2 * 1024**2
MAX_RESULT = 128 * 1024
SCOPE = ('Native CPU software observations and numerical replay only. Saved/factory differences can be legal. '
         'Training recipe, model predictions, policy competence, physical execution and causal benefit remain unknown.')
_LAUNCH = """import os,resource,sys
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(280,280))
resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**2,8*1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
if hasattr(os,'sched_getaffinity'):
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
os.execv(sys.argv[1],[sys.argv[1],'-I','-B',sys.argv[2]])
"""
_VERSIONS = """import importlib.metadata as m,json,sys
names=['numpy','scipy','anyio','torch','torchvision','transformers','tokenizers','datasets','pyarrow','av','safetensors','huggingface_hub','draccus']
v={'python':sys.version.split()[0]}
for n in names:
 try:v[n]=m.version(n)
 except m.PackageNotFoundError:v[n]=None
print(json.dumps(v,sort_keys=True))
"""


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _hash(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE or path.is_symlink():
        raise InputError('Native probe files must be bounded regular files without symlinks')
    digest = hashlib.sha256()
    size = 0
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns):
            raise InputError('Native file changed before opening')
        while data := stream.read(1024**2):
            size += len(data)
            if size > MAX_FILE:
                raise InputError('Native probe file grew beyond the bound')
            digest.update(data)
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise InputError('Native input changed during hashing')
    return digest.hexdigest(), size


def _inside(root, name):
    if (not isinstance(name, str) or not name or '\\' in name or '\x00' in name
            or Path(name).is_absolute() or '..' in Path(name).parts or str(Path(name)) != name or name == '.'):
        raise InputError('Native artifact paths must be contained relative paths')
    path = root / name
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            raise InputError('Native artifact symlinks are not supported')
        cursor = cursor.parent
    if not path.resolve().is_relative_to(root.resolve()):
        raise InputError('Native artifact path escapes its root')
    return path


def _json(path, limit=MAX_JSON):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise InputError('Native JSON must be a bounded regular file without symlinks')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (before.st_dev, before.st_ino, before.st_size):
            raise InputError('Native JSON changed before opening')
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise InputError('Native JSON exceeds the bounded intake')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
        pending, nodes = [(value, 0)], 0
        while pending:
            item, depth = pending.pop()
            nodes += 1
            if depth > 32 or nodes > 500_000:
                raise ValueError('Native JSON nesting/node bound exceeded')
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError('Nonfinite JSON')
            if isinstance(item, dict):
                pending.extend((v, depth + 1) for v in item.values())
            elif isinstance(item, list):
                pending.extend((v, depth + 1) for v in item)
        return value
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise InputError('Need bounded finite native JSON: ' + str(exc)) from exc


def _inventory(root):
    if root.is_symlink() or not root.is_dir():
        raise InputError('Require an existing regular native artifact directory')
    files, total, entries = {}, 0, 0
    for base, directories, names in os.walk(root, followlinks=False):
        entries += len(directories) + len(names)
        if entries > 1024:
            raise InputError('Native probe tree exceeds 1024 entries')
        if len(Path(base).relative_to(root).parts) > 8:
            raise InputError('Native probe tree exceeds depth8')
        for name in directories:
            if (Path(base) / name).is_symlink():
                raise InputError('Native probe directory symlink refused')
        for name in sorted(names):
            path = Path(base) / name
            digest, size = _hash(path)
            files[str(path.relative_to(root))] = {'sha256': digest, 'size_bytes': size}
            total += size
            if len(files) > 256 or total > MAX_TOTAL:
                raise InputError('Native pilot accepts bounded trees: 256 files,256MiB total')
    return files


def _source_tree(root):
    files, total = {}, 0
    for path in sorted(root.rglob('*.py')):
        if any(part == '__pycache__' for part in path.parts):
            continue
        digest, size = _hash(path)
        files[str(path.relative_to(root))] = digest
        total += size
        if len(files) > 4096 or total > 32 * 1024**2:
            raise InputError('Trusted native source exceeds pilot bounds')
    if not files:
        raise InputError('Native runtime must identify original Python source')
    return hashlib.sha256(_encode(files)).hexdigest()


def load_native_runtimes(path, *, require_initial_cache=True):
    value = _json(Path(path).expanduser(), 64 * 1024)
    if type(value) is not dict or set(value) != {'providers'} or type(value['providers']) is not dict:
        raise InputError('Native runtime config accepts only a providers mapping')
    if not value['providers'] or set(value['providers']) - PROVIDERS.keys():
        raise InputError('Unknown or empty native provider configuration')
    result = {}
    for provider, runtime in value['providers'].items():
        required = {'python', 'source_root'}
        allowed = required | {'cache_root'}
        if provider == 'groot_adapter':
            required = required | {'groot_source_root'}
            allowed = required | {'reference_source_root'}
        if (type(runtime) is not dict or not required <= runtime.keys()
                or set(runtime) - allowed
                or any(type(v) is not str or not v for v in runtime.values())):
            raise InputError('Native runtime requires explicitly configured provider code roots')
        # Preserve the venv executable spelling; resolving its symlink loses pyvenv.cfg.
        python = Path(runtime['python']).expanduser().absolute()
        source = Path(runtime['source_root']).expanduser().resolve()
        if not python.is_file() or not os.access(python, os.X_OK) or not source.is_dir():
            raise InputError('Configured native interpreter/source must exist')
        result[provider] = {'python': str(python), 'source_root': str(source)}
        if provider == 'groot_adapter':
            for key in ('groot_source_root', 'reference_source_root'):
                if key in runtime:
                    auxiliary = Path(runtime[key]).expanduser().resolve()
                    if not auxiliary.is_dir():
                        raise InputError('Configured native auxiliary source root must exist')
                    result[provider][key] = str(auxiliary)
        if 'cache_root' in runtime:
            cache = Path(runtime['cache_root']).expanduser().resolve()
            if (require_initial_cache or cache.exists()) and not cache.is_dir():
                raise InputError('Configured native cache root must exist')
            result[provider]['cache_root'] = str(cache)
    return result


def _environment(runtime, scratch):
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')
           and not k.startswith(('HF_', 'HUGGINGFACE_', 'TRANSFORMERS_', 'XDG_CACHE_'))}
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', RAYON_NUM_THREADS='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
        HF_HOME=str(scratch / 'hf'), HF_HUB_CACHE=runtime.get('cache_root', str(scratch / 'hf' / 'hub')))
    return env


def _fingerprint(provider, runtime):
    worker = Path(__file__).with_name(PROVIDERS[provider])
    with tempfile.TemporaryDirectory(prefix='policy-doctor-native-version-') as folder:
        p = subprocess.run([runtime['python'], '-I', '-B', '-c', _VERSIONS], cwd=folder,
            env=_environment(runtime, Path(folder)), capture_output=True, timeout=20)
    if p.returncode or len(p.stdout) > 16 * 1024:
        raise InputError('Unable to attest configured native interpreter metadata')
    result = {'engine_version': __version__, 'provider': provider, 'worker_sha256': _hash(worker)[0],
        'parent_sha256': _hash(Path(__file__))[0],
        'row_intake_sha256': _hash(Path(__file__).with_name('native_rows_worker.py'))[0],
        'dependency_intake_sha256': _hash(Path(__file__).with_name('native_intake.py'))[0],
        'dependency_planner_sha256': _hash(Path(__file__).with_name('native_lerobot_plan_worker.py'))[0],
        'parquet_guard_sha256': _hash(Path(__file__).with_name('native_parquet_guard.py'))[0],
        'source_tree_sha256': _source_tree(Path(runtime['source_root'])), 'versions': json.loads(p.stdout)}
    if provider == 'groot_adapter':
        result['auxiliary_source_tree_sha256'] = {
            key: _source_tree(Path(runtime[key])) for key in ('groot_source_root', 'reference_source_root') if key in runtime}
    return result


def _samples(value):
    if type(value) is not list or not 1 <= len(value) <= 2:
        raise InputError('Native probes require one or two explicit sample identifiers')
    normalized = []
    for sample in value:
        if type(sample) is not dict or set(sample) not in ({'episode_index', 'frame_index'}, {'episode_index', 'physical_row'}) or any(
                type(v) is not int or not 0 <= v <= 10**9 for v in sample.values()):
            raise InputError('Native sample IDs require episode_index and either native frame_index or explicit physical_row')
        if sample in normalized:
            raise InputError('Native sample identifiers must be unique')
        if normalized and set(sample) != set(normalized[0]):
            raise InputError('Use one selection basis for all samples in a native probe')
        normalized.append(dict(sample))
    return normalized


def _validate_inputs(provider, model, dataset):
    inventories = {'model': _inventory(model), 'dataset': _inventory(dataset)}
    for root in (model, dataset):
        for path in sorted(root.rglob('*.json')):
            _json(path)
    info = _json(_inside(dataset, 'meta/info.json'))
    if type(info) is not dict:
        raise InputError('Native info.json must be an object')
    data_templates = {'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                      'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'}
    video_templates = {'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
                       'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}
    if info.get('data_path') not in data_templates or (info.get('video_path') and info['video_path'] not in video_templates):
        raise InputError('Native pilot requires supported contained native data/video templates')
    features = info.get('features')
    if type(features) is not dict or any(not isinstance(k, str) or '/' in k or '\\' in k or '..' in k for k in features):
        raise InputError('Native feature keys must not contain path components')
    if provider == 'groot_state' and not _inside(dataset, 'meta/modality.json').is_file():
        raise InputError('NVIDIA state probe requires original modality mapping')
    if provider == 'groot_state' and info['data_path'] != 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet':
        raise InputError('NVIDIA pilot currently supports original per-episode Parquet files only')
    return inventories


def _invoke(script, runtime, request, workspace):
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=False)
    with (workspace / 'stdout.log').open('wb') as stdout, (workspace / 'stderr.log').open('wb') as stderr:
        try:
            p = subprocess.run([sys.executable, '-I', '-B', '-c', _LAUNCH, runtime['python'], str(script)],
                input=_encode(request), stdout=stdout, stderr=stderr, cwd=workspace,
                env=_environment(runtime, workspace), timeout=300)
        except subprocess.TimeoutExpired as exc:
            raise InputError('Native CPU worker exceeded its300second wall limit') from exc
    # This fixed producer retains larger step evidence in result.json. Its small
    # receipt is validated by the source-capture parent before reading that file.
    receipt_workers = {'native_processor_source_worker.py', 'native_groot_adapter_worker.py'}
    result_path = output / ('worker_receipt.json' if Path(script).name in receipt_workers else 'result.json')
    if not result_path.is_file():
        message = (workspace / 'stderr.log').read_bytes()[:2048].decode(errors='replace')
        raise InputError('Native CPU probe failed within its resource bounds: ' + message)
    result = _json(result_path, MAX_RESULT)
    if p.returncode and (type(result) is not dict or result.get('execution_status') not in ('partial', 'error')):
        raise InputError('Native worker exited unsuccessfully without a valid failure observation')
    return result


def _groot_rows(dataset, samples, workspace):
    modality = _json(_inside(dataset, 'meta/modality.json'))
    state = modality.get('state')
    if type(state) is not dict or not state:
        raise InputError('Original NVIDIA modality state mapping is missing')
    if any(type(entry) is not dict for entry in state.values()):
        raise InputError('Original NVIDIA state mappings must be objects')
    columns = sorted({entry.get('original_key', 'observation.state') for entry in state.values()
                      if type(entry) is dict})
    if not columns or any(type(k) is not str or not k or len(k) > 256 for k in columns):
        raise InputError('Need explicit original native state columns')
    info = _json(_inside(dataset, 'meta/info.json'))
    chunk_size = info.get('chunks_size')
    if type(chunk_size) is not int or chunk_size <= 0:
        raise InputError('Original native episode chunks_size is missing')
    collected = []
    for index, sample in enumerate(samples):
        relative = info['data_path'].format(episode_chunk=sample['episode_index'] // chunk_size,
                                            episode_index=sample['episode_index'])
        path = _inside(dataset, relative)
        run = workspace / ('rows-' + str(index))
        run.mkdir()
        result = _invoke(Path(__file__).with_name('native_rows_worker.py'), {'python': sys.executable},
            {'path': str(path), 'relative_path': relative, 'columns': columns, 'samples': [sample],
             'output': str(run / 'output')}, run)
        collected.extend(result['raw_samples'])
    return collected


def _projection(report):
    return {key: report[key] for key in ('provider', 'execution_status', 'samples')}


def _validate_report(report, provider, samples=None):
    if (type(report) is not dict or report.get('provider') != provider
            or report.get('execution_status') not in ('complete', 'partial', 'error')
            or type(report.get('samples')) is not list or len(report['samples']) > 2):
        raise InputError('Native worker returned an invalid provider/result schema')
    if report['execution_status'] == 'complete' and samples is not None and len(report['samples']) != len(samples):
        raise InputError('Native worker did not process every selected sample')
    if report['execution_status'] == 'complete' and not report['samples']:
        raise InputError('Complete native processing requires an observed sample')
    for index, sample in enumerate(report['samples']):
        if type(sample) is not dict or type(sample.get('identity')) is not dict:
            raise InputError('Native worker sample identity is missing')
        identity = sample['identity']
        if any(type(identity.get(k)) is not int or identity[k] < 0 for k in ('episode_index', 'index')):
            raise InputError('Native worker sample IDs are invalid')
        row_mode = sample.get('selection_basis') == 'single_episode_physical_row'
        if row_mode:
            if identity.get('frame_index') is not None or type(identity.get('physical_row')) is not int or identity['physical_row'] < 0:
                raise InputError('Physical-row selection cannot invent a native frame identity')
        elif type(identity.get('frame_index')) is not int or identity['frame_index'] < 0:
            raise InputError('Native frame identity is missing')
        if samples is not None and any(identity.get(k) != v for k, v in samples[index].items()):
            raise InputError('Native worker processed a different requested sample')
        if samples is not None and ('physical_row' in samples[index]) != row_mode:
            raise InputError('Native worker changed the requested selection basis')
        routes = sample.get('routes')
        required = {'saved', 'current_factory'} if provider == 'lerobot' else {'native_state'}
        if type(routes) is not list or any(type(route) is not dict for route in routes):
            raise InputError('Native worker sample routes are missing')
        if report['execution_status'] == 'complete' and ({r.get('name') for r in routes} != required
                or len(routes) != len(required) or any(r.get('status') != 'complete'
                    or type(r.get('input')) is not dict or not r['input']
                    or type(r.get('output')) is not dict or not r['output']
                    or type(r.get('steps')) is not list or not r['steps'] for r in routes)):
            raise InputError('Native worker did not report every required native route')
    receipts = report.get('artifact_receipts', [])
    if type(receipts) is not list or len(receipts) > 128:
        raise InputError('Invalid native original artifact receipts')
    for receipt in receipts:
        if (type(receipt) is not dict or not {'root', 'path', 'sha256', 'size_bytes'} <= receipt.keys()
                or receipt['root'] not in ('model', 'dataset')
                or type(receipt['size_bytes']) is not int or not 0 <= receipt['size_bytes'] <= MAX_FILE
                or type(receipt['sha256']) is not str or len(receipt['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in receipt['sha256'])):
            raise InputError('Invalid native original artifact receipt')
    return report


def _response(report, **extra):
    result = {'kind': 'native_processing_probe', 'status': 'observed' if report['execution_status'] == 'complete' else 'unknown',
              'scope': SCOPE, 'probe': report, 'observed_differences': _observed_differences(report), **extra}
    if len(_encode(result)) > 192 * 1024:
        raise InputError('Native probe response exceeds192KiB; narrow sample selection')
    return result


def _observed_differences(report):
    """Compare native output summaries; a difference is an observation, not a fault."""
    differences = []
    def compact(value):
        if type(value) is dict and 'sha256' in value:
            summary = {k: value[k] for k in ('dtype', 'shape', 'min', 'max', 'sha256') if k in value}
            if isinstance(value.get('preview'), list) and len(value['preview']) <= 4:
                summary['preview'] = value['preview']
            return summary
        return value
    for sample in report['samples']:
        routes = sample.get('routes', [])
        if len(routes) != 2 or any(type(route.get('output')) is not dict for route in routes):
            continue
        left, right = (route['output'] for route in routes)
        changed = []
        for key in sorted(left.keys() | right.keys()):
            if left.get(key) != right.get(key) or (key in left) != (key in right):
                changed.append({'key': key, 'reference': compact(left.get(key)), 'candidate': compact(right.get(key)),
                                'reference_present': key in left, 'candidate_present': key in right})
        differences.append({**{key: sample['identity'][key] for key in ('episode_index', 'frame_index')},
            'reference_route': routes[0]['name'], 'candidate_route': routes[1]['name'],
            'changed_outputs': changed[:32], 'total_changed_outputs': len(changed),
            'interpretation': 'Observed native output differences; expected recipe and policy correctness remain unknown.'})
    return differences


def probe_native_processing(provider, model, dataset, *, runtimes, samples, destination=None,
                            mode='observation', processor_source=None, intent=None, approved_sources=None, contract=None):
    # Replay reconstructs its own cache. A new probe still needs any cache the
    # operator explicitly configured, even when the server started for replay.
    if type(provider) is str and provider in runtimes:
        cache = runtimes[provider].get('cache_root')
        if cache is not None and not Path(cache).is_dir():
            raise InputError('Configured native cache root must exist for an initial probe')
    if provider == 'groot_adapter':
        if mode != 'observation' or any(value is not None for value in (processor_source, intent, approved_sources)):
            raise InputError('GR00T adapter input contracts use the observation mode')
        from .native_observation_processing import probe_native_observation_source

        return probe_native_observation_source(model, dataset, runtimes=runtimes, samples=samples,
            destination=destination, contract=contract)
    if contract is not None:
        raise InputError('An observation source contract requires the configured GR00T adapter provider')
    if mode == 'processor_source':
        if provider != 'lerobot':
            raise InputError('Processor-source mode supports the configured LeRobot SmolVLA path only')
        from .native_source_processing import probe_native_processor_source

        return probe_native_processor_source(model, dataset, runtimes=runtimes, samples=samples,
            destination=destination, processor_source=processor_source, intent=intent, approved_sources=approved_sources)
    if mode != 'observation' or any(value is not None for value in (processor_source, intent, approved_sources)):
        raise InputError('Processor source, intent and approved snapshots require processor_source mode')
    if type(provider) is not str or provider not in PROVIDERS or provider not in runtimes:
        raise InputError('Native provider is not explicitly configured')
    samples = _samples(samples)
    if provider != 'groot_state' and any('physical_row' in sample for sample in samples):
        raise InputError('This native provider requires actual native frame_index selection')
    original_model, original_dataset = Path(model).expanduser(), Path(dataset).expanduser()
    if original_model.is_symlink() or original_dataset.is_symlink():
        raise InputError('Native input root symlinks are not supported')
    model, dataset = original_model.resolve(), original_dataset.resolve()
    runtime = runtimes[provider]
    verifier = _fingerprint(provider, runtime)
    target = Path(destination).expanduser().absolute() if destination is not None else None
    if target is not None:
        target = target.parent.resolve() / target.name
        if target.exists() or target.is_symlink() or any(target.is_relative_to(root) for root in (model, dataset)):
            raise InputError('Choose a new native capsule destination outside input trees')
        target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.policy-doctor-native-', dir=target.parent if target else None) as folder:
        workspace = Path(folder)
        stage = workspace / 'capsule'
        prepared = _prepare_native_inputs(provider, model, dataset, samples, workspace, runtime, _invoke)
        selected = {(r['root'], r['path']): r for r in prepared['receipts']}
        request = {'operation': 'probe', 'source_root': runtime['source_root'], 'model': str(prepared['model']),
                   'dataset': str(prepared['dataset']), 'samples': samples, 'output': str(stage)}
        if 'cache_root' in runtime:
            request['cache_root'] = runtime['cache_root']
        if provider == 'groot_state':
            request['raw_samples'] = _groot_rows(prepared['dataset'], samples, workspace)
        report = _validate_report(_invoke(Path(__file__).with_name(PROVIDERS[provider]), runtime, request, workspace), provider, samples)
        _verify_native_inputs(prepared['selection'])
        if verifier != _fingerprint(provider, runtime):
            raise InputError('Native source/runtime changed during probe')
        for receipt in report.get('artifact_receipts', []):
            accepted = selected.get((receipt['root'], receipt['path']))
            if accepted is None or any(receipt[k] != accepted[k] for k in ('sha256', 'size_bytes')):
                raise InputError('Native worker used an unapproved dependency or mismatched original bytes')
        selection_report = {'files': prepared['receipts'], 'omissions': prepared['omissions'],
                            'native_input_role': 'automatic byte-identical private dependency view'}
        if target is None:
            return _response(report, retained=False, artifact_files_available=False,
                             retention_note='Temporary trace/export files were removed; use --case to retain them.',
                             input_selection=selection_report)
        if report['execution_status'] != 'complete':
            raise InputError('An incomplete native probe cannot publish a numerical replay capsule: '
                             + json.dumps(report.get('error', report.get('unverified', [])))[:4096])
        _durable_payload(stage, report)
        copied = set()
        for receipt in prepared['receipts']:
            if receipt.get('root') not in ('model', 'dataset'):
                raise InputError('Unknown original artifact receipt root')
            root = prepared['model'] if receipt['root'] == 'model' else prepared['dataset']
            source = _inside(root, receipt['path'])
            sha, size = _hash(source)
            if sha != receipt['sha256'] or size != receipt['size_bytes']:
                raise InputError('Original native artifact receipt does not match bytes')
            copy = _inside(stage, 'originals/' + receipt['root'] + '/' + receipt['path'])
            if copy in copied:
                if _hash(copy) != (sha, size):
                    raise InputError('Conflicting original artifact receipts')
                continue
            copy.parent.mkdir(parents=True, exist_ok=True)
            with copy.open('xb') as stream, source.open('rb') as incoming:
                shutil.copyfileobj(incoming, stream)
            if _hash(copy) != (sha, size):
                raise InputError('Original native artifact changed while retaining bytes')
            copied.add(copy)
        (stage / 'expected.json').write_bytes(_encode(_projection(report)))
        (stage / 'input_selection.json').write_bytes(_encode(selection_report))
        (stage / 'README.md').write_text('# Native processing numerical capsule\n\n' + SCOPE + '\n\n'
            'Configure the same trusted native runtime and use reproduce-native-probe. '
            'Inputs/exports are retained numerical evidence, not a replacement training dataset. '
            'Original membership is supported only where original source files are included; other source hashes are capture assertions. '
            'Check original model/data/native licenses before redistribution. No model weights or policy inference are included.\n')
        _verify_native_inputs(prepared['selection'])
        inventory = _inventory(stage)
        manifest = {'kind': KIND, 'schema_version': 1, 'provider': provider, 'verifier': verifier,
                    'files': inventory, 'scope': SCOPE}
        (stage / 'manifest.json').write_bytes(_encode(manifest))
        (stage / 'READY.json').write_bytes(_encode({'manifest_sha256': _hash(stage / 'manifest.json')[0]}))
        _inventory(stage)
        publish_capture(stage, target)
        return _response(report, retained=True, directory=str(target), verifier=verifier,
                         input_selection=selection_report)


def _prepare_native_inputs(*args, **kwargs):
    from .native_intake import prepare_native_inputs

    return prepare_native_inputs(*args, **kwargs)


def _verify_native_inputs(selection):
    from .native_intake import verify_native_inputs

    verify_native_inputs(selection)


def _durable_payload(stage, report):
    """Remove only owned temporary output, then enforce a positive payload scope."""
    allowed = {'samples', 'traces', 'exports', 'declarations', 'inputs', 'metadata',
               'worker_manifest.json', 'result.json', 'runtime'}
    for child in stage.iterdir():
        if child.name not in allowed or child.is_symlink():
            raise InputError('Unexpected native worker output cannot be retained')
    approved = {'worker_manifest.json', 'result.json'}
    manifest = _json(_inside(stage, 'worker_manifest.json'))
    if type(manifest) is not dict or manifest.get('provider') != report.get('provider'):
        raise InputError('Native durable payload requires its provider manifest')
    def descriptor(name, receipt=None):
        path = _inside(stage, name)
        if Path(name).parts[0] == 'runtime' or not path.exists():
            raise InputError('Durable native evidence cannot depend on temporary runtime output')
        if receipt is not None and 'sha256' in receipt:
            sha, size = _hash(path)
            if sha != receipt['sha256'] or size != receipt.get('size_bytes', receipt.get('bytes', size)):
                raise InputError('Durable native artifact descriptor does not match retained bytes')
        if path.is_file():
            approved.add(name)
    def exports(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'export' and isinstance(item, str):
                    descriptor(item)
                else:
                    exports(item)
        elif isinstance(value, list):
            for item in value:
                exports(item)
    for index, sample in enumerate(report['samples']):
        for route in sample.get('routes', []):
            if isinstance(route.get('trace'), dict) and isinstance(route['trace'].get('path'), str):
                if (report.get('provider') != 'lerobot' or route['trace']['path'] !=
                        f"traces/sample_{index}_{route.get('name')}.json"):
                    raise InputError('Unsupported native step-trace payload path')
                descriptor(route['trace']['path'], route['trace'])
        exports(sample)
    if report['provider'] == 'lerobot':
        if (manifest.get('schema') != 'policy_doctor.native_lerobot.v1'
                or type(manifest.get('samples')) is not list
                or len(manifest['samples']) != len(report['samples'])
                or manifest.get('exports') != report.get('exports', [])):
            raise InputError('Invalid native LeRobot durable manifest')
        for index, item in enumerate(manifest['samples']):
            for field, suffix in (('arrays', 'npz'), ('metadata', 'json')):
                value = item.get(field)
                expected = f'samples/sample_{index}.{suffix}'
                if type(value) is not dict or value.get('path') != expected:
                    raise InputError('Unsupported native sample payload path')
                descriptor(expected, value)
    elif report['provider'] == 'groot_state':
        permitted = {'metadata/model/processor_config.json', 'metadata/model/statistics.json',
                     'metadata/dataset/meta/info.json', 'metadata/dataset/meta/modality.json', 'inputs/raw_samples.json',
                     'worker_manifest.json'}
        for index, sample in enumerate(report['samples']):
            for route in sample['routes']:
                for value in route['output'].values():
                    if type(value) is dict and isinstance(value.get('export'), str):
                        if not re.fullmatch(f'samples/{index:02d}/native_state/[0-9]{{2}}[.]json', value['export']):
                            raise InputError('Unsupported native state payload path')
                        permitted.add(value['export'])
        if (manifest.get('schema_version') != 1 or type(manifest.get('retained_files')) is not list
                or {item.get('path') for item in report.get('exports', [])} != permitted
                or {item.get('path') for item in manifest['retained_files']} != permitted - {'worker_manifest.json'}):
            raise InputError('Invalid native GR1 durable manifest')
    else:
        raise InputError('Unsupported native durable provider')
    for item in report.get('exports', []):
        if isinstance(item, dict) and isinstance(item.get('path'), str):
            descriptor(item['path'], item)
            permitted_export = None
            if report['provider'] == 'lerobot':
                from .native_lerobot_worker import TOKENIZER_FILES, _validate_pipeline

                base = _inside(stage, item['path'])
                filename = item.get('config_filename')
                if item['path'] != 'exports/' + item.get('route', '') or type(filename) is not str:
                    raise InputError('Invalid native processor export root')
                config = _validate_pipeline(base, filename, replay=True)
                permitted_export = {filename}
                for step in config['steps']:
                    if 'state_file' in step:
                        permitted_export.add(step['state_file'])
                    for folder in step.get('artifacts', {}).values():
                        for candidate in _inside(base, folder).iterdir():
                            if candidate.name not in TOKENIZER_FILES:
                                raise InputError('Unexpected native tokenizer export')
                            permitted_export.add(str(Path(folder) / candidate.name))
                if {child.get('path') for child in item.get('files', [])} != permitted_export:
                    raise InputError('Native export contains unreferenced files')
            for child in item.get('files', []):
                if type(child) is not dict or not isinstance(child.get('path'), str):
                    raise InputError('Invalid native export file descriptor')
                descriptor(str(Path(item['path']) / child['path']), child)
            declaration = item.get('original_declaration')
            if isinstance(declaration, dict) and isinstance(declaration.get('path'), str):
                descriptor(declaration['path'], declaration)
    runtime = stage / 'runtime'
    if runtime.exists():
        if not runtime.is_dir():
            raise InputError('Native temporary runtime must be an owned directory')
        shutil.rmtree(runtime)
    inventory = _inventory(stage)
    if set(inventory) != approved:
        raise InputError('Native durable payload contains unreferenced or missing files')
    approved_directories = {str(parent) for name in approved for parent in Path(name).parents if str(parent) != '.'}
    for base, directories, _ in os.walk(stage):
        for name in directories:
            if str((Path(base) / name).relative_to(stage)) not in approved_directories:
                raise InputError('Native durable payload contains an unreferenced directory')


def reproduce_native_processing(directory, *, runtimes):
    candidate = Path(directory).expanduser()
    if not candidate.is_symlink() and (candidate / 'manifest.json').is_file():
        declaration = _json(_inside(candidate.resolve(), 'manifest.json'))
        if type(declaration) is dict and declaration.get('kind') in ('native_observation_source_attempt', 'native_observation_source_view'):
            from .native_observation_processing import reproduce_native_observation_source

            return reproduce_native_observation_source(candidate.resolve(), runtimes=runtimes)
        if type(declaration) is dict and declaration.get('kind') in ('native_processor_source_attempt', 'native_processor_source_view'):
            from .native_source_processing import reproduce_native_processor_source

            return reproduce_native_processor_source(directory, runtimes=runtimes)
    if not candidate.is_symlink() and (candidate / 'capture.json').is_file():
        declaration = _json(_inside(candidate.resolve(), 'capture.json'))
        if type(declaration) is dict and declaration.get('kind') == 'native_observation_source_capture':
            from .native_observation_processing import reproduce_native_observation_source

            return reproduce_native_observation_source(candidate.resolve(), runtimes=runtimes)
        if type(declaration) is dict and declaration.get('kind') == 'native_processor_source_capture':
            from .native_source_processing import reproduce_native_processor_source

            return reproduce_native_processor_source(directory, runtimes=runtimes)
    original = Path(directory).expanduser()
    if original.is_symlink():
        raise InputError('Native capsule root symlink refused')
    root = original.resolve()
    before = _inventory(root)
    manifest = _json(_inside(root, 'manifest.json'))
    ready = _json(_inside(root, 'READY.json'))
    if (type(manifest) is not dict or manifest.get('kind') != KIND or type(manifest.get('schema_version')) is not int
            or manifest.get('schema_version') != 1 or manifest.get('provider') not in PROVIDERS
            or ready != {'manifest_sha256': _hash(root / 'manifest.json')[0]}):
        raise InputError('Invalid or uncommitted native capsule')
    files = {k: v for k, v in before.items() if k not in ('manifest.json', 'READY.json')}
    if files != manifest.get('files'):
        raise InputError('Native capsule file hashes or membership changed')
    provider = manifest['provider']
    if provider not in runtimes:
        raise InputError('Capsule provider requires an explicitly configured native runtime')
    runtime = runtimes[provider]
    if _fingerprint(provider, runtime) != manifest.get('verifier'):
        return {'kind': 'native_processing_reproduction', 'status': 'unknown', 'verifier_matches': False,
                'candidate_evaluated': False, 'scope': SCOPE}
    with tempfile.TemporaryDirectory(prefix='policy-doctor-native-replay-') as folder:
        workspace = Path(folder)
        report = _validate_report(_invoke(Path(__file__).with_name(PROVIDERS[provider]), runtime,
            {'operation': 'replay', 'source_root': runtime['source_root'], 'capsule_root': str(root),
             'output': str(workspace / 'output')}, workspace), provider)
    expected = _json(_inside(root, 'expected.json'), MAX_RESULT)
    matches = report['execution_status'] == 'complete' and _encode(_projection(report)) == _encode(expected)
    if before != _inventory(root) or _fingerprint(provider, runtime) != manifest['verifier']:
        raise InputError('Native capsule/runtime changed during numerical replay')
    return _response(report, kind='native_processing_reproduction', status='pass' if matches else 'error',
                     reproduction_matches=matches, verifier_matches=True, directory=str(root),
                     input_selection=_json(_inside(root, 'input_selection.json')),
                     replay_input_role='Retained arrays and native exports; original videos were not decoded again.')
