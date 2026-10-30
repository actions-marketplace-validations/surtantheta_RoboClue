"""A training entrypoint's portable GR00T startup gate, with explicit exits."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Examples travel alongside the installed RoboClue wheel, never as PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline  # noqa: E402

from policy_doctor.training_guard import TargetContract, TargetContractViolation  # noqa: E402
from policy_doctor.training_guard_capture import save_capture  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('reference', 'check'))
    parser.add_argument('--packet', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', type=Path, required=True, help='new directory; existing outputs are never overwritten')
    parser.add_argument('--representation', choices=('relative', 'absolute'), required=True)
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--criterion-sha256')
    parser.add_argument('--samples', type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    if not args.packet.is_dir():
        parser.error('--packet directory was not found')
    if args.mode == 'check' and (args.contract is None or args.criterion_sha256 is None):
        parser.error('check requires --contract and --criterion-sha256 retained before the candidate change')
    if args.mode == 'reference' and (args.contract is not None or args.criterion_sha256 is not None):
        parser.error('reference cannot accept an existing --contract or --criterion-sha256')
    if args.mode == 'reference' and args.samples != 2:
        parser.error('the preview reference declares exactly --samples 2')
    if sys.flags.optimize:
        parser.error('run this preview without Python -O; native assertions are part of the profile')
    output = args.output.absolute()
    try:
        if any(path.is_symlink() for path in (output, *output.parents)):
            raise ValueError('--output must not traverse a symlink')
        output.mkdir(mode=0o700)
        sys.path.insert(0, str(args.packet.resolve()))
        from install import verify_packet

        verify_packet(args.packet.resolve())
        contract = (TargetContract.load(args.contract, expected_sha256=args.criterion_sha256)
                    if args.mode == 'check' else None)
        if contract is not None and len(contract.document['references']) != 2:
            raise ValueError('This portable startup profile requires a declared two-sample criterion')
        checkpoint = pipeline.prepare_checkpoint(args.packet, output / 'checkpoint',
                                                 relative=args.representation == 'relative')
        # No network is needed after the private packet has been installed.
        def offline(event, _arguments):
            if event in ('socket.connect', 'socket.getaddrinfo'):
                raise RuntimeError('The installed startup probe is offline')
        sys.addaudithook(offline)
        session = pipeline.build_probe(args.packet, checkpoint, contract=contract, max_samples=args.samples)
        references, reports, native_accepted, delivered = [], [], 0, 0
        status, code = 'pass', 0
        try:
            for index in range(args.samples):
                try:
                    batch = session.next()
                except TargetContractViolation as error:
                    batch = session.tap.last_batch
                    session.preview_schema(batch)  # demonstrate finite, native shape-valid target/mask
                    native_accepted += 1
                    save_capture(batch, [record.source_key for record in session.tap.last_records],
                                 output / f'capture-{index}.json')
                    reports.append(error.report)
                    status = error.report['status']
                    code = 1 if status == 'error' else 3
                    break
                session.preview_schema(batch)
                native_accepted += 1
                records = session.tap.last_records
                save_capture(batch, [record.source_key for record in records], output / f'capture-{index}.json')
                if contract is None:
                    references.append((batch, [record.source_key for record in records],
                                       [record.provenance for record in records]))
                else:
                    reports.append(session.guard.last_report)
                    # Actual boundary delivery: reached only after original collator + guard PASS.
                    delivered += 1
        finally:
            cleanup = session.close()
            pipeline.write(output / 'cleanup.json', cleanup)
        if session.loader.collate_fn is not session.original_collator:
            raise RuntimeError('Startup probe did not restore the original loader collator')
        if contract is None:
            contract = TargetContract.from_references(references, intent={
                'objective': 'Preserve declared native startup targets and masks',
                'representation': args.representation, 'normalization': 'min_max',
                'scope': pipeline.PROFILE_SCOPE,
            })
            contract.save(output / 'contract.json')
            with (output / 'criterion.sha256').open('x') as stream:
                stream.write(contract.criterion_sha256 + '\n')
        result = {
            'status': status, 'mode': args.mode, 'criterion_sha256': contract.criterion_sha256,
            'native_batches_accepted': native_accepted, 'guarded_batches_delivered': delivered,
            'training_may_start': args.mode == 'check' and code == 0,
            'scope': pipeline.PROFILE_SCOPE, 'reports': reports,
            'versions': session.preview_versions,
            'existing_native_loader_reused': session.existing_loader,
            'original_collator_restored': session.loader.collate_fn is session.original_collator,
            'no_model_weights_or_optimizer_executed': True,
        }
        pipeline.write(output / 'result.json', result)
        print(json.dumps(result, sort_keys=True))
        return code
    except (OSError, ValueError, RuntimeError, RecursionError) as error:
        print(json.dumps({'status': 'input_error', 'message': str(error),
                          'training_may_start': False}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    raise SystemExit(main())
