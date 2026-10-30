"""Generated parser/source-data controls; native serving is never invoked."""
import builtins
import hashlib
import json
import os
import shutil
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from policy_doctor import openpi_interface
from policy_doctor.openpi_interface import MANIFEST, resolve_serving_command

ROOT = Path(tempfile.gettempdir())
FIXTURES = Path(__file__).parent / 'fixtures/openpi_source_profile_215abfb'
SOURCE = None
_SOURCE_TEMP = None
BASE = 'uv run scripts/serve_policy.py'
CHECKPOINT = 'gs://openpi-assets/checkpoints/pi0_base'


def setUpModule():
    global SOURCE, _SOURCE_TEMP
    _SOURCE_TEMP = tempfile.TemporaryDirectory(prefix='openpi-source-profile-')
    SOURCE = Path(_SOURCE_TEMP.name)
    for relative, identity in MANIFEST['files'].items():
        data = (FIXTURES / (relative + '.txt')).read_bytes()
        assert hashlib.sha256(data).hexdigest() == identity['sha256']
        assert len(data) == identity['size_bytes']
        target = SOURCE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def tearDownModule():
    _SOURCE_TEMP.cleanup()


def command(config='pi0_droid', checkpoint=CHECKPOINT):
    return f'{BASE} policy:checkpoint --policy.config={config} --policy.dir={checkpoint}'


class BuiltinResolution(unittest.TestCase):
    def resolve(self, text):
        result = resolve_serving_command(text, SOURCE)
        self.assertEqual(result['status'], 'resolved_supported_builtin', result)
        self.assertFalse(result['checkpoint_io_performed'])
        self.assertFalse(result['native_factory_executed'])
        return result

    def test_actual_pi0_droid_source_defaults_and_inverse_boundary(self):
        result = self.resolve(command())
        self.assertEqual(result['model'], {'class': 'Pi0Config', 'model_type_enum': 'PI0', 'model_type': 'pi0',
                                           'action_dim': 32, 'action_horizon': 10})
        self.assertFalse(result['normalization']['use_quantile_norm'])
        self.assertEqual(result['asset_id'], 'droid')
        self.assertEqual(result['statistics_reference'], CHECKPOINT + '/assets/droid/norm_stats.json')
        self.assertEqual(result['action_boundaries']['declared_model_action_width'], 32)
        self.assertEqual(result['action_boundaries']['final_DroidOutputs_slice_width'], 8)
        self.assertLess(result['source_bytes_read'], 2 * 1024 * 1024)
        self.assertEqual(result['parser_dependency_coverage']['supported_native_lock_tyro'], '0.9.22')
        self.assertEqual(result['parser_dependency_coverage']['actual_user_runtime'], 'uninspected')

    def test_pi05_model_type_derives_from_native_pi05_flag_and_factory_overwrites_base_false(self):
        result = self.resolve(command('pi05_droid'))
        self.assertEqual(result['model']['model_type_enum'], 'PI05')
        self.assertEqual(result['model']['action_horizon'], 15)
        self.assertEqual(result['model']['action_dim'], 32)
        self.assertIs(result['normalization']['base_config_flag'], False)
        self.assertIs(result['normalization']['use_quantile_norm'], True)

    def test_fast_model_property_and_output_decoder_dimension(self):
        result = self.resolve(command('pi0_fast_droid'))
        self.assertEqual(result['model']['model_type_enum'], 'PI0_FAST')
        self.assertEqual(result['model']['action_horizon'], 10)
        self.assertEqual(result['model']['action_dim'], 8)
        self.assertTrue(result['normalization']['use_quantile_norm'])
        self.assertIn('decodes', result['action_boundaries']['pre_inverse_width_basis'])

    def test_default_droid_is_pi05_checkpoint_not_pi0_base(self):
        result = self.resolve(BASE + ' --env=DROID')
        self.assertEqual(result['config_id'], 'pi05_droid')
        self.assertEqual(result['checkpoint_reference'], 'gs://openpi-assets/checkpoints/pi05_droid')
        self.assertNotEqual(result['checkpoint_reference'], CHECKPOINT)

    def test_explicit_checkpoint_ignores_environment_default(self):
        result = self.resolve(BASE + ' --env=DROID policy:checkpoint --policy.config=pi0_droid --policy.dir=' + CHECKPOINT)
        self.assertEqual(result['config_id'], 'pi0_droid')
        self.assertEqual(result['checkpoint_reference'], CHECKPOINT)

    def test_supported_space_syntax_and_explicit_default_route(self):
        self.assertEqual(self.resolve(BASE + ' --env DROID policy:default')['config_id'], 'pi05_droid')
        text = BASE + ' policy:checkpoint --policy.config pi0_droid --policy.dir ' + CHECKPOINT
        self.assertEqual(self.resolve(text)['config_id'], 'pi0_droid')

    def test_actual_output_order_keeps_native_inverse_before_robot_slice(self):
        result = self.resolve(command())
        output = result['transform_order']['output_transforms']
        self.assertEqual(output, ['data_config.model_transforms.outputs', 'Unnormalize',
                                  'data_config.data_transforms.outputs', 'repack_transforms.outputs'])
        self.assertEqual(result['historical_training_mode'], 'unknown')
        self.assertEqual(result['checkpoint_reference_immutability'], 'unassessed')

    def test_no_factory_import_network_or_checkpoint_params_open(self):
        opened = []
        original_open = os.open
        original_import = builtins.__import__

        def guarded_open(path, *args, **kwargs):
            self.assertNotIn('NOT_A_CHECKPOINT', str(path))
            self.assertNotIn('params', str(path))
            opened.append(str(path))
            return original_open(path, *args, **kwargs)

        def guarded_import(name, *args, **kwargs):
            self.assertFalse(name.startswith('openpi'))
            return original_import(name, *args, **kwargs)

        with patch('os.open', guarded_open), patch('builtins.__import__', guarded_import), \
                patch.object(socket, 'create_connection', side_effect=AssertionError('network attempted')):
            result = self.resolve(command(checkpoint='/NOT_A_CHECKPOINT'))
        self.assertEqual(result['statistics_reference'], '/NOT_A_CHECKPOINT/assets/droid/norm_stats.json')
        self.assertTrue(opened)

    def test_unknown_and_unsupported_commands_refuse_without_raw_values(self):
        examples = [command('custom_untrusted'), BASE, command() + ' --use-quantiles=true',
                    BASE + '\npolicy:checkpoint --policy.config=pi0_droid --policy.dir=' + CHECKPOINT,
                    command(checkpoint='/tmp/NOT_A\x00PATH'), command() + '\r',
                    command() + ' --policy.api_key=FAKE_SECRET_MUST_NOT_RENDER',
                    command() + ' --policy.dir=/duplicate', command(checkpoint='$UNRESOLVED/path'),
                    command(checkpoint='https://user:FAKE_SECRET_MUST_NOT_RENDER@example.com/model'),
                    command(checkpoint='gs://openpi-assets/path?token=FAKE_SECRET_MUST_NOT_RENDER'),
                    command() + ' --env=DROID',
                    BASE + ' --policy.config=pi0_droid policy:checkpoint --policy.dir=' + CHECKPOINT,
                    BASE + ' --env=NOT_A_NATIVE_ENUM policy:checkpoint --policy.config=pi0_droid --policy.dir=' + CHECKPOINT,
                    command() + ' --env=NOT_A_NATIVE_ENUM', command() + ' --env=droid', command() + '; echo BAD', command() + ';', command() + ' policy:default',
                    BASE + ' --env=DROID --policy.config=pi0_droid --policy.dir=' + CHECKPOINT,
                    'python arbitrary_config.py']
        for text in examples:
            with self.subTest(text_hash=hashlib.sha256(text.encode()).hexdigest()):
                result = resolve_serving_command(text, SOURCE)
                self.assertEqual(result['status'], 'unknown')
                self.assertNotIn('FAKE_SECRET_MUST_NOT_RENDER', json.dumps(result))
                self.assertNotIn('normalization', result)

    def test_aggregate_budget_refusal_reads_no_body_when_first_file_exceeds_remaining(self):
        with patch.object(openpi_interface, 'MAX_TOTAL', 1), patch('os.read', side_effect=AssertionError('body read over budget')):
            result = resolve_serving_command(command(), SOURCE)
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['reason'], 'source byte limit exceeded')

    def test_malformed_api_inputs_refuse_without_stringifying_caller_objects(self):
        class Untrusted:
            def __str__(self):
                raise AssertionError('caller object was stringified')

        for value in [None, 4, Untrusted(), '\ud800']:
            result = resolve_serving_command(value, SOURCE)
            self.assertEqual(result['status'], 'unknown')
            self.assertIsNone(result['command_sha256'])
        for value in [None, {}, Untrusted()]:
            result = resolve_serving_command(command(), value)
            self.assertEqual(result['status'], 'unknown')
            self.assertIsNone(result['command_sha256'])

    def copy_source(self, target):
        for relative in MANIFEST['files']:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / relative, destination)

    def test_source_fingerprint_drift_does_not_evaluate_injected_code(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            target = Path(directory)
            self.copy_source(target)
            path = target / 'src/openpi/training/config.py'
            path.write_text(path.read_text() + '\nraise RuntimeError("MUST_NOT_EXECUTE")\n')
            result = resolve_serving_command(command(), target)
            self.assertEqual(result['status'], 'unknown')
            self.assertEqual(result['reason'], 'source fingerprint drift')
            self.assertNotIn('MUST_NOT_EXECUTE', json.dumps(result))

    def test_source_symlink_is_refused(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            target = Path(directory)
            self.copy_source(target)
            path = target / 'src/openpi/training/config.py'
            path.unlink()
            path.symlink_to(SOURCE / 'src/openpi/training/config.py')
            result = resolve_serving_command(command(), target)
            self.assertEqual(result['status'], 'unknown')

    def test_source_root_ancestor_symlink_is_refused_before_any_body_read(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            link = Path(directory) / 'ancestor'
            link.symlink_to(SOURCE.parent, target_is_directory=True)
            with patch('os.read', side_effect=AssertionError('outside source body read')):
                result = resolve_serving_command(command(), link / SOURCE.name)
            self.assertEqual(result['status'], 'unknown')

    def test_fifo_source_leaf_is_opened_nonblocking_and_refused_before_read(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            target = Path(directory)
            self.copy_source(target)
            leaf = target / 'scripts/serve_policy.py'
            leaf.unlink()
            os.mkfifo(leaf)
            original = os.open

            def guarded_open(path, flags, *args, **kwargs):
                if str(path) == 'serve_policy.py':
                    self.assertTrue(flags & os.O_NONBLOCK)
                return original(path, flags, *args, **kwargs)

            with patch('os.open', guarded_open), patch('os.read', side_effect=AssertionError('FIFO body read')):
                result = resolve_serving_command(command(), target)
            self.assertEqual(result['status'], 'unknown')
            self.assertEqual(result['reason'], 'source file size or type refused')

    def test_checkpoint_reference_is_data_and_does_not_authenticate_asset_contents(self):
        result = self.resolve(command(checkpoint='gs://a-public-bucket/changed-checkpoint'))
        self.assertEqual(result['statistics_reference'], 'gs://a-public-bucket/changed-checkpoint/assets/droid/norm_stats.json')
        self.assertEqual(result['checkpoint_reference_immutability'], 'unassessed')
        self.assertNotIn('statistics_validated', result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
