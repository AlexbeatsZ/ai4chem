"""Deployment boundaries: portable inputs, immutable budget and explicit device."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('cloud_worker',Path(__file__).resolve().parents[1]/'scripts/cloud_worker.py')
cloud=importlib.util.module_from_spec(spec); spec.loader.exec_module(cloud)


class CloudWorkerTests(unittest.TestCase):
    def test_relative_inputs_relocate_with_bundle_and_cannot_be_overridden(self):
        with tempfile.TemporaryDirectory(dir='/tmp/.agents') as temporary:
            root=Path(temporary)
            for name in ['train.xyz','valid.xyz','foundation.model']: (root/name).touch()
            task={'name':'fixed_task','train_file':'train.xyz','valid_file':'valid.xyz',
                  'foundation_model':'foundation.model','expected_epochs':40,'settings':{'max_num_epochs':40}}
            result=cloud.resolve_inputs(task,root/'task.json')
            self.assertEqual(result['train_file'],str((root/'train.xyz').resolve()))
            task['settings']['train_file']='other.xyz'
            with self.assertRaisesRegex(ValueError,'protected'): cloud.resolve_inputs(task,root/'task.json')

    def test_reported_epoch_budget_cannot_differ_from_submitted_training(self):
        task={'name':'fixed_task','expected_epochs':2,'settings':{'max_num_epochs':40}}
        with self.assertRaisesRegex(ValueError,'budget'): cloud.resolve_inputs(task,Path('/tmp/.agents/task.json'))

    def test_missing_cuda_never_silently_falls_back_to_cpu(self):
        fake=SimpleNamespace(version=SimpleNamespace(cuda='13.4'),cuda=SimpleNamespace(is_available=lambda:False))
        with patch.dict('sys.modules',{'torch':fake}),patch.object(cloud,'installed',return_value={}):
            with self.assertRaisesRegex(RuntimeError,'no automatic CPU fallback'): cloud.doctor('cuda')


if __name__=='__main__': unittest.main()
