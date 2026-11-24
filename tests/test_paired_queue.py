"""The fixed pair must preserve a failed arm and still attempt its paired control."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec=importlib.util.spec_from_file_location('paired_queue',Path(__file__).resolve().parents[1]/'scripts/run_paired_queue.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


class PairedQueueTests(unittest.TestCase):
    def test_pair_failure_is_retained_without_retry_and_next_arm_benchmarked(self):
        with tempfile.TemporaryDirectory(dir='/tmp/.agents') as temporary:
            root=Path(temporary);(root/'output/mac').mkdir(parents=True)
            protocol={'output':'output','training':{},'nodes':{'mac':{'device':'cpu'}}}
            calls=[]
            def train(experiment, protocol, manifest, **kwargs):
                calls.append(experiment['dataset'])
                if experiment['dataset']=='random4096': raise RuntimeError('preserved test failure')
                manifest.append({'trained_epochs':40})
            benchmark=Mock(return_value={'models':{'mh1_path4096_s27':{'aggregate':{'barrier_mae_eV':.1}}}})
            fake=SimpleNamespace(train=train,benchmark=benchmark)
            with patch.object(queue,'ROOT',root),patch.dict('sys.modules',{
                    'torch':SimpleNamespace(set_num_threads=lambda n:None),'run_model_study':fake}):
                queue.worker(protocol,'mac',27)
                with self.assertRaises(FileExistsError): queue.worker(protocol,'mac',27)
            result=json.loads((root/'output/mac/pair-27/state.json').read_text())
            self.assertEqual(calls,['random4096','path4096'])
            self.assertEqual(result['status'],'completed_with_failures')
            self.assertEqual([x['status'] for x in result['outcomes']],['failed','completed'])
            self.assertFalse(result['holdout_labels_evaluated'])
            self.assertEqual([x.kwargs['role'] for x in benchmark.call_args_list],['validation','meta_development'])

    def test_truncated_epoch_budget_is_failed_before_any_benchmark(self):
        with tempfile.TemporaryDirectory(dir='/tmp/.agents') as temporary:
            root=Path(temporary);(root/'output/mac').mkdir(parents=True)
            def train(experiment, protocol, manifest, **kwargs): manifest.append({'trained_epochs':3})
            benchmark=Mock()
            with patch.object(queue,'ROOT',root),patch.dict('sys.modules',{
                    'torch':SimpleNamespace(set_num_threads=lambda n:None),
                    'run_model_study':SimpleNamespace(train=train,benchmark=benchmark)}):
                queue.worker({'output':'output','training':{},'nodes':{'mac':{'device':'cpu'}}},'mac',27)
            benchmark.assert_not_called()
            state=json.loads((root/'output/mac/pair-27/state.json').read_text())
            self.assertEqual([x['status'] for x in state['outcomes']],['failed','failed'])

    def test_frozen_assignments_have_no_seed_overlap(self):
        protocol=json.loads((Path(__file__).resolve().parents[1]/'configs/overnight-20261124.json').read_text())
        mac=set(protocol['nodes']['mac']['seeds']);omen=set(protocol['nodes']['omen']['seeds'])
        self.assertFalse(mac & omen)
        self.assertFalse((mac|omen) & {20261116,20261117,20261118})
        self.assertEqual(len(mac|omen),7)
        self.assertEqual(protocol['training']['epochs'],40)
        self.assertEqual(protocol['training']['patience'],100)


if __name__=='__main__':unittest.main()
