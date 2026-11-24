"""The fixed pair must preserve a failed arm and still attempt its paired control."""
import importlib.util
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec=importlib.util.spec_from_file_location('paired_queue',Path(__file__).resolve().parents[1]/'scripts/run_paired_queue.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


class PairedQueueTests(unittest.TestCase):
    def test_native_takeover_keeps_live_workers_and_expires_without_duplicate_arms(self):
        with tempfile.TemporaryDirectory(dir='/tmp/.agents') as temporary:
            root=Path(temporary).resolve();(root/'scripts').mkdir()
            runner=root/'scripts/run_paired_queue.py'
            # Shorten scheduler polling only; use actual processes, locks and signals.
            source=Path(queue.__file__).read_text().replace('if active: time.sleep(30)',
                                                           'if active: time.sleep(.1)')
            runner.write_text(source)
            (root/'torch.py').write_text('def set_num_threads(n): pass\n')
            (root/'run_model_study.py').write_text('''import os, subprocess, sys
from pathlib import Path
def train(experiment, protocol, manifest, **kwargs):
    with open('attempts.txt', 'a') as trace:
        trace.write(experiment['name']+'\\n')
    subprocess.run([sys.executable,'-c','import time; time.sleep(2)'],check=True)
    manifest.append({'trained_epochs':40})
def benchmark(name, models, checkpoints, **kwargs):
    return {'models':{key:{'aggregate':{}} for key in models}}
''')
            protocol=root/'protocol.json'
            protocol.write_text(json.dumps({'output':'output','training':{},
                'nodes':{'mac':{'device':'cpu','workers':2,'seeds':[27,28,29,30]}}}))
            execution=root/'execution.json'
            execution.write_text(json.dumps({'boost_until_utc':
                (datetime.now(timezone.utc)+timedelta(seconds=2)).isoformat(),
                'boost_workers':4,'baseline_workers':2,'after_window_nice':10}))
            invocation=[sys.executable,'-u',str(runner),'--protocol',str(protocol),'--node','mac']
            state_path=root/'output/mac/state.json'
            processes=[];last_state={}
            def wait_for(predicate):
                nonlocal last_state
                deadline=time.monotonic()+8
                while time.monotonic()<deadline:
                    if state_path.exists():
                        last_state=json.loads(state_path.read_text())
                        if predicate(last_state): return last_state
                    time.sleep(.05)
                self.fail(str(last_state)+'\n'+(root/'console.log').read_text())
            with (root/'console.log').open('w') as log:
                try:
                    original=subprocess.Popen(invocation,cwd=root,stdout=log,stderr=log)
                    processes.append(original)
                    state=wait_for(lambda x:len(x.get('active',{}))==2)
                    original_children=dict(state['active'])
                    replacement=subprocess.Popen(invocation+['--adopt-owner',str(original.pid),
                        '--execution',str(execution)],cwd=root,stdout=log,stderr=log)
                    processes.append(replacement)
                    state=wait_for(lambda x:x['owner_pid']==replacement.pid and len(x['active'])==4)
                    self.assertEqual({key:state['active'][key] for key in original_children},original_children)
                    state=wait_for(lambda x:x.get('worker_nice')==10)
                    self.assertEqual(state['admission_capacity'],2)
                    for pid in state['active'].values():
                        self.assertGreaterEqual(os.getpriority(os.PRIO_PROCESS,pid),10)
                    replacement.wait(timeout=8);original.wait(timeout=2)
                    self.assertEqual(replacement.returncode,0,(root/'console.log').read_text())
                    state=json.loads(state_path.read_text())
                    self.assertEqual(state['status'],'completed')
                    self.assertEqual(len(state['outcomes']),4)
                    attempts=(root/'attempts.txt').read_text().splitlines()
                    self.assertEqual(len(attempts),8)
                    self.assertEqual(len(set(attempts)),8)
                    self.assertTrue((root/f'output/mac/handoff-from-{original.pid}.json').exists())
                finally:
                    for pid in last_state.get('active',{}).values():
                        try:
                            if str(root) in (queue.identity(pid) or ''): os.kill(pid,9)
                        except ProcessLookupError: pass
                    for process in processes:
                        if process.poll() is None: process.kill()
                        process.wait()

    def test_temporary_boost_expires_at_exact_utc8_deadline(self):
        execution={'boost_until_utc':'2026-11-25T07:00:00+00:00',
                   'boost_workers':4,'baseline_workers':2,'after_window_nice':10}
        self.assertEqual(queue.execution_policy(execution,2,
            datetime(2026,11,25,6,59,59,tzinfo=timezone.utc)),(4,0))
        self.assertEqual(queue.execution_policy(execution,2,
            datetime(2026,11,25,7,0,tzinfo=timezone.utc)),(2,10))
        with self.assertRaises(ValueError):
            queue.execution_policy({**execution,'boost_workers':5},2)
        with self.assertRaises(ValueError):
            queue.execution_policy({**execution,'boost_until_utc':'2026-11-25T07:00:00'},2)

    def test_priority_drop_includes_nested_trainer_but_not_other_work(self):
        active={27:SimpleNamespace(pid=100,poll=lambda:None)}
        priorities={100:0,101:0,102:15}
        with patch.object(queue.subprocess,'check_output',return_value='100 1\n101 100\n102 101\n999 1\n'), \
             patch.object(queue.os,'getpriority',side_effect=lambda kind,pid:priorities[pid]), \
             patch.object(queue.os,'setpriority') as change:
            queue.lower_priority(active,10)
        self.assertEqual({x.args[1:3] for x in change.call_args_list},
                         {(100,10),(101,10),(102,15)})

    def test_mismatched_owner_is_never_stopped(self):
        with patch.object(queue.subprocess,'check_output',return_value='unrelated program'), \
             patch.object(queue.os,'kill') as kill:
            with self.assertRaises(RuntimeError):
                queue.adopt_owner(123,Path('/unused'),Path('protocol.json'),'mac')
        kill.assert_not_called()

    def test_pair_failure_is_retained_without_retry_and_next_arm_benchmarked(self):
        with tempfile.TemporaryDirectory(dir='/tmp/.agents') as temporary:
            root=Path(temporary);(root/'output/mac').mkdir(parents=True)
            protocol={'output':'output','training':{},'nodes':{'mac':{'device':'cpu'}}}
            calls=[]
            def train(experiment, protocol, manifest, **kwargs):
                self.assertEqual(protocol['training_seed'],experiment['seed'])
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
