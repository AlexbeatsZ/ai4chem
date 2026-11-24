"""A queued candidate is neither a lost owner nor an exportable terminal job."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
spec=importlib.util.spec_from_file_location('gaussian_relay',Path(__file__).resolve().parents[1]/'scripts/continue_gaussian_relay.py')
relay=importlib.util.module_from_spec(spec);spec.loader.exec_module(relay)


class QueuedRelayTests(unittest.TestCase):
    def test_queued_job_waits_without_export_or_owner_missing(self):
        with tempfile.TemporaryDirectory(dir='/tmp/.agents') as temporary:
            root=Path(temporary);protocol=root/'protocol.json'
            protocol.write_text(json.dumps({'jobs':[{'job_id':'queued1'}],'frozen_at_utc':'fixed'}))
            observed={'state':{'job_id':'queued1','status':'queued','raw_chain_completed':False},'owner_alive':False}
            argv=['relay','--relay-ssh','test@relay','--relay-root','C:/relay','--worker-root','C:/worker',
                  '--worker-job','C:/worker/job','--protocol',str(protocol),'--output',str(root/'output'),'--max-hours','1']
            with patch.object(sys,'argv',argv),patch.object(relay.time,'monotonic',side_effect=[0,1,4000]), \
                 patch.object(relay.time,'sleep'),patch.object(relay.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(observed))) as command:
                relay.main()
            self.assertEqual(command.call_count,1)
            self.assertEqual(command.call_args.args[0][command.call_args.args[0].index('-Action')+1],'Status')
            state=json.loads((root/'output/state.json').read_text())
            self.assertEqual(state['status'],'stopped_deadline')
            self.assertNotIn('connection_error',state)
            self.assertFalse((root/'output/return.zip').exists())


if __name__=='__main__':unittest.main()
