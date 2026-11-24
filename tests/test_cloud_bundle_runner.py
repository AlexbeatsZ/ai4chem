"""Unattended failure must return evidence without starting scientific training."""
from pathlib import Path
import shutil
import os
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def test_wsl_requires_native_owner_before_claiming_submission(self):
        with tempfile.TemporaryDirectory(dir='/tmp/.agents') as temporary:
            root=Path(temporary)
            bundle=root/'bundle'; bundle.mkdir(); (bundle/'task.json').write_text('{}')
            binary=root/'bin'; binary.mkdir()
            uname=binary/'uname'; uname.write_text('#!/bin/sh\necho 6.18-microsoft-standard-WSL2\n'); uname.chmod(0o755)
            environment={**os.environ,'PATH':str(binary)+os.pathsep+os.environ['PATH']}
            output=root/'output'
            result=subprocess.run(['bash',str(ROOT/'scripts/run_cloud_training_bundle.sh'),str(bundle),str(output)],
                                  capture_output=True,text=True,env=environment,timeout=10)
            self.assertEqual(result.returncode,2)
            self.assertIn('native Windows owner',result.stderr)
            self.assertFalse(output.exists())


    def test_install_failure_is_archived_without_training_even_with_spaces(self):
        with tempfile.TemporaryDirectory(dir='/tmp/.agents') as temporary:
            root = Path(temporary)/'project with spaces'
            scripts = root/'scripts'
            scripts.mkdir(parents=True)
            runner = scripts/'run_cloud_training_bundle.sh'
            shutil.copyfile(ROOT/'scripts'/runner.name, runner)
            (scripts/'cloud_worker.sh').write_text('#!/bin/bash\necho "simulated download failure"\nexit 13\n')
            bundle = root/'input bundle'
            bundle.mkdir()
            (bundle/'task.json').write_text('{}')
            output = root/'returned evidence'
            output.mkdir()
            checkpoint=output/'checkpoints'; checkpoint.mkdir(); (checkpoint/'optimizer.pt').write_text('not a return artifact')
            result = subprocess.run(['bash', str(runner), '--worker', str(bundle), str(output)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 13)
            self.assertEqual((output/'exit-code.txt').read_text().strip(), '13')
            self.assertFalse((output/'training').exists())
            with tarfile.open(str(output)+'.tar.gz') as archive:
                receipt = archive.extractfile('returned evidence/RETURN.txt').read().decode()
                self.assertFalse(any('/checkpoints/' in name for name in archive.getnames()))
                self.assertIn('FAILED (exit 13)', receipt)
                self.assertIn('do not restart training', receipt)


if __name__ == '__main__':
    unittest.main()
