"""Unattended failure must return evidence without starting scientific training."""
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
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
            result = subprocess.run(['bash', str(runner), '--worker', str(bundle), str(output)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 13)
            self.assertEqual((output/'exit-code.txt').read_text().strip(), '13')
            self.assertFalse((output/'training').exists())
            with tarfile.open(str(output)+'.tar.gz') as archive:
                receipt = archive.extractfile('returned evidence/RETURN.txt').read().decode()
                self.assertIn('FAILED (exit 13)', receipt)
                self.assertIn('do not restart training', receipt)


if __name__ == '__main__':
    unittest.main()
