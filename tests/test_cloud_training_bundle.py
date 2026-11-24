"""A relocated timing bundle keeps labels/budget and strips prior output paths."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_cloud', ROOT/'scripts/prepare_cloud_training.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BundleTests(unittest.TestCase):
    def test_relocated_bundle_is_bounded_and_preserves_training_settings(self):
        with tempfile.TemporaryDirectory(dir='/tmp/.agents') as directory:
            root = Path(directory)
            config = {'name': 'existing_development', 'seed': 42, 'max_num_epochs': 40,
                      'forces_weight': 10, 'default_dtype': 'float64', 'device': 'cpu',
                      'model_dir': '/old/models', 'restart_latest': True}
            for key in ('train_file', 'valid_file', 'foundation_model'):
                path = root/key
                path.write_text(key)
                config[key] = str(path)
            source = root/'config.yaml'
            source.write_text(yaml.safe_dump(config))
            bundle = root/'bundle'
            module.prepare(source, bundle)
            moved = root/'relocated'
            bundle.rename(moved)
            task = json.loads((moved/'task.json').read_text())
            for key in ('train_file', 'valid_file', 'foundation_model'):
                self.assertEqual((moved/task[key]).read_text(), key)
            self.assertEqual(task['settings']['max_num_epochs'], 2)
            self.assertEqual(task['settings']['forces_weight'], 10)
            self.assertEqual(task['settings']['device'], 'cuda')
            self.assertNotIn('model_dir', task['settings'])
            self.assertNotIn('restart_latest', task['settings'])
            with self.assertRaises(FileExistsError):
                module.prepare(source, moved)


if __name__ == '__main__':
    unittest.main()
