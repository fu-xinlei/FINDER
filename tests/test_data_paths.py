"""Check portable processed-input paths without loading graph dependencies."""

import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_runtime(protocol):
    path = ROOT / "finder" / protocol / "origin_runtime.py"
    spec = importlib.util.spec_from_file_location(f"finder_{protocol}_paths", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DataPathTest(unittest.TestCase):
    def setUp(self):
        self.runtimes = {
            protocol: load_runtime(protocol)
            for protocol in ("baseline_comparison", "preprocessing")
        }

    def test_default_uses_the_sibling_data_repository(self):
        with patch.dict(os.environ):
            os.environ.pop("FINDER_DATA_DIR", None)
            expected = (ROOT.parent / "data" / "inputs" / "finder").resolve()
            for protocol, runtime in self.runtimes.items():
                with self.subTest(protocol=protocol):
                    self.assertEqual(runtime.get_data_dir(), expected)

    def test_relative_override_is_resolved_from_the_code_root(self):
        with patch.dict(os.environ, {"FINDER_DATA_DIR": "../custom-data/inputs/finder"}):
            expected = (ROOT.parent / "custom-data" / "inputs" / "finder").resolve()
            for protocol, runtime in self.runtimes.items():
                with self.subTest(protocol=protocol):
                    self.assertEqual(runtime.get_data_dir(), expected)

    def test_absolute_override_is_used(self):
        expected = (ROOT / "external-inputs").resolve()
        with patch.dict(os.environ, {"FINDER_DATA_DIR": str(expected)}):
            for protocol, runtime in self.runtimes.items():
                with self.subTest(protocol=protocol):
                    self.assertEqual(runtime.get_data_dir(), expected)

    def test_paths_do_not_depend_on_the_working_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(ROOT / "finder")
            for configured in (None, "../custom-data/inputs/finder"):
                with patch.dict(os.environ):
                    if configured is None:
                        os.environ.pop("FINDER_DATA_DIR", None)
                        expected = ROOT.parent / "data" / "inputs" / "finder"
                    else:
                        os.environ["FINDER_DATA_DIR"] = configured
                        expected = ROOT.parent / "custom-data" / "inputs" / "finder"
                    for protocol, runtime in self.runtimes.items():
                        with self.subTest(protocol=protocol, override=configured):
                            self.assertEqual(runtime.get_data_dir(), expected.resolve())
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
