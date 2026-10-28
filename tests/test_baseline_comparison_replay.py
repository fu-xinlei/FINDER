"""Check the archived two-process fold schedule without loading a graph or GPU."""

import ast
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "finder/baseline_comparison/c02_model_train.py"
LAUNCHER = ROOT / "scripts/run_finder_baseline_comparison.sh"


def selected_folds(environment):
    """Execute real pre-loop setup/guards, substituting only expensive fold work."""
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    function = copy.deepcopy(next(node for node in tree.body
                                 if isinstance(node, ast.FunctionDef)
                                 and node.name == "k_fold_cross_validation"))
    loop_index = next(index for index, node in enumerate(function.body)
                      if isinstance(node, ast.For))
    loop = function.body[loop_index]
    # The first status print marks entry into expensive per-fold training.
    training_start = next(index for index, node in enumerate(loop.body)
                          if isinstance(node, ast.Expr)
                          and isinstance(node.value, ast.Call)
                          and isinstance(node.value.func, ast.Name)
                          and node.value.func.id == "print")
    loop.body = loop.body[:training_start] + ast.parse("record(fold)").body
    function.body = function.body[:loop_index + 1]
    function.args = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                                  kw_defaults=[], defaults=[])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(
        numpy=lambda: SimpleNamespace(T=None)))
    splitter = SimpleNamespace(split=lambda *_: [(None, None)] * 10)
    rng = random.Random(93)
    recorded = []
    scope = {"os": os, "k": 10, "train_edges": tensor, "train_labels": tensor,
             "StratifiedKFold": lambda **_: splitter,
             "record": lambda fold: recorded.append((fold + 1, rng.random()))}
    with patch.dict(os.environ, environment, clear=True):
        exec(compile(module, str(TRAINER), "exec"), scope)
        scope["k_fold_cross_validation"]()
    return recorded


class Table2ReplayTest(unittest.TestCase):
    def test_fold_ranges_preserve_rng_restart_before_fold_five(self):
        first = selected_folds({"FINDER_FOLD_START": "1", "FINDER_FOLD_END": "4"})
        last = selected_folds({"FINDER_FOLD_START": "5", "FINDER_FOLD_END": "10"})
        all_folds = selected_folds({})
        self.assertEqual([fold for fold, _ in first], [1, 2, 3, 4])
        self.assertEqual([fold for fold, _ in last], [5, 6, 7, 8, 9, 10])
        self.assertEqual([fold for fold, _ in all_folds], list(range(1, 11)))
        self.assertEqual(first[0][1], last[0][1])
        self.assertNotEqual(last[0][1], all_folds[4][1])

    def test_invalid_fold_bounds_are_rejected(self):
        for start, end in (("0", "4"), ("5", "4"), ("1", "11"),
                           ("not-an-integer", "10")):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                selected_folds({"FINDER_FOLD_START": start, "FINDER_FOLD_END": end})

    def test_launcher_starts_two_processes_with_archived_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            log = temporary / "calls.jsonl"
            recorder = temporary / "python-recorder"
            recorder.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "keys = ['FINDER_FOLD_START', 'FINDER_FOLD_END', 'FINDER_OUTPUT_DIR', "
                "'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', "
                "'MPLBACKEND', 'FINDER_RESUME']\n"
                "with open(os.environ['REPLAY_CALL_LOG'], 'a') as stream:\n"
                "    stream.write(json.dumps({'pid': os.getpid(), 'argv': sys.argv[1:], "
                "'env': {key: os.environ.get(key) for key in keys}}) + '\\n')\n",
                encoding="utf-8",
            )
            recorder.chmod(0o755)
            run_root = temporary / "outputs with spaces"
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith("FINDER_")}
            environment.update(PYTHON_BIN=str(recorder), RUN_ROOT=str(run_root),
                               REPLAY_CALL_LOG=str(log), DEVICE="cpu", RESUME="never",
                               OMP_NUM_THREADS="32", OPENBLAS_NUM_THREADS="32",
                               MKL_NUM_THREADS="32", MPLBACKEND="TkAgg")
            result = subprocess.run(["bash", str(LAUNCHER)], env=environment,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 2)
            self.assertNotEqual(calls[0]["pid"], calls[1]["pid"])
            for call, start, end, directory in zip(
                calls, ("1", "5"), ("4", "10"), ("folds1to4", "folds5to10")
            ):
                self.assertEqual(call["argv"], ["finder/baseline_comparison/c02_model_train.py"])
                self.assertEqual(call["env"]["FINDER_FOLD_START"], start)
                self.assertEqual(call["env"]["FINDER_FOLD_END"], end)
                self.assertEqual(call["env"]["FINDER_OUTPUT_DIR"], str(run_root / directory))
                for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                    self.assertEqual(call["env"][variable], "2")
                self.assertEqual(call["env"]["MPLBACKEND"], "Agg")
                self.assertEqual(call["env"]["FINDER_RESUME"], "never")


if __name__ == "__main__":
    unittest.main()
