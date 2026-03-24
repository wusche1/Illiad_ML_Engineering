"""Execute every lecture notebook with solutions injected, verify all tests pass."""
import importlib.util
import sys
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent.parent
LECTURES = ROOT / "lectures"


def discover_exercises():
    """Find exercise dirs that have both notebook.ipynb and solutions.py."""
    dirs = sorted(p.parent for p in LECTURES.glob("*/exercises/*/solutions.py"))
    return [(d, f"{d.parent.parent.name}/{d.name}") for d in dirs if (d / "notebook.ipynb").exists()]


def load_solutions(exercise_dir: Path) -> dict:
    spec = importlib.util.spec_from_file_location("solutions", exercise_dir / "solutions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SOLUTIONS


@pytest.mark.parametrize("exercise_dir", [d for d, _ in discover_exercises()], ids=[n for _, n in discover_exercises()])
def test_notebook(exercise_dir):
    solutions = load_solutions(exercise_dir)
    nb = nbformat.read(exercise_dir / "notebook.ipynb", as_version=4)

    for cell in nb.cells:
        eid = cell.metadata.get("exercise_id")
        if eid and eid in solutions:
            cell.source = solutions[eid]

    ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
    ep.preprocess(nb, resources={"metadata": {"path": str(exercise_dir)}})

    for i, cell in enumerate(nb.cells):
        for output in cell.get("outputs", []):
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            if "FAIL" in text:
                pytest.fail(f"Cell {i} output contains FAIL:\n{text}")
