from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


class A100NotebookTests(unittest.TestCase):
    def test_notebook_code_cells_compile_and_include_overnight_contract(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "Koemi-3HIP_A100.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(4, notebook["nbformat"])
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 3)
        for cell_index, cell in enumerate(code_cells):
            compile("".join(cell["source"]), f"{notebook_path}:cell-{cell_index}", "exec")
        notebook_source = "\n".join("".join(cell["source"]) for cell in code_cells)
        self.assertIn("run_internal_contract_tests()", notebook_source)
        self.assertIn("session_seconds=9 * 60 * 60", notebook_source)
        self.assertIn("checkpoint_interval_seconds=15 * 60", notebook_source)
        self.assertIn("A100_RUN_SOURCE", notebook_source)
        source_assignment = next(
            node
            for node in ast.walk(ast.parse(notebook_source))
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "A100_RUN_SOURCE" for target in node.targets)
        )
        embedded_runner = ast.literal_eval(source_assignment.value)
        module_runner = (notebook_path.parent.parent / "src" / "koemi" / "training" / "a100_run.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(module_runner.rstrip(), embedded_runner.rstrip())


if __name__ == "__main__":
    unittest.main()
