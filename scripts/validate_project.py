"""Fast structural validation for the tutorial repository.

This is intentionally lighter than executing every notebook. It is suitable for
pull requests and catches broken nbformat, Python syntax, expected notebook
links, and missing maintenance artifacts. Full code-cell execution remains a
local/release check because the learned-model notebooks may require PyTorch.
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
EXPECTED_NOTEBOOKS = 20


def notebook_links(text: str) -> set[str]:
    links = re.findall(r"(?:notebooks/)?([\w-]+\.ipynb)", text)
    return set(links)


def main() -> None:
    warnings.filterwarnings("ignore", category=nbformat.validator.MissingIDFieldWarning)
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if len(notebooks) != EXPECTED_NOTEBOOKS:
        raise AssertionError(f"expected {EXPECTED_NOTEBOOKS} notebooks, found {len(notebooks)}")

    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=f"{path}:{index}")

    expected_links = {path.name for path in notebooks}
    for document in (ROOT / "README.md", ROOT / "notebooks" / "README.md", ROOT / "index.html"):
        missing = expected_links - notebook_links(document.read_text(encoding="utf-8"))
        if missing:
            raise AssertionError(f"{document}: missing notebook links: {sorted(missing)}")

    required = [
        ROOT / "PROJECT_REFERENCE.md",
        ROOT / "AGENTS.md",
        ROOT / "review" / "README.md",
        ROOT / "review" / "roles" / "reviewer.md",
        ROOT / "review" / "roles" / "devils-advocate.md",
    ]
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"missing maintenance artifacts: {missing}")

    print(f"PASS: {len(notebooks)} notebooks have valid nbformat and Python syntax")
    print("PASS: README, Notebook Track, and HTML link to every notebook")
    print("PASS: project reference and dual-role review artifacts exist")


if __name__ == "__main__":
    main()
