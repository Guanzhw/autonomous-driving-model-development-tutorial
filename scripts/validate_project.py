"""Check notebook syntax, entry points and repository-local documentation links."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import nbformat


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "course" / "first_loop"
ARCHIVE = ROOT / "reference" / "legacy"


def check_links(source: str, directory: Path, label: str) -> None:
    links = re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", source)
    links += re.findall(r"""(?:href|src)=["']([^"']+)["']""", source)
    for link in links:
        parts = urlsplit(link)
        if parts.scheme or parts.netloc or not parts.path:
            continue
        target = directory / unquote(parts.path)
        if not target.exists():
            raise AssertionError(f"{label}: broken local link {link}")


def main() -> None:
    active = sorted(ACTIVE.glob("*.ipynb"))
    archived = sorted(ARCHIVE.rglob("*.ipynb"))
    if len(active) != 2 or len(archived) != 14:
        raise AssertionError(
            f"expected 2 active + 14 archived notebooks, got {len(active)} + {len(archived)}"
        )
    if list((ROOT / "course").glob("*.ipynb")) or list((ROOT / "labs").glob("*.ipynb")):
        raise AssertionError("old notebooks remain outside the archive")
    for path in active + archived:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=f"{path}:{cell.id}")
            elif cell.cell_type == "markdown":
                check_links(cell.source, path.parent, f"{path}:{cell.id}")
        if path in archived and "归档材料" not in notebook.cells[0].source:
            raise AssertionError(f"{path}: missing archive notice")

    documents = list(ROOT.glob("*.md")) + [ROOT / "index.html"]
    for directory in ("course", "reference", "labs", "notebooks", "review"):
        documents.extend((ROOT / directory).rglob("*.md"))
    for path in documents:
        check_links(path.read_text(encoding="utf-8"), path.parent, str(path))
    for base in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
        for path in base.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    unit_index = (ACTIVE / "README.md").read_text(encoding="utf-8")
    archive_index = (ARCHIVE / "README.md").read_text(encoding="utf-8")
    for path in active:
        if path.name not in unit_index:
            raise AssertionError(f"active unit index misses {path.name}")
    for path in archived:
        if path.name not in archive_index:
            raise AssertionError(f"archive index misses {path.name}")
    print(
        f"PASS: {len(active)} active + {len(archived)} archived notebooks, Python syntax, local links and indexes"
    )


if __name__ == "__main__":
    main()
