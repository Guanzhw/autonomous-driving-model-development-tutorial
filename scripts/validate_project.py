"""Check notebook syntax, entry points and repository-local documentation links."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import nbformat
from course_catalog import ACTIVE_UNITS, notebook_paths


ROOT = Path(__file__).resolve().parents[1]
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
    active = notebook_paths()
    archived = sorted(ARCHIVE.rglob("*.ipynb"))
    if len(archived) != 14:
        raise AssertionError(f"expected 14 archived notebooks, got {len(archived)}")
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
    documents.append(ROOT / "course" / "README.md")
    for name in ACTIVE_UNITS:
        documents.extend((ROOT / "course" / name).rglob("*.md"))
    for directory in ("reference", "labs", "notebooks", "review"):
        documents.extend((ROOT / directory).rglob("*.md"))
    for path in documents:
        check_links(path.read_text(encoding="utf-8"), path.parent, str(path))
    for base in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
        for path in base.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    archive_index = (ARCHIVE / "README.md").read_text(encoding="utf-8")
    for path in active:
        unit_index = (path.parent / "README.md").read_text(encoding="utf-8")
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
