"""Execute the active unit (or archived material) using this Python interpreter."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy", action="store_true", help="execute 11 archived chapters then 3 labs"
    )
    args = parser.parse_args()
    if args.legacy:
        base = ROOT / "reference" / "legacy"
        paths = sorted((base / "course").glob("*.ipynb")) + sorted(
            (base / "labs").glob("*.ipynb")
        )
    else:
        paths = sorted((ROOT / "course" / "first_loop").glob("*.ipynb"))
    if not paths:
        raise RuntimeError("no notebooks found")
    destination = (
        ROOT / "artifacts" / "executed" / ("legacy" if args.legacy else "first_loop")
    )
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        client = NotebookClient(
            notebook,
            timeout=300,
            km=manager,
            resources={"metadata": {"path": str(ROOT)}},
        )
        start = time.perf_counter()
        try:
            client.execute(env={**os.environ, "MPLBACKEND": "Agg"})
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            nbformat.write(notebook, destination / path.name)
        print(
            f"PASS {path.relative_to(ROOT)} ({time.perf_counter() - start:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
