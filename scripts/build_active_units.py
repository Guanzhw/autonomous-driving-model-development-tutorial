"""Regenerate delivered units from the same catalog used by execution checks."""

import argparse
import subprocess
import sys
from course_catalog import ACTIVE_UNITS, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", choices=["all", *ACTIVE_UNITS], default="all")
    args = parser.parse_args()
    names = ACTIVE_UNITS if args.unit == "all" else [args.unit]
    for name in names:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / ACTIVE_UNITS[name]["generator"])],
            check=True,
        )


if __name__ == "__main__":
    main()
