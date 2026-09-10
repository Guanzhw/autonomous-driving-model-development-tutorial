"""Registered, delivered units shared by generation, execution and validation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_UNITS = {
    "first_loop": {
        "generator": "build_first_unit.py",
        "notebooks": ["01_drive_and_observe.ipynb", "02_delay_and_recovery.ipynb"],
    },
    "state_estimation": {
        "generator": "build_state_estimation.py",
        "notebooks": [
            "03_frames_measurements.ipynb",
            "04_state_estimation_closed_loop.ipynb",
        ],
    },
}


def notebook_paths(unit="all"):
    names = ACTIVE_UNITS if unit == "all" else [unit]
    return [
        ROOT / "course" / name / notebook
        for name in names
        for notebook in ACTIVE_UNITS[name]["notebooks"]
    ]
