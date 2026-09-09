"""Shared utilities for the urban cut-in tutorial.

The package deliberately stays small and dependency-light.  The notebooks are
the lesson; this package only prevents every lesson from inventing a different
scene, coordinate convention, or artifact format.
"""

from .scene import (
    ARTIFACT_DIR,
    BEVConfig,
    UrbanCutInScene,
    build_bev_dataset,
    build_urban_cut_in_scene,
    ensure_artifact_dir,
    load_json_artifact,
    load_numpy_artifact,
    rasterize_points,
    save_json_artifact,
    save_numpy_artifact,
    scene_to_bev,
)

__all__ = [
    "ARTIFACT_DIR",
    "BEVConfig",
    "UrbanCutInScene",
    "build_bev_dataset",
    "build_urban_cut_in_scene",
    "ensure_artifact_dir",
    "load_json_artifact",
    "load_numpy_artifact",
    "rasterize_points",
    "save_json_artifact",
    "save_numpy_artifact",
    "scene_to_bev",
    "UrbanCutInScene",
]
