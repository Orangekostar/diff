"""Deterministic rebuild and tree-hash replay for formal MVA A5."""

from __future__ import annotations

from pathlib import Path

from .a5_artifacts import (
    A5PackageValidation,
    aggregate_a5_shards,
    finalize_a5_package,
    validate_a5_package,
)
from .a5_figures import render_a5_figures
from .a5_report import render_a5_report


class A5ReplayError(ValueError):
    """Raised when an A5 replay differs from the formal package."""


def replay_a5_package(
    formal_dir: str | Path,
    replay_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> A5PackageValidation:
    """Rebuild A5 from validated outer shards and require an identical tree hash."""

    root = Path(project_root).resolve(strict=True)
    formal = validate_a5_package(
        formal_dir, project_root=root, config_path=config_path
    )
    work = aggregate_a5_shards(config_path, project_root=root)
    render_a5_figures(work, config_path=config_path, project_root=root)
    render_a5_report(work, config_path=config_path, project_root=root)
    replay = finalize_a5_package(
        work,
        replay_dir,
        project_root=root,
        config_path=config_path,
    )
    if replay.output_tree_sha256 != formal.output_tree_sha256:
        raise A5ReplayError("A5 replay output tree differs from formal evidence")
    return replay


__all__ = ["A5ReplayError", "replay_a5_package"]
