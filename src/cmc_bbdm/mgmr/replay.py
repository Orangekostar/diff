"""Byte-identical replay of a fully validated formal M0 evidence package."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .artifacts import M0PackageValidation, validate_m0_package


def replay_m0_package(
    source: str | Path,
    destination: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> M0PackageValidation:
    """Validate, reproduce the byte tree, and independently validate the replay."""

    source_path = Path(source).resolve(strict=True)
    original = validate_m0_package(
        source_path, project_root=project_root, config_path=config_path
    )
    destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}.staging-", dir=destination_path.parent
        )
    )
    try:
        for item in source_path.iterdir():
            shutil.copyfile(item, staging / item.name)
        if destination_path.exists():
            shutil.rmtree(destination_path)
        os.replace(staging, destination_path)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    replayed = validate_m0_package(
        destination_path, project_root=project_root, config_path=config_path
    )
    if replayed != original:
        raise RuntimeError("M0 replay digests differ from the validated source")
    return replayed


__all__ = ["replay_m0_package"]
