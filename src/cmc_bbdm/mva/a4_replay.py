"""Byte-identical replay of a validated MVA A4 evidence package."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .a4_artifacts import A4PackageValidation, validate_a4_package


def replay_a4_package(
    source: str | Path,
    destination: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> A4PackageValidation:
    """Validate, copy transactionally, and independently revalidate A4."""

    source_path = Path(source).resolve(strict=True)
    original = validate_a4_package(
        source_path, project_root=project_root, config_path=config_path
    )
    destination_path = Path(destination).resolve()
    if destination_path == source_path:
        raise ValueError("A4 replay destination must differ from its source")
    if destination_path.exists():
        existing = validate_a4_package(
            destination_path, project_root=project_root, config_path=config_path
        )
        if existing != original:
            raise RuntimeError("existing A4 replay differs from its source")
        return existing
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}.staging-", dir=destination_path.parent
        )
    )
    try:
        for item in source_path.iterdir():
            target = staging / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copyfile(item, target)
        staged = validate_a4_package(
            staging, project_root=project_root, config_path=config_path
        )
        if staged != original:
            raise RuntimeError("staged A4 replay digests differ from its source")
        os.replace(staging, destination_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    replayed = validate_a4_package(
        destination_path, project_root=project_root, config_path=config_path
    )
    if replayed != original:
        raise RuntimeError("A4 replay digests differ from the validated source")
    return replayed


__all__ = ["replay_a4_package"]
