"""Byte-identical replay of a validated MVA A2 evidence package."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .artifacts import MVAPackageValidation, validate_mva_package


def replay_mva_package(
    source: str | Path,
    destination: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> MVAPackageValidation:
    """Validate, reproduce, and independently revalidate an MVA package."""

    source_path = Path(source).resolve(strict=True)
    original = validate_mva_package(
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
            target = staging / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copyfile(item, target)
        if destination_path.exists():
            shutil.rmtree(destination_path)
        os.replace(staging, destination_path)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    replayed = validate_mva_package(
        destination_path, project_root=project_root, config_path=config_path
    )
    if replayed != original:
        raise RuntimeError("MVA replay digests differ from the validated source")
    return replayed


__all__ = ["replay_mva_package"]
