"""Deterministic MVD evidence publication replay and checksum verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .artifacts import publish_m0, publish_m1
from .config import load_mvd_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(directory: str | Path) -> None:
    """Fail if any listed artifact byte differs from its issued checksum."""

    root = Path(directory).resolve(strict=True)
    checksum_path = root / "CHECKSUMS.sha256"
    if not checksum_path.is_file():
        raise ValueError("checksum file is missing")
    observed: set[str] = set()
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        digest, separator, name = line.partition("  ")
        path = root / name
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError("artifact checksum path escaped package") from error
        if (
            separator != "  "
            or len(digest) != 64
            or not path.is_file()
            or _sha256(path) != digest
            or name in observed
        ):
            raise ValueError("artifact checksum verification failed")
        observed.add(name)
    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    }
    if not observed or observed != expected:
        raise ValueError("artifact checksum roster changed")


def _file_map(path: Path) -> dict[str, str]:
    return {
        file.relative_to(path).as_posix(): _sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def replay_mvd_packages(
    config_path: str | Path, *, project_root: str | Path
) -> Path:
    """Republish both formal gates and require byte-identical packages."""

    root = Path(project_root).resolve(strict=True)
    config = load_mvd_config(config_path, project_root=root)
    replay = root / config.output_replay
    replay.mkdir(parents=True, exist_ok=True)
    formal_m0 = root / config.output_m0
    formal_m1 = root / config.output_m1
    replay_m0 = replay / "m0_one_shot_oracle"
    replay_m1 = replay / "m1_observability"
    publish_m0(config_path, project_root=root, output_path=replay_m0)
    publish_m1(config_path, project_root=root, output_path=replay_m1)
    maps = {
        "m0": (_file_map(formal_m0), _file_map(replay_m0)),
        "m1": (_file_map(formal_m1), _file_map(replay_m1)),
    }
    for formal, reproduced in maps.values():
        if formal != reproduced:
            raise ValueError("MVD replay package is not byte-identical")
    verify_checksums(formal_m0)
    verify_checksums(formal_m1)
    verify_checksums(replay_m0)
    verify_checksums(replay_m1)
    summary = {
        "schema_version": 1,
        "replay_verified": True,
        "m0_file_count": len(maps["m0"][0]),
        "m1_file_count": len(maps["m1"][0]),
        "m0_package_sha256": hashlib.sha256(
            json.dumps(maps["m0"][0], sort_keys=True).encode("ascii")
        ).hexdigest(),
        "m1_package_sha256": hashlib.sha256(
            json.dumps(maps["m1"][0], sort_keys=True).encode("ascii")
        ).hexdigest(),
    }
    (replay / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return replay


__all__ = ["replay_mvd_packages", "verify_checksums"]
