"""Strict adapters for frozen CAI MPa and feature authorities."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def load_mpa_targets(
    path: str | Path, *, allowed_keys: set[str] | None = None
) -> dict[str, float]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError("CAI MPa authority is unavailable") from error
    required = {"domain_id", "specimen_id", "cai_strength_mpa"}
    if not rows or not required <= set(rows[0]):
        raise ValueError("CAI MPa authority schema is invalid")
    targets: dict[str, float] = {}
    for row in rows:
        key = f"{row['domain_id']}:{row['specimen_id']}"
        if allowed_keys is not None and key not in allowed_keys:
            continue
        if key in targets:
            raise ValueError("CAI MPa authority contains duplicate specimen")
        try:
            value = float(row["cai_strength_mpa"])
        except (TypeError, ValueError) as error:
            raise ValueError("CAI MPa authority value is invalid") from error
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("CAI MPa authority value is invalid")
        targets[key] = value
    return targets


__all__ = ["load_mpa_targets"]
