"""Read-only identity binding to the frozen MVA A2 authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .contracts import PRIMARY_COUNTS

A2_SHA256 = "6b289f2f6f74ac75dde47ea7cbfefcda1c49f025e74227bfb34ef269182ff963"
_A2_RELATIVE_PATH = Path("results/mva/a2_oracle_value/oracle_values.parquet")


class FrozenBindingError(ValueError):
    """Raised when a frozen identity authority has drifted."""


@dataclass(frozen=True, slots=True)
class FrozenA2Binding:
    relative_path: str
    sha256: str
    columns: tuple[str, ...]
    specimen_count: int
    initial_row_count: int
    cell_ids: tuple[int, ...]
    domain_counts: tuple[tuple[str, int], ...]
    from_level: int
    to_level: int
    nominal_checkpoint: float
    predictor_state_hashes: tuple[str, ...]
    identity_state_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "columns": list(self.columns),
            "specimen_count": self.specimen_count,
            "initial_row_count": self.initial_row_count,
            "cell_ids": list(self.cell_ids),
            "domain_counts": dict(self.domain_counts),
            "from_level": self.from_level,
            "to_level": self.to_level,
            "nominal_checkpoint": self.nominal_checkpoint,
            "predictor_state_hashes": list(self.predictor_state_hashes),
            "identity_state_sha256": self.identity_state_sha256,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_frozen_a2(project_root: str | Path) -> FrozenA2Binding:
    """Verify A2 and inspect only specimen/domain/cell identity columns."""

    root = Path(project_root).resolve(strict=True)
    path = root / _A2_RELATIVE_PATH
    if path.is_symlink() or not path.is_file() or _sha256(path) != A2_SHA256:
        raise FrozenBindingError("frozen A2 authority SHA-256 changed")
    try:
        schema = pl.read_parquet_schema(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise FrozenBindingError("frozen A2 schema cannot be read") from error
    required = {
        "specimen_id",
        "dataset_id",
        "method",
        "step",
        "nominal_checkpoint",
        "cell_index",
        "from_level",
        "to_level",
        "candidate",
        "p_a_predictor_state_sha256",
    }
    if not required <= set(schema):
        raise FrozenBindingError("frozen A2 schema is missing identity columns")
    try:
        identities = (
            pl.scan_parquet(path)
            .filter(
                (pl.col("method") == "mechanical_oracle")
                & (pl.col("step") == 0)
            )
            .select(
                "specimen_id",
                "dataset_id",
                "nominal_checkpoint",
                "cell_index",
                "from_level",
                "to_level",
                "candidate",
                "p_a_predictor_state_sha256",
            )
            .collect()
        )
    except pl.exceptions.PolarsError as error:
        raise FrozenBindingError("frozen A2 identity rows cannot be read") from error
    if identities.is_duplicated().any():
        raise FrozenBindingError("frozen A2 initial identity rows are duplicated")
    cells = tuple(identities["cell_index"].unique().sort().to_list())
    if cells != tuple(range(64)):
        raise FrozenBindingError("frozen A2 action cells changed")
    if not identities.select((pl.col("candidate") == pl.col("cell_index")).all()).item():
        raise FrozenBindingError("frozen A2 candidate-cell identity changed")
    levels = identities.select("from_level", "to_level").unique()
    checkpoints = identities["nominal_checkpoint"].unique().sort().to_list()
    if levels.rows() != [(0, 1)] or checkpoints != [0.0625]:
        raise FrozenBindingError("frozen A2 post-scout state semantics changed")
    predictor_hashes = tuple(
        identities["p_a_predictor_state_sha256"].unique().sort().to_list()
    )
    if len(predictor_hashes) != 6 or any(
        type(value) is not str
        or len(value) != 64
        or set(value) - set("0123456789abcdef")
        for value in predictor_hashes
    ):
        raise FrozenBindingError("frozen A2 predictor-state identities changed")
    per_specimen = identities.group_by("specimen_id", "dataset_id").len()
    if per_specimen.height != 276 or set(per_specimen["len"].to_list()) != {64}:
        raise FrozenBindingError("frozen A2 specimen-cell roster changed")
    domain_rows = (
        per_specimen.group_by("dataset_id")
        .len()
        .sort("dataset_id")
        .iter_rows(named=True)
    )
    domain_counts = tuple((row["dataset_id"], row["len"]) for row in domain_rows)
    if dict(domain_counts) != dict(PRIMARY_COUNTS):
        raise FrozenBindingError("frozen A2 domain roster changed")
    per_specimen_states = identities.group_by("specimen_id", "dataset_id").agg(
        pl.col("p_a_predictor_state_sha256").n_unique().alias("state_n")
    )
    if set(per_specimen_states["state_n"].to_list()) != {1}:
        raise FrozenBindingError("frozen A2 specimen state identity changed")
    identity_records = identities.sort("dataset_id", "specimen_id", "cell_index").rows()
    identity_payload = json.dumps(
        identity_records,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return FrozenA2Binding(
        relative_path=_A2_RELATIVE_PATH.as_posix(),
        sha256=A2_SHA256,
        columns=tuple(schema),
        specimen_count=per_specimen.height,
        initial_row_count=identities.height,
        cell_ids=cells,
        domain_counts=domain_counts,
        from_level=0,
        to_level=1,
        nominal_checkpoint=0.0625,
        predictor_state_hashes=predictor_hashes,
        identity_state_sha256=hashlib.sha256(identity_payload).hexdigest(),
    )


__all__ = [
    "A2_SHA256",
    "FrozenA2Binding",
    "FrozenBindingError",
    "bind_frozen_a2",
]
