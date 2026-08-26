"""Checksum-bound deployable historical policy sources for MAVIS baselines."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import polars as pl

from cmc_bbdm.mva.measurement_state import RefinementAction

from .state_bank import PlannedAction


class MAVISHistoricalSourceError(ValueError):
    """Raised when a frozen historical policy artifact changes."""


_A4_COLUMNS = {
    "specimen_id",
    "dataset_id",
    "outer_domain",
    "method",
    "ranking_position",
    "cell_index",
    "from_level",
    "to_level",
    "nominal_checkpoint",
}
_A5_COLUMNS = {
    "specimen_id",
    "dataset_id",
    "outer_domain",
    "method",
    "step",
    "cell_index",
    "from_level",
    "to_level",
    "nominal_checkpoint",
}
_MVD_COLUMNS = {
    "outer_domain",
    "specimen_id",
    "dataset_id",
    "method",
    "cell_index",
    "predicted_value",
}


def _sha(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MAVISHistoricalSourceError("historical source hash is invalid")
    return value


def _table(path: str | Path, expected: str, columns: set[str]) -> pl.DataFrame:
    source = Path(path)
    try:
        payload = source.read_bytes()
        table = pl.read_parquet(source)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise MAVISHistoricalSourceError("historical source is unavailable") from error
    if hashlib.sha256(payload).hexdigest() != _sha(expected) or not columns <= set(
        table.columns
    ):
        raise MAVISHistoricalSourceError("historical source hash or schema changed")
    return table


class HistoricalPolicySource:
    def __init__(
        self,
        *,
        a4_path: str | Path,
        a4_sha256: str,
        a5_path: str | Path,
        a5_sha256: str,
        mvd_m1_path: str | Path,
        mvd_m1_sha256: str,
        checkpoints: tuple[float, ...],
    ) -> None:
        if (
            type(checkpoints) is not tuple
            or not checkpoints
            or tuple(sorted(checkpoints)) != checkpoints
            or len(set(checkpoints)) != len(checkpoints)
        ):
            raise MAVISHistoricalSourceError("historical checkpoint roster is invalid")
        self._a4 = _table(a4_path, a4_sha256, _A4_COLUMNS)
        self._a5 = _table(a5_path, a5_sha256, _A5_COLUMNS)
        self._mvd = _table(mvd_m1_path, mvd_m1_sha256, _MVD_COLUMNS).select(
            *sorted(_MVD_COLUMNS)
        )
        self._checkpoints = checkpoints
        self.state_sha256 = hashlib.sha256(
            f"{a4_sha256}{a5_sha256}{mvd_m1_sha256}".encode("ascii")
        ).hexdigest()

    @staticmethod
    def _identity(
        table: pl.DataFrame,
        *,
        specimen_id: str,
        dataset_id: str,
        outer_domain: str,
        method: str,
    ) -> pl.DataFrame:
        if any(
            type(value) is not str or not value
            for value in (specimen_id, dataset_id, outer_domain, method)
        ):
            raise MAVISHistoricalSourceError("historical identity is invalid")
        return table.filter(
            (pl.col("specimen_id") == specimen_id)
            & (pl.col("dataset_id") == dataset_id)
            & (pl.col("outer_domain") == outer_domain)
            & (pl.col("method") == method)
        )

    def _plan(
        self,
        table: pl.DataFrame,
        *,
        order_column: str,
        specimen_id: str,
        dataset_id: str,
        outer_domain: str,
        method: str,
    ) -> tuple[PlannedAction, ...]:
        selected = self._identity(
            table,
            specimen_id=specimen_id,
            dataset_id=dataset_id,
            outer_domain=outer_domain,
            method=method,
        ).sort(order_column)
        if (
            selected.height == 0
            or selected.get_column(order_column).to_list()
            != list(range(selected.height))
            or any(
                float(value) not in self._checkpoints
                for value in selected.get_column("nominal_checkpoint")
            )
        ):
            raise MAVISHistoricalSourceError("historical action plan is incomplete")
        try:
            return tuple(
                PlannedAction(
                    action=RefinementAction(
                        cell_index=int(row["cell_index"]),
                        from_level=int(row["from_level"]),
                        to_level=int(row["to_level"]),
                    ),
                    nominal_checkpoint=float(row["nominal_checkpoint"]),
                )
                for row in selected.iter_rows(named=True)
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise MAVISHistoricalSourceError("historical action is invalid") from error

    def action_plans(
        self,
        *,
        specimen_id: str,
        dataset_id: str,
        outer_domain: str,
    ) -> Mapping[str, tuple[PlannedAction, ...]]:
        plans = {
            "global_mechanical": self._plan(
                self._a4,
                order_column="ranking_position",
                specimen_id=specimen_id,
                dataset_id=dataset_id,
                outer_domain=outer_domain,
                method="global_mechanical_mask",
            ),
            "mva_a5": self._plan(
                self._a5,
                order_column="step",
                specimen_id=specimen_id,
                dataset_id=dataset_id,
                outer_domain=outer_domain,
                method="imitation_policy",
            ),
        }
        return MappingProxyType(plans)

    def o2_scores(
        self,
        *,
        specimen_id: str,
        dataset_id: str,
        outer_domain: str,
    ) -> np.ndarray:
        selected = self._identity(
            self._mvd,
            specimen_id=specimen_id,
            dataset_id=dataset_id,
            outer_domain=outer_domain,
            method="o2_global_candidate",
        ).sort("cell_index")
        if (
            selected.height != 64
            or selected.get_column("cell_index").to_list() != list(range(64))
        ):
            raise MAVISHistoricalSourceError("historical O2 cell roster is incomplete")
        scores = np.ascontiguousarray(
            selected.get_column("predicted_value").to_numpy(),
            dtype="<f8",
        )
        if scores.shape != (64,) or not np.all(np.isfinite(scores)):
            raise MAVISHistoricalSourceError("historical O2 scores are invalid")
        output = np.frombuffer(scores.tobytes(order="C"), dtype="<f8")
        output.setflags(write=False)
        return output


__all__ = ["HistoricalPolicySource", "MAVISHistoricalSourceError"]
