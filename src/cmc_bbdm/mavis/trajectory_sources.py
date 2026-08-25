"""Load checksum-bound frozen MAVIS source action plans."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import polars as pl

from cmc_bbdm.mva.measurement_state import RefinementAction

from .config import MAVISConfig
from .state_bank import PlannedAction


class MAVISTrajectorySourceError(ValueError):
    """Raised when a frozen MAVIS action source violates its contract."""


_A2_METHODS = (
    ("random", "random"),
    ("uniform", "uniform"),
    ("reconstruction_driven", "reconstruction_oracle"),
    ("sequential_mechanical_oracle", "mechanical_oracle"),
)
_A2_COLUMNS = {
    "record_type",
    "specimen_id",
    "dataset_id",
    "method",
    "seed",
    "step",
    "nominal_checkpoint",
    "cell_index",
    "from_level",
    "to_level",
}
_MVD_COLUMNS = {
    "specimen_id",
    "dataset_id",
    "method",
    "step",
    "nominal_checkpoint",
    "cell_index",
    "from_level",
    "to_level",
}


def _bound_source_path(
    config: MAVISConfig, project_root: str | Path, source_name: str
) -> Path:
    try:
        binding = config.sources[source_name]
    except (KeyError, TypeError) as error:
        raise MAVISTrajectorySourceError(
            f"source binding {source_name} is unavailable"
        ) from error
    root = Path(project_root)
    try:
        path = root / binding.path
        payload = path.read_bytes()
    except (OSError, TypeError) as error:
        raise MAVISTrajectorySourceError(
            f"source {source_name} is unavailable"
        ) from error
    actual = hashlib.sha256(payload).hexdigest()
    if actual != binding.sha256:
        raise MAVISTrajectorySourceError(f"source {source_name} hash changed")
    return path


def _read_parquet(
    path: Path, source_name: str, required_columns: set[str]
) -> pl.DataFrame:
    try:
        table = pl.read_parquet(path)
    except Exception as error:
        raise MAVISTrajectorySourceError(
            f"source {source_name} cannot be read"
        ) from error
    if not required_columns <= set(table.columns):
        raise MAVISTrajectorySourceError(f"source {source_name} schema changed")
    return table


def _check_identity(
    table: pl.DataFrame,
    specimen_id: str,
    dataset_id: str,
    source_name: str,
) -> None:
    if table.height == 0:
        raise MAVISTrajectorySourceError(f"source {source_name} has an empty group")
    if (
        table.get_column("specimen_id").n_unique() != 1
        or table.get_column("dataset_id").n_unique() != 1
        or table.get_column("specimen_id").item(0) != specimen_id
        or table.get_column("dataset_id").item(0) != dataset_id
    ):
        raise MAVISTrajectorySourceError(f"source {source_name} identity changed")
    steps = table.get_column("step").to_list()
    if steps != list(range(table.height)):
        raise MAVISTrajectorySourceError(f"source {source_name} step sequence changed")


def _actions(
    table: pl.DataFrame,
    *,
    config: MAVISConfig,
    specimen_id: str,
    dataset_id: str,
    source_name: str,
) -> tuple[PlannedAction, ...]:
    _check_identity(table, specimen_id, dataset_id, source_name)
    actions: list[PlannedAction] = []
    for row in table.iter_rows(named=True):
        checkpoint = float(row["nominal_checkpoint"])
        if checkpoint not in config.checkpoints:
            raise MAVISTrajectorySourceError(
                f"source {source_name} checkpoint is not registered"
            )
        try:
            action = RefinementAction(
                cell_index=int(row["cell_index"]),
                from_level=int(row["from_level"]),
                to_level=int(row["to_level"]),
            )
            actions.append(
                PlannedAction(action=action, nominal_checkpoint=checkpoint)
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise MAVISTrajectorySourceError(
                f"source {source_name} action is invalid"
            ) from error
    if not actions:
        raise MAVISTrajectorySourceError(f"source {source_name} has an empty group")
    return tuple(actions)


def load_frozen_action_plans(
    config: MAVISConfig,
    project_root: str | Path,
    specimen_id: str,
    dataset_id: str,
) -> Mapping[str, tuple[PlannedAction, ...]]:
    """Load the registered A2 and MVD action plans for one exact specimen/domain."""

    return FrozenActionPlanSource(config, project_root=project_root).plans(
        specimen_id,
        dataset_id,
    )


class FrozenActionPlanSource:
    """Read the two checksum-bound trajectory tables once per execution worker."""

    def __init__(self, config: MAVISConfig, *, project_root: str | Path) -> None:
        if type(config) is not MAVISConfig:
            raise MAVISTrajectorySourceError("trajectory source config is invalid")
        a2_path = _bound_source_path(config, project_root, "a2_oracle_trajectories")
        mvd_path = _bound_source_path(config, project_root, "mvd_m0_actions")
        self._config = config
        self._a2 = _read_parquet(
            a2_path,
            "a2_oracle_trajectories",
            _A2_COLUMNS,
        )
        self._mvd = _read_parquet(mvd_path, "mvd_m0_actions", _MVD_COLUMNS)

    def plans(
        self,
        specimen_id: str,
        dataset_id: str,
    ) -> Mapping[str, tuple[PlannedAction, ...]]:
        config = self._config
        if (
            type(specimen_id) is not str
            or not specimen_id
            or type(dataset_id) is not str
            or not dataset_id
        ):
            raise MAVISTrajectorySourceError("trajectory source inputs are invalid")

        a2 = self._a2
        mvd = self._mvd
        a2_identity = (pl.col("specimen_id") == specimen_id) & (
            pl.col("dataset_id") == dataset_id
        )
        mvd_identity = (pl.col("specimen_id") == specimen_id) & (
            pl.col("dataset_id") == dataset_id
        )
        plans: dict[str, tuple[PlannedAction, ...]] = {}
        for output_method, source_method in _A2_METHODS:
            selection = a2.filter(
                a2_identity
                & (pl.col("record_type") == "action")
                & (pl.col("method") == source_method)
            )
            if source_method == "random":
                selection = selection.filter(
                    pl.col("seed") == config.trajectory_random_seed
                )
            plans[output_method] = _actions(
                selection,
                config=config,
                specimen_id=specimen_id,
                dataset_id=dataset_id,
                source_name="a2_oracle_trajectories",
            )

        mvd_selection = mvd.filter(
            mvd_identity & (pl.col("method") == "one_shot_mechanical_oracle")
        )
        plans["one_shot_mechanical_oracle"] = _actions(
            mvd_selection,
            config=config,
            specimen_id=specimen_id,
            dataset_id=dataset_id,
            source_name="mvd_m0_actions",
        )
        return MappingProxyType(plans)


__all__ = [
    "FrozenActionPlanSource",
    "MAVISTrajectorySourceError",
    "load_frozen_action_plans",
]
