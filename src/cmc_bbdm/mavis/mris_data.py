"""Immutable P1-to-P2 feature bank with target-separated model inputs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import polars as pl

from .authority import MAVISAuthority
from .state_encoder import (
    MRISInput,
    build_mris_shuffle_mapping,
    build_shuffled_mris_input_from_input,
    summarize_mris_input,
)


class MAVISMRISDataError(ValueError):
    """Raised when the P1 state bank cannot issue leakage-safe P2 arrays."""


_REQUIRED_COLUMNS = {
    "state_id",
    "specimen_id",
    "domain_id",
    "trajectory_id",
    "method",
    "seed",
    "nominal_checkpoint",
    "exact_acquired_cost",
    "native_count",
    "effective_budget",
    "context_features",
    "revealed_rows",
    "revealed_columns",
    "revealed_red",
    "revealed_green",
    "revealed_blue",
    "teacher_outer_domains",
    "strict_oof_cai_predictions",
}
_MODES = frozenset(("static", "positions_only", "real", "shuffled"))
_CACHE_KEYS = {
    "schema_version",
    "domain_order",
    "state_ids",
    "specimen_ids",
    "domain_ids",
    "trajectory_ids",
    "methods",
    "seeds",
    "nominal_checkpoints",
    "exact_acquired_costs",
    "native_counts",
    "effective_budgets",
    "context_features",
    "real_token_features",
    "positions_token_features",
    "token_masks",
    "cost_features",
    "shuffled_token_features",
    "donor_specimen_ids",
    "donor_relaxations",
    "reconstruction_predictions",
    "targets",
    "input_state_sha256",
    "target_state_sha256",
}


def _readonly(value: object, *, dtype: object, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.ascontiguousarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise MAVISMRISDataError("MRIS feature array cannot be snapshotted") from error
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise MAVISMRISDataError("MRIS feature array is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class ModelInputArrays:
    context_features: np.ndarray
    token_features: np.ndarray
    token_masks: np.ndarray
    cost_features: np.ndarray


@dataclass(frozen=True, slots=True)
class MRISFeatureBank:
    domain_order: tuple[str, ...]
    state_ids: tuple[str, ...]
    specimen_ids: tuple[str, ...]
    domain_ids: tuple[str, ...]
    trajectory_ids: tuple[str, ...]
    methods: tuple[str, ...]
    seeds: tuple[int | None, ...]
    nominal_checkpoints: np.ndarray
    exact_acquired_costs: np.ndarray
    native_counts: np.ndarray
    effective_budgets: np.ndarray
    context_features: np.ndarray
    real_token_features: np.ndarray
    positions_token_features: np.ndarray
    token_masks: np.ndarray
    cost_features: np.ndarray
    shuffled_token_features: np.ndarray
    donor_specimen_ids: MappingProxyType
    donor_relaxations: MappingProxyType
    reconstruction_predictions: np.ndarray
    targets: np.ndarray
    input_state_sha256: str
    target_state_sha256: str

    @property
    def row_count(self) -> int:
        return len(self.state_ids)

    def model_inputs(self, mode: str, *, outer_domain: str) -> ModelInputArrays:
        if mode not in _MODES or outer_domain not in self.domain_order:
            raise MAVISMRISDataError("MRIS model input request is invalid")
        if mode == "static":
            token_features = np.zeros_like(self.real_token_features)
            token_masks = np.zeros_like(self.token_masks)
            cost_features = np.zeros_like(self.cost_features)
            cost_features[:, 1] = 1.0
            token_features.setflags(write=False)
            token_masks.setflags(write=False)
            cost_features.setflags(write=False)
        elif mode == "positions_only":
            token_features = self.positions_token_features
            token_masks = self.token_masks
            cost_features = self.cost_features
        elif mode == "real":
            token_features = self.real_token_features
            token_masks = self.token_masks
            cost_features = self.cost_features
        else:
            outer_index = self.domain_order.index(outer_domain)
            token_features = self.shuffled_token_features[outer_index]
            token_masks = self.token_masks
            cost_features = self.cost_features
        return ModelInputArrays(
            context_features=self.context_features,
            token_features=token_features,
            token_masks=token_masks,
            cost_features=cost_features,
        )


def _state_table(value: object) -> pl.DataFrame:
    if not isinstance(value, pl.DataFrame) or not _REQUIRED_COLUMNS <= set(
        value.columns
    ):
        raise MAVISMRISDataError("P1 state table schema changed")
    if value.height == 0 or value.get_column("state_id").n_unique() != value.height:
        raise MAVISMRISDataError("P1 state table roster is invalid")
    return value.sort(
        ["domain_id", "specimen_id", "method", "nominal_checkpoint", "state_id"]
    )


def _real_input(row: dict[str, object], authority: MAVISAuthority) -> MRISInput:
    specimen_id = str(row["specimen_id"])
    context = authority.policy_context(specimen_id)
    rows = np.asarray(row["revealed_rows"], dtype=np.int64)
    columns = np.asarray(row["revealed_columns"], dtype=np.int64)
    red = np.asarray(row["revealed_red"], dtype=np.uint8)
    green = np.asarray(row["revealed_green"], dtype=np.uint8)
    blue = np.asarray(row["revealed_blue"], dtype=np.uint8)
    count = rows.size
    if (
        columns.shape != (count,)
        or red.shape != (count,)
        or green.shape != (count,)
        or blue.shape != (count,)
        or count != int(row["exact_acquired_cost"])
        or context.native_count != int(row["native_count"])
        or not math.isclose(
            count / context.native_count,
            float(row["effective_budget"]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise MAVISMRISDataError("P1 state cost or measurement roster changed")
    row_context = np.asarray(row["context_features"], dtype=np.float64)
    if not np.array_equal(row_context, context.context_features):
        raise MAVISMRISDataError("P1 state context disagrees with authority")
    positions = np.column_stack((rows, columns))
    values = np.column_stack((red, green, blue))
    issued = authority._reveal_values(specimen_id, positions)
    if not np.array_equal(values, issued):
        raise MAVISMRISDataError("P1 revealed values disagree with authority")
    return MRISInput(
        specimen_id=specimen_id,
        mode="real",
        context_features=context.context_features,
        native_shape=context.native_shape,
        acquired_positions=positions,
        measurement_values=values,
        effective_budget=count / context.native_count,
        remaining_budget=1.0 - count / context.native_count,
        content_specimen_id=specimen_id,
    )


def _reconstruction_prediction(
    row: dict[str, object], outer_domain: str
) -> float:
    domains = tuple(str(value) for value in row["teacher_outer_domains"])
    predictions = np.asarray(row["strict_oof_cai_predictions"], dtype=np.float64)
    if (
        predictions.shape != (len(domains),)
        or len(set(domains)) != len(domains)
        or not np.all(np.isfinite(predictions))
    ):
        raise MAVISMRISDataError("strict-OOF reconstruction predictions changed")
    if str(row["domain_id"]) == outer_domain:
        if outer_domain in domains or predictions.size == 0:
            raise MAVISMRISDataError("target reconstruction control is not strict OOF")
        return float(np.mean(predictions, dtype=np.float64))
    if domains.count(outer_domain) != 1:
        raise MAVISMRISDataError("source reconstruction control is not strict OOF")
    return float(predictions[domains.index(outer_domain)])


def _hash_inputs(
    *,
    state_ids: tuple[str, ...],
    domain_order: tuple[str, ...],
    donors: MappingProxyType,
    arrays: tuple[np.ndarray, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema": 1,
                "state_ids": state_ids,
                "domain_order": domain_order,
                "donors": {key: donors[key] for key in domain_order},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for value in arrays:
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def build_mris_feature_bank(
    states: pl.DataFrame,
    *,
    authority: MAVISAuthority,
    domain_order: tuple[str, ...],
    shuffle_seed: int,
) -> MRISFeatureBank:
    table = _state_table(states)
    if (
        type(authority) is not MAVISAuthority
        or type(domain_order) is not tuple
        or len(domain_order) < 2
        or len(set(domain_order)) != len(domain_order)
        or set(domain_order) != set(authority.dataset_ids)
        or type(shuffle_seed) is not int
    ):
        raise MAVISMRISDataError("MRIS feature-bank request is invalid")
    row_count = table.height
    mapping_by_outer = {
        outer: build_mris_shuffle_mapping(
            authority,
            outer_domain=outer,
            seed=shuffle_seed,
        )
        for outer in domain_order
    }
    donor_lookup = {
        outer: {row.recipient_id: row.donor_id for row in mapping}
        for outer, mapping in mapping_by_outer.items()
    }
    relaxation_lookup = {
        outer: {row.recipient_id: row.relaxation for row in mapping}
        for outer, mapping in mapping_by_outer.items()
    }
    if any(set(mapping) != set(authority.specimen_ids) for mapping in donor_lookup.values()):
        raise MAVISMRISDataError("shuffle mapping does not cover the authority")

    state_ids: list[str] = []
    specimen_ids: list[str] = []
    domain_ids: list[str] = []
    trajectory_ids: list[str] = []
    methods: list[str] = []
    seeds: list[int | None] = []
    checkpoints: list[float] = []
    exact_costs: list[int] = []
    native_counts: list[int] = []
    effective: list[float] = []
    contexts = np.empty((row_count, 34), dtype=np.float64)
    real_tokens = np.empty((row_count, 64, 6), dtype=np.float32)
    masks = np.empty((row_count, 64), dtype=bool)
    costs = np.empty((row_count, 3), dtype=np.float64)
    shuffled = np.empty((len(domain_order), row_count, 64, 6), dtype=np.float32)
    reconstruction = np.empty((len(domain_order), row_count), dtype=np.float64)
    targets = np.empty(row_count, dtype=np.float64)

    for row_index, row in enumerate(table.iter_rows(named=True)):
        real = _real_input(row, authority)
        summary = summarize_mris_input(real)
        specimen_id = real.specimen_id
        domain_id = str(row["domain_id"])
        authority_index = authority.specimen_ids.index(specimen_id)
        if authority.dataset_ids[authority_index] != domain_id:
            raise MAVISMRISDataError("P1 state domain disagrees with authority")
        state_ids.append(str(row["state_id"]))
        specimen_ids.append(specimen_id)
        domain_ids.append(domain_id)
        trajectory_ids.append(str(row["trajectory_id"]))
        methods.append(str(row["method"]))
        seeds.append(None if row["seed"] is None else int(row["seed"]))
        checkpoints.append(float(row["nominal_checkpoint"]))
        exact_costs.append(int(row["exact_acquired_cost"]))
        native_counts.append(int(row["native_count"]))
        effective.append(float(row["effective_budget"]))
        contexts[row_index] = summary.context_features
        real_tokens[row_index] = summary.token_features
        masks[row_index] = summary.token_mask
        costs[row_index] = summary.cost_features
        targets[row_index] = authority.evaluation_view(specimen_id).true_cai
        for outer_index, outer_domain in enumerate(domain_order):
            donor_id = donor_lookup[outer_domain][specimen_id]
            shuffled_input = build_shuffled_mris_input_from_input(
                real,
                authority=authority,
                donor_specimen_id=donor_id,
            )
            shuffled[outer_index, row_index] = summarize_mris_input(
                shuffled_input
            ).token_features
            reconstruction[outer_index, row_index] = _reconstruction_prediction(
                row, outer_domain
            )

    frozen_contexts = _readonly(contexts, dtype="<f8", shape=contexts.shape)
    frozen_real = _readonly(real_tokens, dtype="<f4", shape=real_tokens.shape)
    positions = real_tokens.copy()
    positions[:, :, 2:5] = 0.0
    frozen_positions = _readonly(positions, dtype="<f4", shape=positions.shape)
    frozen_masks = _readonly(masks, dtype=bool, shape=masks.shape)
    frozen_costs = _readonly(costs, dtype="<f8", shape=costs.shape)
    frozen_shuffled = _readonly(shuffled, dtype="<f4", shape=shuffled.shape)
    frozen_reconstruction = _readonly(
        reconstruction,
        dtype="<f8",
        shape=reconstruction.shape,
    )
    frozen_targets = _readonly(targets, dtype="<f8", shape=targets.shape)
    frozen_checkpoints = _readonly(
        checkpoints,
        dtype="<f8",
        shape=(row_count,),
    )
    frozen_exact_costs = _readonly(
        exact_costs,
        dtype="<i8",
        shape=(row_count,),
    )
    frozen_native_counts = _readonly(
        native_counts,
        dtype="<i8",
        shape=(row_count,),
    )
    frozen_effective = _readonly(
        effective,
        dtype="<f8",
        shape=(row_count,),
    )
    donor_rows = MappingProxyType(
        {
            outer: tuple(donor_lookup[outer][specimen] for specimen in specimen_ids)
            for outer in domain_order
        }
    )
    relaxation_rows = MappingProxyType(
        {
            outer: tuple(
                relaxation_lookup[outer][specimen] for specimen in specimen_ids
            )
            for outer in domain_order
        }
    )
    frozen_arrays = (
        frozen_contexts,
        frozen_real,
        frozen_positions,
        frozen_masks,
        frozen_costs,
        frozen_shuffled,
        frozen_reconstruction,
        frozen_checkpoints,
        frozen_exact_costs,
        frozen_native_counts,
        frozen_effective,
    )
    input_state = _hash_inputs(
        state_ids=tuple(state_ids),
        domain_order=domain_order,
        donors=donor_rows,
        arrays=frozen_arrays,
    )
    target_digest = hashlib.sha256()
    target_digest.update(input_state.encode("ascii"))
    target_digest.update(frozen_targets.tobytes(order="C"))
    return MRISFeatureBank(
        domain_order=domain_order,
        state_ids=tuple(state_ids),
        specimen_ids=tuple(specimen_ids),
        domain_ids=tuple(domain_ids),
        trajectory_ids=tuple(trajectory_ids),
        methods=tuple(methods),
        seeds=tuple(seeds),
        nominal_checkpoints=frozen_checkpoints,
        exact_acquired_costs=frozen_exact_costs,
        native_counts=frozen_native_counts,
        effective_budgets=frozen_effective,
        context_features=frozen_contexts,
        real_token_features=frozen_real,
        positions_token_features=frozen_positions,
        token_masks=frozen_masks,
        cost_features=frozen_costs,
        shuffled_token_features=frozen_shuffled,
        donor_specimen_ids=donor_rows,
        donor_relaxations=relaxation_rows,
        reconstruction_predictions=frozen_reconstruction,
        targets=frozen_targets,
        input_state_sha256=input_state,
        target_state_sha256=target_digest.hexdigest(),
    )


def save_mris_feature_bank(bank: MRISFeatureBank, path: str | Path) -> None:
    if type(bank) is not MRISFeatureBank:
        raise MAVISMRISDataError("issued MRIS feature bank is required")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                schema_version=np.asarray(1, dtype=np.int64),
                domain_order=np.asarray(bank.domain_order, dtype=np.str_),
                state_ids=np.asarray(bank.state_ids, dtype=np.str_),
                specimen_ids=np.asarray(bank.specimen_ids, dtype=np.str_),
                domain_ids=np.asarray(bank.domain_ids, dtype=np.str_),
                trajectory_ids=np.asarray(bank.trajectory_ids, dtype=np.str_),
                methods=np.asarray(bank.methods, dtype=np.str_),
                seeds=np.asarray(
                    [-1 if value is None else value for value in bank.seeds],
                    dtype=np.int64,
                ),
                nominal_checkpoints=bank.nominal_checkpoints,
                exact_acquired_costs=bank.exact_acquired_costs,
                native_counts=bank.native_counts,
                effective_budgets=bank.effective_budgets,
                context_features=bank.context_features,
                real_token_features=bank.real_token_features,
                positions_token_features=bank.positions_token_features,
                token_masks=bank.token_masks,
                cost_features=bank.cost_features,
                shuffled_token_features=bank.shuffled_token_features,
                donor_specimen_ids=np.asarray(
                    [bank.donor_specimen_ids[domain] for domain in bank.domain_order],
                    dtype=np.str_,
                ),
                donor_relaxations=np.asarray(
                    [bank.donor_relaxations[domain] for domain in bank.domain_order],
                    dtype=np.str_,
                ),
                reconstruction_predictions=bank.reconstruction_predictions,
                targets=bank.targets,
                input_state_sha256=np.asarray(bank.input_state_sha256, dtype=np.str_),
                target_state_sha256=np.asarray(bank.target_state_sha256, dtype=np.str_),
            )
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, destination)
        checksum = destination.with_suffix(destination.suffix + ".sha256")
        checksum_descriptor, checksum_name = tempfile.mkstemp(
            prefix=f".{checksum.name}.",
            suffix=".tmp",
            dir=checksum.parent,
        )
        checksum_temporary = Path(checksum_name)
        try:
            with os.fdopen(checksum_descriptor, "wb") as handle:
                handle.write(f"{digest}\n".encode("ascii"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(checksum_temporary, checksum)
        finally:
            if checksum_temporary.exists():
                checksum_temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def _cached_array(
    archive: np.lib.npyio.NpzFile,
    name: str,
    *,
    dtype: object,
    shape: tuple[int, ...],
) -> np.ndarray:
    return _readonly(archive[name], dtype=dtype, shape=shape)


def load_mris_feature_bank(path: str | Path) -> MRISFeatureBank:
    source = Path(path)
    try:
        expected_file_sha = source.with_suffix(source.suffix + ".sha256").read_text(
            encoding="ascii"
        ).strip()
        if (
            len(expected_file_sha) != 64
            or any(value not in "0123456789abcdef" for value in expected_file_sha)
            or hashlib.sha256(source.read_bytes()).hexdigest() != expected_file_sha
        ):
            raise MAVISMRISDataError("MRIS feature-bank file hash changed")
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != _CACHE_KEYS or int(archive["schema_version"]) != 1:
                raise MAVISMRISDataError("MRIS feature-bank cache schema changed")
            domain_order = tuple(str(value) for value in archive["domain_order"])
            state_ids = tuple(str(value) for value in archive["state_ids"])
            specimen_ids = tuple(str(value) for value in archive["specimen_ids"])
            domain_ids = tuple(str(value) for value in archive["domain_ids"])
            trajectory_ids = tuple(str(value) for value in archive["trajectory_ids"])
            methods = tuple(str(value) for value in archive["methods"])
            seed_values = np.asarray(archive["seeds"], dtype=np.int64)
            input_state_sha256 = str(archive["input_state_sha256"])
            target_state_sha256 = str(archive["target_state_sha256"])
            row_count = len(state_ids)
            domain_count = len(domain_order)
            if (
                row_count == 0
                or domain_count < 2
                or len(set(domain_order)) != domain_count
                or len(set(state_ids)) != row_count
                or any(
                    len(values) != row_count
                    for values in (
                        specimen_ids,
                        domain_ids,
                        trajectory_ids,
                        methods,
                        seed_values,
                    )
                )
                or set(domain_ids) != set(domain_order)
                or len(input_state_sha256) != 64
                or len(target_state_sha256) != 64
            ):
                raise MAVISMRISDataError("MRIS feature-bank cache roster changed")
            checkpoints = _cached_array(
                archive,
                "nominal_checkpoints",
                dtype="<f8",
                shape=(row_count,),
            )
            exact_costs = _cached_array(
                archive,
                "exact_acquired_costs",
                dtype="<i8",
                shape=(row_count,),
            )
            native_counts = _cached_array(
                archive,
                "native_counts",
                dtype="<i8",
                shape=(row_count,),
            )
            effective = _cached_array(
                archive,
                "effective_budgets",
                dtype="<f8",
                shape=(row_count,),
            )
            contexts = _cached_array(
                archive,
                "context_features",
                dtype="<f8",
                shape=(row_count, 34),
            )
            real = _cached_array(
                archive,
                "real_token_features",
                dtype="<f4",
                shape=(row_count, 64, 6),
            )
            positions = _cached_array(
                archive,
                "positions_token_features",
                dtype="<f4",
                shape=(row_count, 64, 6),
            )
            masks = _cached_array(
                archive,
                "token_masks",
                dtype=bool,
                shape=(row_count, 64),
            )
            costs = _cached_array(
                archive,
                "cost_features",
                dtype="<f8",
                shape=(row_count, 3),
            )
            shuffled = _cached_array(
                archive,
                "shuffled_token_features",
                dtype="<f4",
                shape=(domain_count, row_count, 64, 6),
            )
            reconstruction = _cached_array(
                archive,
                "reconstruction_predictions",
                dtype="<f8",
                shape=(domain_count, row_count),
            )
            targets = _cached_array(
                archive,
                "targets",
                dtype="<f8",
                shape=(row_count,),
            )
            donor_array = np.asarray(archive["donor_specimen_ids"], dtype=np.str_)
            relaxation_array = np.asarray(
                archive["donor_relaxations"], dtype=np.str_
            )
    except MAVISMRISDataError:
        raise
    except (OSError, UnicodeError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise MAVISMRISDataError("MRIS feature-bank cache is invalid") from error
    if donor_array.shape != (domain_count, row_count) or relaxation_array.shape != (
        domain_count,
        row_count,
    ):
        raise MAVISMRISDataError("MRIS feature-bank donor cache changed")
    donors = MappingProxyType(
        {
            domain: tuple(str(value) for value in donor_array[index])
            for index, domain in enumerate(domain_order)
        }
    )
    relaxations = MappingProxyType(
        {
            domain: tuple(str(value) for value in relaxation_array[index])
            for index, domain in enumerate(domain_order)
        }
    )
    frozen_arrays = (
        contexts,
        real,
        positions,
        masks,
        costs,
        shuffled,
        reconstruction,
        checkpoints,
        exact_costs,
        native_counts,
        effective,
    )
    observed_input_state = _hash_inputs(
        state_ids=state_ids,
        domain_order=domain_order,
        donors=donors,
        arrays=frozen_arrays,
    )
    target_digest = hashlib.sha256()
    target_digest.update(observed_input_state.encode("ascii"))
    target_digest.update(targets.tobytes(order="C"))
    if (
        observed_input_state != input_state_sha256
        or target_digest.hexdigest() != target_state_sha256
    ):
        raise MAVISMRISDataError("MRIS feature-bank cache hash changed")
    return MRISFeatureBank(
        domain_order=domain_order,
        state_ids=state_ids,
        specimen_ids=specimen_ids,
        domain_ids=domain_ids,
        trajectory_ids=trajectory_ids,
        methods=methods,
        seeds=tuple(None if int(value) == -1 else int(value) for value in seed_values),
        nominal_checkpoints=checkpoints,
        exact_acquired_costs=exact_costs,
        native_counts=native_counts,
        effective_budgets=effective,
        context_features=contexts,
        real_token_features=real,
        positions_token_features=positions,
        token_masks=masks,
        cost_features=costs,
        shuffled_token_features=shuffled,
        donor_specimen_ids=donors,
        donor_relaxations=relaxations,
        reconstruction_predictions=reconstruction,
        targets=targets,
        input_state_sha256=input_state_sha256,
        target_state_sha256=target_state_sha256,
    )


__all__ = [
    "MAVISMRISDataError",
    "MRISFeatureBank",
    "ModelInputArrays",
    "build_mris_feature_bank",
    "load_mris_feature_bank",
    "save_mris_feature_bank",
]
