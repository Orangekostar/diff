"""Privileged initial Mechanical Value arrays for MVD M0 and label creation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import polars as pl

from .authority import CompactMVDAuthority, MVDAuthorityError


@dataclass(frozen=True, slots=True)
class InitialValueDataset:
    outer_domain: str
    source_domains: tuple[str, ...]
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    initial_budget: float
    initial_embeddings: np.ndarray
    candidate_embeddings: np.ndarray
    mechanical_values: np.ndarray
    candidate_costs: np.ndarray
    current_predictions: np.ndarray
    predictor_state_sha256: tuple[str, ...]
    candidate_bank_state_sha256: str
    authority_state_sha256: str
    state_sha256: str

    @property
    def specimen_count(self) -> int:
        return len(self.specimen_ids)


def _readonly(value: object, *, dtype: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise MVDAuthorityError("initial-value array changed")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


def _state_sha256(dataset: InitialValueDataset) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "authority_state_sha256": dataset.authority_state_sha256,
                "candidate_bank_state_sha256": dataset.candidate_bank_state_sha256,
                "dataset_ids": dataset.dataset_ids,
                "initial_budget": dataset.initial_budget,
                "outer_domain": dataset.outer_domain,
                "predictor_state_sha256": dataset.predictor_state_sha256,
                "source_domains": dataset.source_domains,
                "specimen_ids": dataset.specimen_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for name in (
        "initial_embeddings",
        "candidate_embeddings",
        "mechanical_values",
        "candidate_costs",
        "current_predictions",
    ):
        value = getattr(dataset, name)
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def build_source_initial_value_dataset(
    authority: CompactMVDAuthority, *, outer_domain: str
) -> InitialValueDataset:
    """Align strict source OOF Mechanical Values with one CandidateBank."""

    if type(authority) is not CompactMVDAuthority:
        raise MVDAuthorityError("issued compact MVD authority is required")
    domains = tuple(dict.fromkeys(authority.dataset_ids))
    if outer_domain not in domains:
        raise MVDAuthorityError("outer domain is not registered")
    budget = 0.03125 if outer_domain == "74t7kcdgkr" else 0.015625
    bank = authority.candidate_banks[budget]
    indices = tuple(
        index
        for index, dataset_id in enumerate(authority.dataset_ids)
        if dataset_id != outer_domain
    )
    specimen_ids = tuple(authority.specimen_ids[index] for index in indices)
    dataset_ids = tuple(authority.dataset_ids[index] for index in indices)
    mechanical = authority.source_values.filter(
        (pl.col("outer_domain") == outer_domain)
        & (pl.col("method") == "global_mechanical_mask")
    )
    rows_by_specimen: dict[str, list[dict[str, object]]] = {}
    for row in mechanical.sort(["specimen_id", "cell_index"]).iter_rows(named=True):
        rows_by_specimen.setdefault(str(row["specimen_id"]), []).append(row)
    values = np.empty((len(indices), 64), dtype="<f8")
    costs = np.empty((len(indices), 64), dtype="<i8")
    current = np.empty(len(indices), dtype="<f8")
    predictor_hashes: list[str] = []
    for row_index, (global_index, specimen_id, dataset_id) in enumerate(
        zip(indices, specimen_ids, dataset_ids, strict=True)
    ):
        rows = rows_by_specimen.get(specimen_id, [])
        if (
            len(rows) != 64
            or tuple(int(row["cell_index"]) for row in rows) != tuple(range(64))
            or {str(row["dataset_id"]) for row in rows} != {dataset_id}
            or {str(row["candidate_bank_state_sha256"]) for row in rows}
            != {bank.state_sha256}
        ):
            raise MVDAuthorityError("initial-value specimen roster changed")
        predictions = {float(row["current_prediction"]) for row in rows}
        hashes = {str(row["predictor_state_sha256"]) for row in rows}
        if len(predictions) != 1 or len(hashes) != 1:
            raise MVDAuthorityError("initial-value predictor binding changed")
        values[row_index] = [float(row["primary_value"]) for row in rows]
        costs[row_index] = [int(row["added_measurements"]) for row in rows]
        if not np.array_equal(costs[row_index], bank.added_measurements[global_index]):
            raise MVDAuthorityError("initial-value action costs changed")
        current[row_index] = predictions.pop()
        predictor_hashes.append(hashes.pop())
    dataset = InitialValueDataset(
        outer_domain=outer_domain,
        source_domains=tuple(domain for domain in domains if domain != outer_domain),
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        initial_budget=budget,
        initial_embeddings=_readonly(
            bank.initial_embeddings[np.asarray(indices)],
            dtype="<f8",
            shape=(len(indices), 512),
        ),
        candidate_embeddings=_readonly(
            bank.embeddings[np.asarray(indices)],
            dtype="<f8",
            shape=(len(indices), 64, 512),
        ),
        mechanical_values=_readonly(
            values, dtype="<f8", shape=(len(indices), 64)
        ),
        candidate_costs=_readonly(costs, dtype="<i8", shape=(len(indices), 64)),
        current_predictions=_readonly(
            current, dtype="<f8", shape=(len(indices),)
        ),
        predictor_state_sha256=tuple(predictor_hashes),
        candidate_bank_state_sha256=bank.state_sha256,
        authority_state_sha256=authority.state_sha256,
        state_sha256="",
    )
    object.__setattr__(dataset, "state_sha256", _state_sha256(dataset))
    return dataset


__all__ = ["InitialValueDataset", "build_source_initial_value_dataset"]
