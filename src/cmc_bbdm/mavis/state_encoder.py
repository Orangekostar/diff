"""Small permutation-invariant mechanics-relevant information encoder."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from cmc_bbdm.cpb_v3.controls import (
    ControlSpecimen,
    MappingRecord,
    build_stratified_shuffle,
)

from .authority import MAVISAuthority
from .contracts import InspectionState


class MAVISStateEncoderError(ValueError):
    """Raised when an MRIS input or encoder request is invalid."""


_MODES = frozenset(("static", "positions_only", "real", "shuffled"))


def _readonly(value: object, *, dtype: object, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.ascontiguousarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise MAVISStateEncoderError("MRIS input array is invalid") from error
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise MAVISStateEncoderError("MRIS input array is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True, eq=False)
class MRISInput:
    specimen_id: str
    mode: str
    context_features: np.ndarray
    native_shape: tuple[int, int]
    acquired_positions: np.ndarray
    measurement_values: np.ndarray | None
    effective_budget: float
    remaining_budget: float
    content_specimen_id: str | None = None
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.specimen_id) is not str
            or not self.specimen_id
            or self.mode not in _MODES
            or type(self.native_shape) is not tuple
            or len(self.native_shape) != 2
            or any(type(value) is not int or value < 2 for value in self.native_shape)
        ):
            raise MAVISStateEncoderError("MRIS input identity is invalid")
        context = _readonly(
            self.context_features,
            dtype="<f8",
            shape=(34,),
        )
        positions_array = np.asarray(self.acquired_positions)
        if positions_array.ndim != 2 or positions_array.shape[1:] != (2,):
            raise MAVISStateEncoderError("MRIS positions are invalid")
        positions = _readonly(
            positions_array,
            dtype="<i8",
            shape=positions_array.shape,
        )
        count = positions.shape[0]
        linear = positions[:, 0] * self.native_shape[1] + positions[:, 1]
        if (
            np.any(positions < 0)
            or np.any(positions[:, 0] >= self.native_shape[0])
            or np.any(positions[:, 1] >= self.native_shape[1])
            or np.unique(linear).size != count
        ):
            raise MAVISStateEncoderError("MRIS positions are invalid")
        if self.measurement_values is None:
            values = None
        else:
            values = _readonly(
                self.measurement_values,
                dtype=np.uint8,
                shape=(count, 3),
            )
        budget = float(self.effective_budget)
        remaining = float(self.remaining_budget)
        content_specimen_id = self.content_specimen_id
        if content_specimen_id is not None and (
            type(content_specimen_id) is not str or not content_specimen_id
        ):
            raise MAVISStateEncoderError("MRIS content specimen is invalid")
        if (
            isinstance(self.effective_budget, bool)
            or isinstance(self.remaining_budget, bool)
            or not math.isfinite(budget)
            or not math.isfinite(remaining)
            or not 0.0 <= budget <= 1.0
            or not 0.0 <= remaining <= 1.0
            or budget + remaining > 1.0 + 1.0e-15
            or (self.mode == "static" and (count != 0 or values is not None or budget != 0.0))
            or (self.mode == "positions_only" and (count == 0 or values is not None))
            or (self.mode in {"real", "shuffled"} and (count == 0 or values is None))
            or (
                self.mode in {"static", "positions_only"}
                and content_specimen_id is not None
            )
            or (self.mode == "real" and content_specimen_id != self.specimen_id)
            or (
                self.mode == "shuffled"
                and content_specimen_id in {None, self.specimen_id}
            )
        ):
            raise MAVISStateEncoderError("MRIS mode or budget is invalid")
        payload = {
            "schema": 1,
            "specimen_id": self.specimen_id,
            "mode": self.mode,
            "content_specimen_id": content_specimen_id,
            "native_shape": self.native_shape,
            "context_sha256": hashlib.sha256(context.tobytes(order="C")).hexdigest(),
            "positions_sha256": hashlib.sha256(
                positions.tobytes(order="C")
            ).hexdigest(),
            "values_sha256": None
            if values is None
            else hashlib.sha256(values.tobytes(order="C")).hexdigest(),
            "effective_budget": budget,
            "remaining_budget": remaining,
        }
        state = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if self.state_sha256 not in ("", state):
            raise MAVISStateEncoderError("MRIS input state hash changed")
        object.__setattr__(self, "context_features", context)
        object.__setattr__(self, "acquired_positions", positions)
        object.__setattr__(self, "measurement_values", values)
        object.__setattr__(self, "effective_budget", budget)
        object.__setattr__(self, "remaining_budget", remaining)
        object.__setattr__(self, "state_sha256", state)

    @property
    def native_count(self) -> int:
        return self.native_shape[0] * self.native_shape[1]

    def permuted(self, permutation: object) -> MRISInput:
        order = np.asarray(permutation)
        count = self.acquired_positions.shape[0]
        if (
            order.dtype.kind not in "iu"
            or order.shape != (count,)
            or not np.array_equal(np.sort(order), np.arange(count))
        ):
            raise MAVISStateEncoderError("MRIS permutation is invalid")
        return MRISInput(
            specimen_id=self.specimen_id,
            mode=self.mode,
            context_features=self.context_features,
            native_shape=self.native_shape,
            acquired_positions=self.acquired_positions[order],
            measurement_values=None
            if self.measurement_values is None
            else self.measurement_values[order],
            effective_budget=self.effective_budget,
            remaining_budget=self.remaining_budget,
            content_specimen_id=self.content_specimen_id,
        )

    def __eq__(self, other: object) -> bool:
        return type(other) is MRISInput and self.state_sha256 == other.state_sha256


def build_mris_input(state: InspectionState, *, mode: str) -> MRISInput:
    if type(state) is not InspectionState or mode not in {"static", "positions_only", "real"}:
        raise MAVISStateEncoderError("inspection state or MRIS mode is invalid")
    if mode == "static":
        positions = np.empty((0, 2), dtype="<i8")
        values = None
        budget = 0.0
        remaining = 1.0
    else:
        positions = state.acquired_positions
        values = state.measurement_values if mode == "real" else None
        budget = state.effective_budget
        remaining = 1.0 - budget
    content_specimen_id = state.specimen_id if mode == "real" else None
    return MRISInput(
        specimen_id=state.specimen_id,
        mode=mode,
        context_features=state.context_features,
        native_shape=state.native_shape,
        acquired_positions=positions,
        measurement_values=values,
        effective_budget=budget,
        remaining_budget=remaining,
        content_specimen_id=content_specimen_id,
    )


@dataclass(frozen=True, slots=True)
class MRISShuffleAssignment:
    recipient_id: str
    donor_id: str
    recipient_domain: str
    donor_domain: str
    recipient_pool: str
    relaxation: str
    relaxation_level: int
    seed: int


@dataclass(frozen=True, slots=True, eq=False)
class MRISTokenSummary:
    context_features: np.ndarray
    token_features: np.ndarray
    token_mask: np.ndarray
    cost_features: np.ndarray
    state_sha256: str

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is MRISTokenSummary
            and self.state_sha256 == other.state_sha256
        )


def summarize_mris_input(state: MRISInput) -> MRISTokenSummary:
    if type(state) is not MRISInput:
        raise MAVISStateEncoderError("MRIS summary input is invalid")
    count = state.acquired_positions.shape[0]
    tokens = np.zeros((64, 6), dtype=np.float64)
    mask = np.zeros(64, dtype=bool)
    if count:
        positions = np.asarray(state.acquired_positions, dtype=np.float64)
        normalized = positions / np.asarray(
            [state.native_shape[0] - 1, state.native_shape[1] - 1],
            dtype=np.float64,
        )
        row_bins = np.minimum((normalized[:, 0] * 8.0).astype(np.int64), 7)
        column_bins = np.minimum((normalized[:, 1] * 8.0).astype(np.int64), 7)
        bins = row_bins * 8 + column_bins
        counts = np.bincount(bins, minlength=64).astype(np.float64)
        mask = counts > 0.0
        tokens[mask, 0] = (
            np.bincount(bins, weights=normalized[:, 0], minlength=64)[mask]
            / counts[mask]
        )
        tokens[mask, 1] = (
            np.bincount(bins, weights=normalized[:, 1], minlength=64)[mask]
            / counts[mask]
        )
        if state.measurement_values is not None:
            values = np.asarray(state.measurement_values, dtype=np.float64) / 255.0
            for channel in range(3):
                tokens[mask, channel + 2] = (
                    np.bincount(
                        bins,
                        weights=values[:, channel],
                        minlength=64,
                    )[mask]
                    / counts[mask]
                )
        tokens[mask, 5] = counts[mask] / count
    acquisition_fraction = (
        0.0 if count == 0 else math.log1p(count) / math.log1p(state.native_count)
    )
    context = _readonly(
        state.context_features,
        dtype="<f8",
        shape=(state.context_features.size,),
    )
    frozen_tokens = _readonly(tokens, dtype="<f8", shape=(64, 6))
    frozen_mask = _readonly(mask, dtype=bool, shape=(64,))
    cost = _readonly(
        (state.effective_budget, state.remaining_budget, acquisition_fraction),
        dtype="<f8",
        shape=(3,),
    )
    digest = hashlib.sha256()
    digest.update(b"mavis-mris-token-summary-v1")
    digest.update(state.state_sha256.encode("ascii"))
    for value in (context, frozen_tokens, frozen_mask, cost):
        digest.update(value.tobytes(order="C"))
    return MRISTokenSummary(
        context_features=context,
        token_features=frozen_tokens,
        token_mask=frozen_mask,
        cost_features=cost,
        state_sha256=digest.hexdigest(),
    )


def _shuffle_record(
    authority: MAVISAuthority,
    specimen_id: str,
) -> ControlSpecimen:
    index = authority.specimen_ids.index(specimen_id)
    context = authority.policy_context(specimen_id).context_features
    if context[2] not in (0.0, 1.0):
        raise MAVISStateEncoderError("shuffle stratification metadata is invalid")
    laminate = "cross_ply" if context[2] == 1.0 else "quasi_isotropic"
    ply_count = round(float(context[1]) * 24.0)
    energy = float(np.expm1(context[9]))
    if ply_count <= 0 or not np.isfinite(energy) or energy < 0.0:
        raise MAVISStateEncoderError("shuffle stratification metadata is invalid")
    return ControlSpecimen(
        specimen_id=specimen_id,
        dataset_id=authority.dataset_ids[index],
        laminate_type=laminate,
        ply_count=ply_count,
        total_impact_energy_j=energy,
        crop_sha256=authority.source_image_sha256[index],
        source_screenshot_sha256=authority.decoded_image_sha256[index],
    )


def _shuffle_assignment(row: MappingRecord) -> MRISShuffleAssignment:
    return MRISShuffleAssignment(
        recipient_id=row.recipient_id,
        donor_id=row.donor_id,
        recipient_domain=row.recipient_dataset_id,
        donor_domain=row.donor_dataset_id,
        recipient_pool=row.recipient_pool,
        relaxation=row.relaxation,
        relaxation_level=row.relaxation_level,
        seed=row.seed,
    )


def build_mris_shuffle_mapping(
    authority: MAVISAuthority,
    *,
    outer_domain: str,
    seed: int,
) -> tuple[MRISShuffleAssignment, ...]:
    """Build a deterministic source/target-separated stratified derangement."""

    if (
        type(authority) is not MAVISAuthority
        or type(outer_domain) is not str
        or outer_domain not in authority.dataset_ids
        or type(seed) is not int
    ):
        raise MAVISStateEncoderError("shuffle mapping request is invalid")
    records = tuple(
        _shuffle_record(authority, specimen_id)
        for specimen_id in authority.specimen_ids
    )
    source_ids = tuple(
        specimen_id
        for specimen_id, domain in zip(
            authority.specimen_ids, authority.dataset_ids, strict=True
        )
        if domain != outer_domain
    )
    target_ids = tuple(
        specimen_id
        for specimen_id, domain in zip(
            authority.specimen_ids, authority.dataset_ids, strict=True
        )
        if domain == outer_domain
    )
    try:
        rows = build_stratified_shuffle(
            records,
            source_ids,
            target_ids,
            seed=seed,
        )
    except (TypeError, ValueError) as error:
        raise MAVISStateEncoderError("shuffle mapping could not be built") from error
    return tuple(_shuffle_assignment(row) for row in rows)


def build_shuffled_mris_input(
    state: InspectionState,
    *,
    authority: MAVISAuthority,
    donor_specimen_id: str,
) -> MRISInput:
    """Replace only recipient measurement content with another specimen's content."""

    if (
        type(state) is not InspectionState
        or type(authority) is not MAVISAuthority
        or type(donor_specimen_id) is not str
        or not donor_specimen_id
        or donor_specimen_id == state.specimen_id
    ):
        raise MAVISStateEncoderError("shuffled MRIS request is invalid")
    return build_shuffled_mris_input_from_input(
        build_mris_input(state, mode="real"),
        authority=authority,
        donor_specimen_id=donor_specimen_id,
    )


def build_shuffled_mris_input_from_input(
    state: MRISInput,
    *,
    authority: MAVISAuthority,
    donor_specimen_id: str,
) -> MRISInput:
    if (
        type(state) is not MRISInput
        or state.mode != "real"
        or type(authority) is not MAVISAuthority
        or type(donor_specimen_id) is not str
        or not donor_specimen_id
        or donor_specimen_id == state.specimen_id
    ):
        raise MAVISStateEncoderError("shuffled MRIS request is invalid")
    donor_shape = authority.policy_context(donor_specimen_id).native_shape
    recipient_positions = state.acquired_positions
    mapped = np.empty_like(recipient_positions)
    mapped[:, 0] = np.rint(
        recipient_positions[:, 0]
        * (donor_shape[0] - 1)
        / (state.native_shape[0] - 1)
    ).astype(np.int64)
    mapped[:, 1] = np.rint(
        recipient_positions[:, 1]
        * (donor_shape[1] - 1)
        / (state.native_shape[1] - 1)
    ).astype(np.int64)
    try:
        donor_values = authority._reveal_values(donor_specimen_id, mapped)
    except (TypeError, ValueError) as error:
        raise MAVISStateEncoderError("shuffled donor content is unavailable") from error
    return MRISInput(
        specimen_id=state.specimen_id,
        mode="shuffled",
        context_features=state.context_features,
        native_shape=state.native_shape,
        acquired_positions=recipient_positions,
        measurement_values=donor_values,
        effective_budget=state.effective_budget,
        remaining_budget=1.0 - state.effective_budget,
        content_specimen_id=donor_specimen_id,
    )


class MRISStateEncoder(nn.Module):
    """Two-layer DeepSets encoder with context and exact-cost fusion."""

    def __init__(
        self,
        *,
        context_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
    ) -> None:
        super().__init__()
        if (
            type(context_dimension) is not int
            or type(hidden_dimension) is not int
            or type(output_dimension) is not int
            or min(context_dimension, hidden_dimension, output_dimension) <= 0
        ):
            raise MAVISStateEncoderError("MRIS encoder dimensions are invalid")
        self.context_dimension = context_dimension
        self.output_dimension = output_dimension
        self.token_mlp = nn.Sequential(
            nn.Linear(6, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dimension + 3, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, output_dimension),
        )

    def forward_batch(
        self,
        contexts: torch.Tensor,
        token_features: torch.Tensor,
        token_masks: torch.Tensor,
        cost_features: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not all(
                isinstance(value, torch.Tensor)
                for value in (contexts, token_features, token_masks, cost_features)
            )
            or contexts.ndim != 2
            or contexts.shape[1] != self.context_dimension
            or token_features.shape != (contexts.shape[0], 64, 6)
            or token_masks.shape != (contexts.shape[0], 64)
            or token_masks.dtype != torch.bool
            or cost_features.shape != (contexts.shape[0], 3)
        ):
            raise MAVISStateEncoderError("MRIS encoder batch is invalid")
        encoded = self.token_mlp(token_features)
        weights = token_masks.unsqueeze(-1).to(dtype=encoded.dtype)
        encoded_sum = torch.sum(encoded * weights, dim=1)
        counts = torch.sum(weights, dim=1).clamp_min(1.0)
        token_mean = encoded_sum / counts
        token_sum = encoded_sum / torch.sqrt(counts)
        context_state = self.context_mlp(torch.cat((contexts, cost_features), dim=1))
        output = self.fusion(
            torch.cat((token_mean, token_sum, context_state), dim=1)
        )
        if output.shape != (contexts.shape[0], self.output_dimension) or not torch.isfinite(
            output
        ).all():
            raise MAVISStateEncoderError("MRIS encoder output is invalid")
        return output

    def forward(self, state: MRISInput) -> torch.Tensor:
        if type(state) is not MRISInput or state.context_features.shape != (
            self.context_dimension,
        ):
            raise MAVISStateEncoderError("MRIS encoder input is invalid")
        parameter = next(self.parameters())
        summary = summarize_mris_input(state)
        output = self.forward_batch(
            torch.tensor(
                np.asarray(summary.context_features)[None],
                dtype=parameter.dtype,
                device=parameter.device,
            ),
            torch.tensor(
                np.asarray(summary.token_features)[None],
                dtype=parameter.dtype,
                device=parameter.device,
            ),
            torch.tensor(
                np.asarray(summary.token_mask)[None],
                dtype=torch.bool,
                device=parameter.device,
            ),
            torch.tensor(
                np.asarray(summary.cost_features)[None],
                dtype=parameter.dtype,
                device=parameter.device,
            ),
        )[0]
        return output


__all__ = [
    "MAVISStateEncoderError",
    "MRISInput",
    "MRISShuffleAssignment",
    "MRISStateEncoder",
    "MRISTokenSummary",
    "build_mris_input",
    "build_mris_shuffle_mapping",
    "build_shuffled_mris_input",
    "build_shuffled_mris_input_from_input",
    "summarize_mris_input",
]
