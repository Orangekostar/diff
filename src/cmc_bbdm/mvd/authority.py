"""Compact immutable bindings to the historical MVA authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import polars as pl

from cmc_bbdm.mva.a4_candidate_bank import CandidateBank, load_candidate_bank

from .config import MVDConfig


class MVDAuthorityError(ValueError):
    """Raised when compact MVD authority is incomplete, leaky, or changed."""


_SOURCE_VALUE_COLUMNS = {
    "specimen_id",
    "dataset_id",
    "outer_domain",
    "method",
    "cell_index",
    "from_level",
    "to_level",
    "added_measurements",
    "candidate_bank_state_sha256",
    "absolute_error_after",
    "absolute_error_before",
    "candidate_prediction",
    "current_prediction",
    "predictor_state_sha256",
    "primary_value",
    "secondary_value",
}
_FIT_AUDIT_COLUMNS = {
    "audit_family",
    "evaluator",
    "checkpoint",
    "stage",
    "held_out_target_domain",
    "query_source_domain",
    "query_domains",
    "fit_domains",
    "query_specimen_ids",
    "fit_specimen_ids",
    "pca_dimension",
    "predictor_state_sha256",
}
_RANKING_COLUMNS = {
    "outer_domain",
    "method",
    "cell_index",
    "ranking_position",
    "cell_score",
    "mean_raw_value",
    "mean_value_per_measurement",
    "source_domains",
    "source_specimen_count",
    "source_label_state_sha256",
}


@dataclass(frozen=True, slots=True)
class CompactMVDAuthority:
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    image_sha256: tuple[str, ...]
    candidate_banks: MappingProxyType
    source_values: pl.DataFrame
    fit_audits: pl.DataFrame
    rankings: pl.DataFrame
    state_sha256: str

    @property
    def specimen_count(self) -> int:
        return len(self.specimen_ids)


def _split(value: object) -> tuple[str, ...]:
    if type(value) is not str or not value:
        raise MVDAuthorityError("source-label roster contains invalid text")
    return tuple(value.split("|"))


def validate_source_label_barriers(
    source_values: pl.DataFrame,
    fit_audits: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
) -> None:
    """Validate A4 source labels and every nested fit exclusion barrier."""

    if (
        type(source_values) is not pl.DataFrame
        or set(source_values.columns) != _SOURCE_VALUE_COLUMNS
        or source_values.height != 264_960
        or type(fit_audits) is not pl.DataFrame
        or set(fit_audits.columns) != _FIT_AUDIT_COLUMNS
        or len(domain_order) != 6
        or len(set(domain_order)) != 6
    ):
        raise MVDAuthorityError("source-label authority schema changed")
    domains = set(domain_order)
    audits = fit_audits.filter(pl.col("audit_family") == "source_value")
    if (
        audits.height != 390
        or set(audits["stage"]) != {"inner", "outer"}
        or audits.filter(pl.col("stage") == "outer").height != 30
    ):
        raise MVDAuthorityError("source-label fit roster changed")

    outer_predictors: dict[tuple[str, str], str] = {}
    for row in audits.iter_rows(named=True):
        held = str(row["held_out_target_domain"])
        query_source = str(row["query_source_domain"])
        query_domains = set(_split(row["query_domains"]))
        fit_domains = set(_split(row["fit_domains"]))
        query_specimens = set(_split(row["query_specimen_ids"]))
        fit_specimens = set(_split(row["fit_specimen_ids"]))
        if (
            held not in domains
            or query_source not in domains - {held}
            or held in fit_domains
            or query_source in fit_domains
            or query_domains & fit_domains
            or query_specimens & fit_specimens
            or not fit_domains <= domains
            or row["evaluator"] != "P-A-label"
            or row["checkpoint"] is not None
        ):
            raise MVDAuthorityError("source-label fit barrier changed")
        if row["stage"] == "outer":
            if query_domains != {query_source} or fit_domains != domains - {
                held,
                query_source,
            }:
                raise MVDAuthorityError("source-label fit barrier changed")
            key = (held, query_source)
            if key in outer_predictors:
                raise MVDAuthorityError("source-label outer predictor duplicated")
            outer_predictors[key] = str(row["predictor_state_sha256"])
        elif len(fit_domains) != 3 or len(query_domains) != 1:
            raise MVDAuthorityError("source-label fit barrier changed")
    if set(outer_predictors) != {
        (held, query) for held in domain_order for query in domain_order if query != held
    }:
        raise MVDAuthorityError("source-label outer predictor roster changed")

    if (
        set(source_values["outer_domain"]) != domains
        or set(source_values["dataset_id"]) != domains
        or set(source_values["method"])
        != {
            "global_appearance_mask",
            "global_reconstruction_mask",
            "global_mechanical_mask",
        }
        or source_values.filter(pl.col("outer_domain") == pl.col("dataset_id")).height
        or source_values.filter(
            (pl.col("cell_index") < 0)
            | (pl.col("cell_index") >= 64)
            | (pl.col("from_level") != 0)
            | (pl.col("to_level") != 1)
            | (pl.col("added_measurements") <= 0)
        ).height
    ):
        raise MVDAuthorityError("source-label value roster changed")
    groups = source_values.group_by(
        ["outer_domain", "dataset_id", "specimen_id", "method"]
    ).agg(
        pl.len().alias("rows"),
        pl.col("cell_index").n_unique().alias("cells"),
        pl.col("candidate_bank_state_sha256").n_unique().alias("bank_hashes"),
    )
    if (
        groups.height != 4_140
        or groups.filter(
            (pl.col("rows") != 64)
            | (pl.col("cells") != 64)
            | (pl.col("bank_hashes") != 1)
        ).height
    ):
        raise MVDAuthorityError("source-label candidate roster changed")

    mechanical = source_values.filter(pl.col("method") == "global_mechanical_mask")
    expected = pl.DataFrame(
        [
            {
                "outer_domain": held,
                "dataset_id": query,
                "expected_predictor": predictor,
            }
            for (held, query), predictor in sorted(outer_predictors.items())
        ]
    )
    joined = mechanical.join(
        expected, on=["outer_domain", "dataset_id"], how="left"
    )
    if (
        mechanical.height != 88_320
        or joined.filter(
            pl.col("expected_predictor").is_null()
            | (pl.col("predictor_state_sha256") != pl.col("expected_predictor"))
            | pl.col("current_prediction").is_null()
            | pl.col("candidate_prediction").is_null()
            | pl.col("primary_value").is_null()
        ).height
    ):
        raise MVDAuthorityError("source-label predictor binding changed")


def _state_sha256(
    config: MVDConfig,
    banks: dict[float, CandidateBank],
    source_values: pl.DataFrame,
    fit_audits: pl.DataFrame,
    rankings: pl.DataFrame,
) -> str:
    payload = {
        "authority_state_sha256": config.authority_state_sha256,
        "bank_states": {str(key): value.state_sha256 for key, value in banks.items()},
        "fit_audits_sha256": config.sources["a4_fit_audits"].sha256,
        "fit_rows": fit_audits.height,
        "ranking_rows": rankings.height,
        "rankings_sha256": config.sources["a4_rankings"].sha256,
        "source_value_rows": source_values.height,
        "source_values_sha256": config.sources["a4_source_values"].sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def load_compact_mvd_authority(
    config: MVDConfig, *, project_root: str | Path
) -> CompactMVDAuthority:
    """Load CandidateBanks and A4 source evidence without raw C-scan access."""

    if type(config) is not MVDConfig:
        raise MVDAuthorityError("issued MVDConfig is required")
    root = Path(project_root).resolve(strict=True)
    banks: dict[float, CandidateBank] = {}
    specimen_ids: tuple[str, ...] | None = None
    image_sha256: tuple[str, ...] | None = None
    dataset_ids: tuple[str, ...] | None = None
    for budget, source_name in (
        (0.015625, "candidate_bank_0p015625"),
        (0.03125, "candidate_bank_0p03125"),
    ):
        bank = load_candidate_bank(
            root / config.sources[source_name].path,
            expected_authority_state_sha256=config.authority_state_sha256,
            expected_specimen_ids=specimen_ids,
            expected_image_sha256=image_sha256,
            expected_initial_budget=budget,
        )
        if bank.state_sha256 != config.candidate_bank_states[budget]:
            raise MVDAuthorityError("candidate bank state changed")
        specimen_ids = bank.specimen_ids
        image_sha256 = bank.image_sha256
        dataset_ids = bank.dataset_ids
        banks[budget] = bank
    assert specimen_ids is not None and image_sha256 is not None and dataset_ids is not None
    if (
        len(specimen_ids) != config.specimen_count
        or tuple(dict.fromkeys(dataset_ids)) != config.domain_order
    ):
        raise MVDAuthorityError("candidate bank cohort changed")
    try:
        source_values = pl.read_parquet(root / config.sources["a4_source_values"].path)
        fit_audits = pl.read_csv(root / config.sources["a4_fit_audits"].path)
        rankings = pl.read_csv(root / config.sources["a4_rankings"].path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise MVDAuthorityError("A4 compact evidence cannot be read") from error
    validate_source_label_barriers(
        source_values, fit_audits, domain_order=config.domain_order
    )
    if (
        set(rankings.columns) != _RANKING_COLUMNS
        or rankings.height != 1_152
        or set(rankings["outer_domain"]) != set(config.domain_order)
        or rankings.group_by(["outer_domain", "method"]).len().filter(
            pl.col("len") != 64
        ).height
    ):
        raise MVDAuthorityError("A4 ranking authority changed")
    return CompactMVDAuthority(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        image_sha256=image_sha256,
        candidate_banks=MappingProxyType(banks),
        source_values=source_values,
        fit_audits=fit_audits,
        rankings=rankings,
        state_sha256=_state_sha256(
            config, banks, source_values, fit_audits, rankings
        ),
    )


__all__ = [
    "CompactMVDAuthority",
    "MVDAuthorityError",
    "load_compact_mvd_authority",
    "validate_source_label_barriers",
]
