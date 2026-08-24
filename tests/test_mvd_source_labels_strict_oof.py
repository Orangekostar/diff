from __future__ import annotations

import polars as pl
import pytest
from test_mvd_config import CONFIG, ROOT

from cmc_bbdm.mvd.authority import (
    MVDAuthorityError,
    load_compact_mvd_authority,
    validate_source_label_barriers,
)
from cmc_bbdm.mvd.config import load_mvd_config


def test_all_a4_source_labels_are_strict_outer_and_query_oof() -> None:
    config = load_mvd_config(CONFIG, project_root=ROOT)
    authority = load_compact_mvd_authority(config, project_root=ROOT)

    source_audits = authority.fit_audits.filter(
        pl.col("audit_family") == "source_value"
    )
    assert source_audits.height == 390
    assert source_audits.filter(pl.col("stage") == "outer").height == 30
    validate_source_label_barriers(
        authority.source_values,
        authority.fit_audits,
        domain_order=config.domain_order,
    )


def test_source_label_barrier_rejects_outer_domain_in_fit_roster() -> None:
    config = load_mvd_config(CONFIG, project_root=ROOT)
    authority = load_compact_mvd_authority(config, project_root=ROOT)
    audits = authority.fit_audits.clone()
    first = audits.row(0, named=True)
    changed = audits.with_row_index("row_index").with_columns(
        pl.when(pl.col("row_index") == 0)
        .then(pl.lit(first["fit_domains"] + "|" + first["held_out_target_domain"]))
        .otherwise(pl.col("fit_domains"))
        .alias("fit_domains")
    ).drop("row_index")

    with pytest.raises(MVDAuthorityError, match="source-label fit barrier changed"):
        validate_source_label_barriers(
            authority.source_values,
            changed,
            domain_order=config.domain_order,
        )
