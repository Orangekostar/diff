from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.msss import coupling_artifacts
from cmc_bbdm.msss.coupling import diagnose_coupling
from cmc_bbdm.msss.coupling_artifacts import (
    CouplingArtifactError,
    build_coupling_diagnostic,
    load_coupling_protocol,
    load_parent_candidate_errors,
    publish_coupling_package,
    replay_coupling_package,
    validate_coupling_package,
)
from cmc_bbdm.msss.scale_features import ScaleCondition

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/msss_no_go_coupling.yaml"


def _diagnostic():
    axes = ("sampling", "gaussian", "wavelet")
    conditions = tuple(
        ScaleCondition(
            condition_id=f"{axis}:candidate={rank}",
            axis=axis,
            value=float(rank),
            coarse_rank=rank,
            primary_eligible=True,
            is_full_identity=rank == 0,
            wavelet="db2" if axis == "wavelet" else None,
            level=rank if axis == "wavelet" else None,
            mode="low_only" if axis == "wavelet" else None,
        )
        for axis in axes
        for rank in range(2)
    )
    specimen_ids = tuple(f"s{index:02d}" for index in range(12))
    datasets = tuple(
        domain for domain in ("d1", "d2", "d3", "d4", "d5", "d6") for _ in range(2)
    )
    damage = np.arange(1.0, 13.0)
    return diagnose_coupling(
        conditions=conditions,
        specimen_ids=specimen_ids,
        dataset_ids=datasets,
        ply_count=tuple(value for value in (8, 8, 16, 16, 24, 24) for _ in range(2)),
        layup_family=tuple(
            value
            for value in (
                "cross_ply",
                "quasi_isotropic",
                "cross_ply",
                "quasi_isotropic",
                "cross_ply",
                "quasi_isotropic",
            )
            for _ in range(2)
        ),
        damage_sizes={
            "damage_area": damage,
            "damage_height": damage[::-1],
            "damage_width": np.roll(damage, 1),
        },
        absolute_errors={
            condition.condition_id: np.full(
                12, 1.0 + 0.02 * condition.coarse_rank, dtype=np.float64
            )
            for condition in conditions
        },
        margin=0.05,
    )


def test_coupling_protocol_binds_the_formal_no_go_parent() -> None:
    protocol = load_coupling_protocol(CONFIG, project_root=ROOT)

    assert protocol.required_gate_status == "NO_GO"
    assert protocol.required_test_only is False
    assert protocol.parent_scientific_digest == (
        "6ac389b0a4e09487202f5a8a9273dfdf5b338ef40de705661c5877e3e9bd0152"
    )
    assert protocol.margin == 0.05
    assert protocol.output_formal == ROOT / "results/msss/s1_no_go_coupling"


def test_parent_loader_accepts_only_the_complete_primary_candidate_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_coupling_protocol(CONFIG, project_root=ROOT)
    monkeypatch.setattr(
        coupling_artifacts,
        "load_protocol",
        lambda *_args, **_kwargs: pytest.fail(
            "parent artifacts must not load the external MSSS source tree"
        ),
    )
    parent = load_parent_candidate_errors(protocol, project_root=ROOT)

    assert parent.specimen_count == 276
    assert len(parent.conditions) == 9 + 9 + 4
    assert set(parent.absolute_errors) == {
        condition.condition_id for condition in parent.conditions
    }
    assert all(array.shape == (276,) for array in parent.absolute_errors.values())

    with pytest.raises(CouplingArtifactError, match="gate"):
        load_parent_candidate_errors(
            replace(protocol, required_gate_status="GO"), project_root=ROOT
        )


def test_coupling_package_is_atomic_checksum_bound_and_non_overwriting(
    tmp_path: Path,
) -> None:
    protocol = load_coupling_protocol(CONFIG, project_root=ROOT)
    output = tmp_path / "coupling"

    published = publish_coupling_package(
        output,
        protocol=protocol,
        diagnostic=_diagnostic(),
        config_path=CONFIG,
    )

    assert published.coupling_status == "NO_CONSISTENT_SIGNAL"
    assert published.validation_status == "NOT_VALIDATED_POST_HOC"
    assert {
        "group_scale_curves.csv",
        "group_scale_selection.csv",
        "damage_size_bins.csv",
        "factor_trends.csv",
        "summary.json",
        "REPORT.md",
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    } == {path.name for path in output.iterdir()}
    for csv_name in (
        "damage_size_bins.csv",
        "factor_trends.csv",
        "group_scale_curves.csv",
        "group_scale_selection.csv",
    ):
        assert b"\r\n" not in (output / csv_name).read_bytes()
    assert validate_coupling_package(
        output, protocol=protocol, config_path=CONFIG
    ) == published
    with pytest.raises(CouplingArtifactError, match="exists"):
        publish_coupling_package(
            output,
            protocol=protocol,
            diagnostic=_diagnostic(),
            config_path=CONFIG,
        )

    with (output / "group_scale_curves.csv").open("a", encoding="utf-8") as handle:
        handle.write("corrupt\n")
    with pytest.raises(CouplingArtifactError, match="checksum"):
        validate_coupling_package(output, protocol=protocol, config_path=CONFIG)


def test_formal_authorities_build_and_replay_the_complete_diagnostic(
    tmp_path: Path,
) -> None:
    protocol = load_coupling_protocol(CONFIG, project_root=ROOT)
    diagnostic = build_coupling_diagnostic(protocol, project_root=ROOT)

    assert len(diagnostic.selections) == 60
    assert len(diagnostic.curves) == 440
    assert len(diagnostic.damage_bins) == 828
    assert len(diagnostic.trends) == 15
    assert len(diagnostic.alignments) == 5

    source = tmp_path / "formal"
    replay = tmp_path / "replay"
    formal_validation = publish_coupling_package(
        source, protocol=protocol, diagnostic=diagnostic, config_path=CONFIG
    )
    replay_validation = replay_coupling_package(
        source,
        replay,
        protocol=protocol,
        project_root=ROOT,
        config_path=CONFIG,
    )

    assert replay_validation == formal_validation
    assert {
        path.name: path.read_bytes() for path in source.iterdir() if path.is_file()
    } == {path.name: path.read_bytes() for path in replay.iterdir() if path.is_file()}
