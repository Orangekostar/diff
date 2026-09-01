from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np

from cmc_bbdm.inspection_agent.cai_assessor import StateFeatureRow
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent_g1 import g1 as g1_module
from cmc_bbdm.inspection_agent_g1.g1 import (
    G1Runtime,
    G1RuntimeSurface,
    build_g1_source_dependencies,
    load_g1_protocol,
    source_teacher_bank_path,
    specimen_integrity_sha256,
)
from cmc_bbdm.mavis.authority import MAVISAuthority

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/inspection_agent_g1.yaml"


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _surface(domain: str, specimen: str) -> G1RuntimeSurface:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    scores = np.linspace(0.0, 1.0, 64)
    scores.setflags(write=False)
    median = np.zeros(3, dtype=np.float64)
    median.setflags(write=False)
    hypothesis = SurfaceHypothesis(
        scores=scores,
        top_cells=tuple(range(63, 55, -1)),
        border_median_rgb=median,
        state_sha256=_sha(f"hypothesis-{domain}-{specimen}"),
    )
    return G1RuntimeSurface(
        dataset_id=domain,
        specimen_id=specimen,
        image=image,
        surface_sha256=_sha(f"surface-{domain}-{specimen}"),
        hypothesis=hypothesis,
    )


def _runtime(domains: tuple[str, ...]) -> G1Runtime:
    specimen_ids = tuple(f"{domain}-s0" for domain in domains)
    images = tuple(
        np.full((41, 43, 3), index * 23, dtype=np.uint8)
        for index, _domain in enumerate(domains)
    )
    authority = MAVISAuthority.from_arrays(
        specimen_ids=specimen_ids,
        dataset_ids=domains,
        images=images,
        targets=np.linspace(0.1, 0.6, len(domains)),
        metadata13=np.zeros((len(domains), 13)),
        profile_stats21=np.zeros((len(domains), 21)),
        source_image_sha256=tuple(_sha(f"cscan-{value}") for value in specimen_ids),
    )
    surfaces = {
        (domain, specimen): _surface(domain, specimen)
        for domain, specimen in zip(domains, specimen_ids, strict=True)
    }
    return G1Runtime(
        mavis=authority,
        surfaces=MappingProxyType(surfaces),
        surface_authority_sha256=_sha("surface-authority"),
    )


def _assessor_rows(
    domains: tuple[str, ...],
    *,
    outer_target: str,
    labeled_domain: str,
) -> tuple[StateFeatureRow, ...]:
    generator = np.random.Generator(np.random.PCG64(20260901))
    output = []
    fit_domains = tuple(
        domain for domain in domains if domain not in {outer_target, labeled_domain}
    )
    for domain_index, domain in enumerate(fit_domains):
        for specimen_index in range(2):
            specimen = f"{domain}-fit-{specimen_index}"
            target = 0.1 * domain_index + 0.01 * specimen_index
            for state_index in range(13):
                output.append(
                    StateFeatureRow(
                        sample_id=f"{specimen}|{state_index}",
                        specimen_id=specimen,
                        dataset_id=domain,
                        policy="WARM_START" if state_index == 0 else "CONTINUATION",
                        observation_sha256=f"{len(output) + 1:064x}",
                        embedding=generator.normal(size=512),
                        effective_budget=state_index / 52,
                        observed_cell_fraction=(state_index + 8) / 64,
                        mean_observed_level=state_index / 12,
                        true_cai=target,
                    )
                )
    return tuple(output)


def test_specimen_integrity_identity_binds_both_measurement_modalities() -> None:
    values = {
        "dataset_id": "domain",
        "specimen_id": "sample",
        "cscan_source_sha256": _sha("cscan-source"),
        "cscan_decoded_sha256": _sha("cscan-decoded"),
        "surface_sha256": _sha("surface"),
    }
    first = specimen_integrity_sha256(**values)
    assert first == specimen_integrity_sha256(**values)
    assert first != specimen_integrity_sha256(
        **{**values, "surface_sha256": _sha("other-surface")}
    )


def test_source_teacher_bank_path_is_fold_bound_and_relative() -> None:
    assert source_teacher_bank_path("banks", "outer", "source").as_posix() == (
        "banks/outer/source.parquet"
    )


def test_runtime_surface_snapshots_policy_carrier_as_read_only() -> None:
    surface = _surface("domain", "sample")
    carrier = g1_module._surface_carrier(surface.hypothesis)

    assert surface.image.shape == (80, 80, 3)
    assert surface.image.flags.writeable is False
    assert carrier.shape == (1, 1, 3)
    assert carrier.flags.writeable is False


def test_source_dependency_orchestration_excludes_outer_and_labeled_domains(
    monkeypatch,
) -> None:
    protocol = load_g1_protocol(CONFIG, project_root=ROOT)
    outer = protocol.domain_order[-1]
    labeled = protocol.domain_order[0]
    runtime = _runtime(protocol.domain_order)
    rows = _assessor_rows(
        protocol.domain_order,
        outer_target=outer,
        labeled_domain=labeled,
    )
    monkeypatch.setattr(
        g1_module,
        "_build_crossfit_assessor_rows",
        lambda *_args, **_kwargs: rows,
    )

    dependencies = build_g1_source_dependencies(
        runtime,
        protocol,
        outer_target=outer,
        labeled_domain=labeled,
        encoder=object(),
    )

    expected = tuple(
        domain for domain in protocol.domain_order if domain not in {outer, labeled}
    )
    assert dependencies.roster.fit_domains == expected
    assert dependencies.prior_fit.prior.source_domains == expected
    assert dependencies.assessor_fit.assessor.fit_domains == expected
    assert set(expected).isdisjoint({outer, labeled})
    assert dependencies.assessor_row_count == 4 * 2 * 13
    assert len(dependencies.state_sha256) == 64
