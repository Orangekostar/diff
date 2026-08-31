from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cmc_bbdm.agentic_nde.p1 import P1ConfigError, load_p1_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/agentic_nde_p1_visual_observability.yaml"


def _payload() -> dict[str, object]:
    payload = yaml.safe_load(CONFIG.read_text(encoding="ascii"))
    assert isinstance(payload, dict)
    return payload


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "p1.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="ascii")
    return path


def test_p1_config_freezes_visual_observability_protocol() -> None:
    config = load_p1_config(CONFIG, project_root=ROOT)

    assert config.config_sha256 == (
        "00f3e0cf45d45dd64c20852513d9b23c69a3c29ad7e0d0d7220fb13f86bfe92e"
    )
    assert config.domain_order == (
        "74t7kcdgkr",
        "cgtnjyggtm",
        "w68dtmpfyf",
        "xcmzfsbd9t",
        "yfxyg8jm46",
        "ykhs7s2dck",
    )
    assert config.authorized_specimen_count == 276
    assert config.authorized_roster_sha256 == (
        "4fd8c6076dd3fcdf908a73739251db215fcb01f570f1a930b7faf250fe6d285a"
    )
    assert config.registration_authority_sha256 == (
        "38ab3cf32e866cda447a5edf2637fa502406c4c5c574bc966c13cc1cbbd2553a"
    )
    assert config.target_rows == 17664
    assert config.target_path == Path("results/mva/a2_oracle_value/oracle_values.parquet")
    assert config.encoder_weight_sha256 == (
        "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
    )
    assert config.surface_transform_sha256 == (
        "2b275ebbc220e6a0376d305d0996f4ffe80509fc8b27223fd919331a100acbe5"
    )
    assert config.bootstrap_resamples == 100000
    assert config.output_result == Path(
        "results/agentic_task_driven_nde/p1_visual_observability"
    )
    assert len(config.sources) == 20


@pytest.mark.parametrize(
    "mutation",
    (
        "authorized_count",
        "orientation",
        "target_rows",
        "optional_encoder",
        "transform_hash",
        "outer_split",
        "c4_roster",
        "bootstrap_resamples",
        "remove_source",
    ),
)
def test_p1_config_rejects_frozen_protocol_mutations(
    tmp_path: Path, mutation: str
) -> None:
    payload = _payload()

    if mutation == "authorized_count":
        payload["p0r_authority"]["authorized_specimen_count"] = 275
    elif mutation == "orientation":
        payload["p0r_authority"]["orientation"] = "IDENTITY"
    elif mutation == "target_rows":
        payload["target"]["expected_rows"] = 17663
    elif mutation == "optional_encoder":
        payload["surface_features"]["optional_encoder_roster"].append("dinov2")
    elif mutation == "transform_hash":
        payload["surface_features"]["transform"]["sha256"] = "0" * 64
    elif mutation == "outer_split":
        payload["models"]["outer_split"] = "leave_two_domains_out"
    elif mutation == "c4_roster":
        payload["controls"]["C4"]["roster"].append("ROT90")
    elif mutation == "bootstrap_resamples":
        payload["bootstrap"]["resamples"] = 99999
    elif mutation == "remove_source":
        del payload["sources"]["p0r_summary"]
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(P1ConfigError):
        load_p1_config(_write(tmp_path, payload), project_root=ROOT)


def test_p1_config_rejects_duplicate_yaml_key(tmp_path: Path) -> None:
    changed = tmp_path / "duplicate.yaml"
    changed.write_text(
        CONFIG.read_text(encoding="ascii") + "\nschema_version: 1\n",
        encoding="ascii",
    )

    with pytest.raises(P1ConfigError):
        load_p1_config(changed, project_root=ROOT)


def test_p1_config_rejects_p0r_summary_source_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["sources"]["p0r_summary"]["sha256"] = "0" * 64

    with pytest.raises(P1ConfigError):
        load_p1_config(_write(tmp_path, payload), project_root=ROOT)
