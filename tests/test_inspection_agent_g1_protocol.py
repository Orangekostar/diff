from __future__ import annotations

from pathlib import Path

import pytest

from cmc_bbdm.inspection_agent_g1.g1 import G1ExecutionError, load_g1_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/inspection_agent_g1.yaml"


def test_g1_protocol_is_hash_bound_and_exact() -> None:
    protocol = load_g1_protocol(CONFIG, project_root=ROOT)
    assert protocol.config_sha256 == (
        "aaf216ab9033fffc390f123131bca29952b0d5ca34752434aa9686c1d7ff1f05"
    )
    assert protocol.domain_order == (
        "74t7kcdgkr",
        "cgtnjyggtm",
        "w68dtmpfyf",
        "xcmzfsbd9t",
        "yfxyg8jm46",
        "ykhs7s2dck",
    )
    assert protocol.specimen_count == 276
    assert protocol.evaluation_checkpoints == (0.0, 0.0625, 0.125, 0.1875, 0.25)
    assert protocol.teacher_temperatures == (0.25, 0.5, 1.0, 2.0)
    assert protocol.dagger_iterations == (0, 1, 2)
    assert protocol.stop_thresholds == (0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99)
    assert protocol.bootstrap_replicates == 100_000
    assert protocol.bootstrap_seed == 2026090104
    assert protocol.default_device == "cuda:2"


def test_g1_protocol_rejects_any_config_byte_change(tmp_path: Path) -> None:
    changed = tmp_path / "inspection_agent_g1.yaml"
    changed.write_bytes(CONFIG.read_bytes().replace(b"epochs: 80", b"epochs: 79"))
    with pytest.raises(G1ExecutionError, match="SHA-256"):
        load_g1_protocol(changed, project_root=ROOT)
