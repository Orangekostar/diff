from __future__ import annotations

from pathlib import Path

import pytest

from cmc_bbdm.mva.a5_config import A5ConfigError, load_a5_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/mva_a5_imitation_policy.yaml"


def test_a5_config_freezes_outer_safe_imitation_protocol() -> None:
    config = load_a5_config(CONFIG, project_root=ROOT)

    assert config.a5_authorization_status == "MVA_A5_AUTHORIZED"
    assert config.a4_global_status == "MVA_A4_GLOBAL_NO_GO"
    assert config.teacher_split == "leave_outer_and_query_domain_out"
    assert config.teacher_trajectory_batch_size == 8
    assert config.state_dimension == 579
    assert config.candidate_dimension == 8
    assert config.hidden_dimensions == ((64, 32), (32, 16), (32,))
    assert config.maximum_parameters == 50000
    assert config.epochs == 50
    assert config.batch_states == 128
    assert config.learning_rate == 1.0e-3
    assert config.weight_decay == 1.0e-4
    assert config.minimum_improved_domains == 4
    assert config.minimum_gap_closure == 0.20
    assert config.bootstrap_resamples == 100000
    assert config.output_dir == Path("results/mva/a5_imitation_policy")
    assert all(not source.path.is_absolute() for source in config.sources.values())


def test_a5_config_rejects_authorization_or_architecture_drift(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    unauthorized = tmp_path / "unauthorized.yaml"
    unauthorized.write_text(
        text.replace("a5_status: MVA_A5_AUTHORIZED", "a5_status: MVA_A5_NOT_AUTHORIZED"),
        encoding="utf-8",
    )
    with pytest.raises(A5ConfigError, match="authorization"):
        load_a5_config(unauthorized, project_root=ROOT)

    oversized = tmp_path / "oversized.yaml"
    oversized.write_text(
        text.replace("maximum_parameters: 50000", "maximum_parameters: 200000"),
        encoding="utf-8",
    )
    with pytest.raises(A5ConfigError, match="policy|architecture"):
        load_a5_config(oversized, project_root=ROOT)


def test_a5_config_rejects_unknown_rescue_stage(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    path = tmp_path / "rescue.yaml"
    path.write_text(text + "\nreinforcement_learning:\n  enabled: true\n", encoding="utf-8")

    with pytest.raises(A5ConfigError, match="keys"):
        load_a5_config(path, project_root=ROOT)


def test_a5_runner_forces_blas_threads_before_mva_imports() -> None:
    text = (ROOT / "scripts/run_mva_a5.py").read_text(encoding="utf-8")

    assert "setdefault" not in text
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        assert f'"{variable}"' in text
    assert text.index('os.environ[variable] = "1"') < text.index(
        "from cmc_bbdm.mva.a5_artifacts import"
    )
