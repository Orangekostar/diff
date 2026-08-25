from __future__ import annotations

from types import MappingProxyType

import numpy as np
import polars as pl

from cmc_bbdm.mavis.mris_data import MRISFeatureBank
from cmc_bbdm.mavis.mris_execution import run_mris_outer_domain
from cmc_bbdm.mavis.mris_training import load_fitted_mris_checkpoint


def _feature_bank() -> MRISFeatureBank:
    domains = ("d0", "d1", "d2", "d3", "d4", "d5")
    specimen_ids = tuple(f"{domain}-{index}" for domain in domains for index in range(2))
    domain_ids = tuple(domain for domain in domains for _ in range(2))
    count = len(specimen_ids)
    generator = np.random.Generator(np.random.PCG64(20260825))
    contexts = generator.normal(size=(count, 34))
    real = generator.uniform(size=(count, 64, 6)).astype(np.float32)
    positions = real.copy()
    positions[:, :, 2:5] = 0.0
    masks = np.ones((count, 64), dtype=bool)
    costs = np.tile(np.asarray([0.1, 0.9, 0.5]), (count, 1))
    shuffled = np.stack([real[::-1] for _ in domains])
    donors = MappingProxyType(
        {
            outer: tuple(
                specimen_ids[index - 1 if index % 2 else index + 1]
                for index in range(count)
            )
            for outer in domains
        }
    )
    relaxations = MappingProxyType(
        {outer: ("dataset",) * count for outer in domains}
    )
    arrays = (
        contexts,
        real,
        positions,
        masks,
        costs,
        shuffled,
        np.tile(np.linspace(0.2, 0.4, count), (6, 1)),
        np.full(count, 0.1),
        np.full(count, 10),
        np.full(count, 100),
        np.full(count, 0.1),
        np.linspace(0.2, 0.4, count),
    )
    for value in arrays:
        value.setflags(write=False)
    return MRISFeatureBank(
        domain_order=domains,
        state_ids=tuple(f"state-{index}" for index in range(count)),
        specimen_ids=specimen_ids,
        domain_ids=domain_ids,
        trajectory_ids=tuple(f"trajectory-{index}" for index in range(count)),
        methods=("uniform",) * count,
        seeds=(None,) * count,
        nominal_checkpoints=arrays[7],
        exact_acquired_costs=arrays[8],
        native_counts=arrays[9],
        effective_budgets=arrays[10],
        context_features=arrays[0],
        real_token_features=arrays[1],
        positions_token_features=arrays[2],
        token_masks=arrays[3],
        cost_features=arrays[4],
        shuffled_token_features=arrays[5],
        donor_specimen_ids=donors,
        donor_relaxations=relaxations,
        reconstruction_predictions=arrays[6],
        targets=arrays[11],
        input_state_sha256="a" * 64,
        target_state_sha256="b" * 64,
    )


def test_mavis_outer_worker_uses_inner_folds_then_predicts_target_only(
    tmp_path,
) -> None:
    bank = _feature_bank()

    complete = run_mris_outer_domain(
        bank,
        outer_domain="d0",
        output_root=tmp_path,
        trainable_modes=("real",),
        hidden_dimension=8,
        mris_dimension=8,
        learning_rate=0.001,
        max_epochs=2,
        patience=1,
        batch_size=4,
        seed=20260825,
        device="cpu",
    )

    predictions = pl.read_parquet(tmp_path / "d0/predictions.parquet")
    audits = pl.read_parquet(tmp_path / "d0/model_selection_audit.parquet")
    checkpoint = tmp_path / "d0/checkpoints/real.npz"
    assert complete == tmp_path / "d0/complete.json"
    assert set(predictions.get_column("mode")) == {"real", "reconstruction"}
    assert set(predictions.get_column("outer_domain")) == {"d0"}
    assert set(predictions.get_column("specimen_id")) == {"d0-0", "d0-1"}
    assert audits.filter(pl.col("record_type") == "inner_fold").height == 5
    assert audits.filter(pl.col("record_type") == "final_refit").height == 1
    inner_checkpoints = sorted((tmp_path / "d0/checkpoints/inner").glob("*.npz"))
    assert [path.stem for path in inner_checkpoints] == [
        f"{domain}__real" for domain in bank.domain_order if domain != "d0"
    ]
    for path in inner_checkpoints:
        inner = load_fitted_mris_checkpoint(path)
        validation_domain = path.stem.split("__", 1)[0]
        assert inner.audit.validation_domains == (validation_domain,)
        assert validation_domain not in inner.audit.fit_domains
    restored = load_fitted_mris_checkpoint(checkpoint)
    assert restored.mode == "real"
    assert restored.outer_domain == "d0"
