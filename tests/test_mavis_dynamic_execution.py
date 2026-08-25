from __future__ import annotations

import hashlib

import polars as pl
from test_mavis_mris_execution import _feature_bank

from cmc_bbdm.mavis.dynamic_execution import run_dynamic_outer_domain
from cmc_bbdm.mavis.dynamic_training import load_fitted_dynamic_checkpoint
from cmc_bbdm.mavis.mris_execution import run_mris_outer_domain
from cmc_bbdm.mavis.mris_training import load_fitted_mris_checkpoint


def _tables(bank):
    states: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    for state_id, specimen_id, domain_id, target in zip(
        bank.state_ids,
        bank.specimen_ids,
        bank.domain_ids,
        bank.targets,
        strict=True,
    ):
        states.append(
            {
                "state_id": state_id,
                "specimen_id": specimen_id,
                "domain_id": domain_id,
                "exact_acquired_cost": 10,
                "native_count": 100,
                "remaining_cost_to_endpoint": 20,
                "candidate_cell_indices": [0, 1],
                "candidate_from_levels": [0, 0],
                "candidate_to_levels": [1, 1],
                "candidate_exact_added_costs": [5, 5],
            }
        )
        for outer_index, outer_domain in enumerate(bank.domain_order):
            if outer_domain == domain_id:
                continue
            current = float(target + 0.1 + 0.01 * outer_index)
            for candidate_index, candidate_prediction in enumerate(
                (current - 0.05, current + 0.05)
            ):
                value = abs(float(target) - current) - abs(
                    float(target) - candidate_prediction
                )
                actions.append(
                    {
                        "state_id": state_id,
                        "specimen_id": specimen_id,
                        "domain_id": domain_id,
                        "outer_domain": outer_domain,
                        "candidate_index": candidate_index,
                        "cell_index": candidate_index,
                        "from_level": 0,
                        "to_level": 1,
                        "exact_added_cost": 5,
                        "teacher_true_cai": float(target),
                        "current_prediction": current,
                        "candidate_prediction": candidate_prediction,
                        "primary_value": value,
                        "teacher_state_sha256": hashlib.sha256(
                            f"{state_id}-{outer_domain}".encode()
                        ).hexdigest(),
                    }
                )
    return pl.DataFrame(states), pl.DataFrame(actions)


def test_dynamic_outer_worker_selects_on_sources_and_scores_target(tmp_path) -> None:
    bank = _feature_bank()
    states, actions = _tables(bank)
    p2_root = tmp_path / "p2"
    run_mris_outer_domain(
        bank,
        outer_domain="d0",
        output_root=p2_root,
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

    complete = run_dynamic_outer_domain(
        bank,
        states=states,
        actions=actions,
        outer_domain="d0",
        p2_checkpoint_root=p2_root / "d0/checkpoints",
        output_root=tmp_path / "p3",
        modes=("real",),
        hidden_dimension=8,
        learning_rate=0.001,
        max_epochs=2,
        patience=1,
        batch_size=4,
        seed=20260825,
        device="cpu",
        loss_weights={"cai": 1.0, "pair": 1.0, "list": 1.0, "value": 0.25},
        recall_k=2,
    )

    metrics = pl.read_parquet(tmp_path / "p3/d0/state_metrics.parquet")
    audits = pl.read_parquet(tmp_path / "p3/d0/model_selection_audit.parquet")
    fitted = load_fitted_dynamic_checkpoint(tmp_path / "p3/d0/checkpoints/real.npz")
    assert complete == tmp_path / "p3/d0/complete.json"
    assert set(metrics.get_column("specimen_id")) == {"d0-0", "d0-1"}
    assert audits.filter(pl.col("record_type") == "inner_fold").height == 5
    assert audits.filter(pl.col("record_type") == "final_refit").height == 1
    inner_checkpoints = sorted((tmp_path / "p3/d0/checkpoints/inner").glob("*.npz"))
    assert [path.stem for path in inner_checkpoints] == [
        f"{domain}__real" for domain in bank.domain_order if domain != "d0"
    ]
    inner_audits = audits.filter(pl.col("record_type") == "inner_fold")
    for row in inner_audits.iter_rows(named=True):
        p2_inner = load_fitted_mris_checkpoint(
            p2_root / f"d0/checkpoints/inner/{row['validation_domain']}__real.npz"
        )
        assert row["p2_model_state_sha256"] == p2_inner.model_state_sha256
    assert fitted.outer_domain == "d0"
