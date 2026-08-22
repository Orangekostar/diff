from __future__ import annotations

from pathlib import Path

from test_msss_s1 import GROUPS, ROOT, synthetic_s1_run

from cmc_bbdm.msss.artifacts import publish_s1_package
from cmc_bbdm.msss.transfer_artifacts import (
    MANDATORY_S2_FILES,
    publish_s2_package,
    replay_s2_package,
    validate_s2_package,
)
from cmc_bbdm.msss.transfer_pipeline import summarize_s2
from cmc_bbdm.msss.transfer_tasks import (
    TransferPrediction,
    TransferSelection,
    TransferTask,
    TransferTaskEvaluation,
)


def _evaluations():
    tasks = [
        ("domain", f"domain:{group}", group, group) for group in GROUPS
    ] + [
        ("ply", f"ply:{value}", str(value), GROUPS[index])
        for index, value in enumerate((8, 16, 24))
    ] + [
        ("layup", f"layup:{value}", value, GROUPS[index])
        for index, value in enumerate(("cross_ply", "quasi_isotropic"))
    ]
    output = []
    for index, (family, task_id, label, dataset) in enumerate(tasks):
        task = TransferTask(
            family=family,
            task_id=task_id,
            target_label=label,
            source_indices=(1, 2, 3),
            target_indices=(0,),
            source_domains=GROUPS[:3],
            target_domains=(dataset,),
        )
        selection = TransferSelection(
            task_id=task_id,
            full_condition_id="sampling:density=1",
            fixed25_condition_id="sampling:density=0.25",
            selected_condition_id="sampling:density=0.25",
            over_coarse_condition_id="sampling:density=0.125",
            boundary_confirmed=True,
            sufficient_sets=((0.05, ("sampling:density=1", "sampling:density=0.25")),),
            candidate_scores=(("sampling:density=1", 0.1), ("sampling:density=0.25", 0.102)),
            candidate_pca_dimensions=(("sampling:density=1", 8), ("sampling:density=0.25", 8)),
        )
        target = 0.7 + index * 0.001
        errors = {"FULL": 0.1, "FIXED_25": 0.09, "SOURCE_MSSS": 0.09, "OVER_COARSE": 0.12}
        conditions = {
            "FULL": "sampling:density=1",
            "FIXED_25": "sampling:density=0.25",
            "SOURCE_MSSS": "sampling:density=0.25",
            "OVER_COARSE": "sampling:density=0.125",
        }
        predictions = tuple(
            TransferPrediction(
                task_family=family,
                task_id=task_id,
                target_label=label,
                comparator=comparator,
                condition_id=conditions[comparator],
                specimen_id=f"transfer-{index}",
                dataset_id=dataset,
                target=target,
                prediction=target + error,
                absolute_error=error,
                selected_pca_dimension=8,
                fit_state_sha256="4" * 64,
            )
            for comparator, error in errors.items()
        )
        output.append(
            TransferTaskEvaluation(
                task=task,
                selection=selection,
                predictions=predictions,
                state_sha256="5" * 64,
            )
        )
    return tuple(output)


def test_s2_package_and_replay_are_checksum_bound(tmp_path: Path) -> None:
    protocol, bank, s1_run = synthetic_s1_run()
    s1 = tmp_path / "s1"
    authorization = publish_s1_package(
        s1,
        protocol=protocol,
        bank=bank,
        run=s1_run,
        config_path=ROOT / "paper_v3/configs/msss.yaml",
        mode="formal",
        test_only=False,
    )
    run = summarize_s2(
        protocol,
        authorization=authorization,
        evaluations=_evaluations(),
        bootstrap_resamples=100,
    )
    output = tmp_path / "s2"
    published = publish_s2_package(
        output,
        protocol=protocol,
        run=run,
        config_path=ROOT / "paper_v3/configs/msss.yaml",
        s1_package=s1,
    )
    validated = validate_s2_package(
        output,
        project_root=ROOT,
        config_path=ROOT / "paper_v3/configs/msss.yaml",
        s1_package=s1,
    )
    replayed = replay_s2_package(
        output,
        tmp_path / "replay",
        project_root=ROOT,
        config_path=ROOT / "paper_v3/configs/msss.yaml",
        s1_package=s1,
    )

    assert published == validated == replayed
    assert MANDATORY_S2_FILES <= {path.name for path in output.iterdir()}
    assert validated.gate_status == "STRONG_GO"
