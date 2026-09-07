from __future__ import annotations

import numpy as np
import torch

from cmc_bbdm.learned_cscan.contracts import Task
from cmc_bbdm.learned_cscan.observation import (
    CELL_FEATURE_COUNT,
    GLOBAL_FEATURE_COUNT,
    HISTORY_FEATURE_COUNT,
    SUBBLOCK_FEATURE_COUNT,
)
from cmc_bbdm.learned_cscan.policies import LearnedCellActor
from cmc_bbdm.learned_cscan.stopping import (
    LearnedStopStatus,
    StopValidationRow,
    calibrate_learned_stop,
    classify_autonomous_outcome,
    learned_stop_decision,
)
from cmc_bbdm.learned_cscan.training import (
    PolicyTrainingExample,
    TrainingRoute,
    fit_actor,
)


def _actor_batch(batch: int = 2):
    cell_features = torch.zeros(batch, 64, CELL_FEATURE_COUNT)
    subblocks = torch.zeros(batch, 64, 16, SUBBLOCK_FEATURE_COUNT)
    global_features = torch.zeros(batch, GLOBAL_FEATURE_COUNT)
    history = torch.zeros(batch, 4, HISTORY_FEATURE_COUNT)
    legal = torch.zeros(batch, 64, dtype=torch.bool)
    legal[:, 2] = True
    legal[:, 5] = True
    return cell_features, subblocks, global_features, history, legal


def test_compact_actor_masks_illegal_cells_and_stays_under_parameter_cap() -> None:
    torch.manual_seed(7)
    model = LearnedCellActor()
    batch = _actor_batch()

    logits = model(*batch)

    assert logits.shape == (2, 64)
    assert torch.isneginf(logits[:, ~batch[-1][0]]).all()
    assert set(torch.argmax(logits, dim=1).tolist()) <= {2, 5}
    assert model.token_width == 128
    assert model.encoder_layers == 2
    assert model.attention_heads == 4
    assert model.parameter_count < 1_000_000


def test_fixed_small_bank_updates_each_minibatch_and_reduces_loss() -> None:
    rng = np.random.default_rng(11)
    examples = []
    for index in range(8):
        cells = rng.normal(0.0, 0.01, (64, CELL_FEATURE_COUNT)).astype(np.float32)
        cells[7, 15:] = 1.0
        examples.append(
            PolicyTrainingExample(
                specimen_key=f"train:specimen-{index % 2}",
                task=Task.LOCATE if index % 2 == 0 else Task.CHARACTERIZE,
                cell_features=cells,
                subblock_features=np.zeros(
                    (64, 16, SUBBLOCK_FEATURE_COUNT), dtype=np.float32
                ),
                global_features=np.zeros(GLOBAL_FEATURE_COUNT, dtype=np.float32),
                history_features=np.zeros(
                    (4, HISTORY_FEATURE_COUNT), dtype=np.float32
                ),
                legal_mask=np.ones(64, dtype=np.bool_),
                queried_cells=tuple(range(64)),
                target_probabilities=np.eye(64, dtype=np.float32)[7],
            )
        )
    model = LearnedCellActor()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    result = fit_actor(
        model,
        tuple(examples),
        route=TrainingRoute.BEHAVIOR_CLONING,
        max_steps=24,
        batch_size=4,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_clip=1.0,
        seed=13,
        device="cpu",
    )

    assert result.optimizer_steps == 24
    assert result.mini_batches == 24
    assert result.final_loss < result.initial_loss
    assert any(
        not torch.equal(before[name], value.detach())
        for name, value in model.named_parameters()
    )


def test_learned_stop_requires_valid_support_and_exhaustion_is_not_stop_success() -> None:
    rows = tuple(
        StopValidationRow(
            specimen_key=f"valid:sample-{index}",
            task=Task.LOCATE,
            probability=0.995,
            label=index == 0,
            mechanically_eligible=True,
        )
        for index in range(4)
    )

    authorization = calibrate_learned_stop(rows)
    decision = learned_stop_decision(
        probability=0.999,
        mechanically_eligible=True,
        authorization=authorization,
    )
    outcome = classify_autonomous_outcome(
        stopped=decision.should_stop,
        report_success=True,
        resource_exhausted=True,
    )

    assert authorization.status is LearnedStopStatus.NOT_AUTHORIZED
    assert authorization.threshold is None
    assert decision.should_stop is False
    assert outcome.completed is False
    assert outcome.stop_success is False
    assert outcome.failure_type == "RESOURCE_EXHAUSTED"
