"""Strict loaders for the frozen C, N, and shared P_all checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from cmc_bbdm.cai_agent_v3 import actor_training
from cmc_bbdm.cai_agent_v3.feature_bank import V3FeatureBank, load_feature_bank
from cmc_bbdm.cai_agent_v3.predictor_training import (
    _cell_costs,
    load_predictor_checkpoint,
)
from scripts.cai_c_retrain.train import load_weight_state

from .context import TaskContext, sha256_file


@dataclass
class FrozenInputs:
    bank: V3FeatureBank
    features: Any
    actor_c: torch.nn.Module
    actor_n: torch.nn.Module
    predictor: torch.nn.Module
    cell_costs: np.ndarray
    device: str
    parameter_hashes: dict[str, str]


def state_dict_sha256(model: torch.nn.Module) -> str:
    return actor_training._state_dict_sha256(model.state_dict())


def load_frozen_inputs(context: TaskContext, *, device: str) -> FrozenInputs:
    scope = context.scope
    bank = load_feature_bank(project_root=context.root)
    if {name: len(bank.indices(name)) for name in ("TRAIN", "VALID", "TEST")} != {
        "TRAIN": 161,
        "VALID": 50,
        "TEST": 65,
    }:
        raise ValueError("feature-bank cohort changed")
    features = actor_training._VLMFeatures(
        bank,
        context.path("c_release") / "vlm/vlm_actor_features_fit.csv",
        required_splits=("TRAIN", "VALID"),
    )
    train_targets = bank.targets_mpa[bank.indices("TRAIN")]
    target_mean = float(np.mean(train_targets))
    target_scale = max(float(np.std(train_targets)), 1.0)

    c_scope = scope["models"]["C"]
    c_path = context.root / c_scope["checkpoint"]
    if sha256_file(c_path) != c_scope["checkpoint_sha256"]:
        raise ValueError("C checkpoint changed")
    c_manifest_path = (
        context.path("c_release") / "w3/models/vlm_spatial_feedback/manifest.json"
    )
    c_manifest = json.loads(c_manifest_path.read_text(encoding="utf-8"))
    actor_c, _ = actor_training.seeded_actor(
        c_scope["method"],
        target_mean=target_mean,
        target_scale=target_scale,
        seed=int(c_manifest["training_seed"]),
        device=device,
    )
    loaded_c = load_weight_state(
        c_path,
        actor_c,
        expected_method=c_scope["method"],
        expected_environment=c_manifest["environment_identity"],
        expected_signature=c_manifest["input_signature"],
    )
    if loaded_c["update"] != c_scope["selected_update"]:
        raise ValueError("C checkpoint update changed")
    actor_c.eval().requires_grad_(False)

    n_scope = scope["models"]["N"]
    n_path = context.root / n_scope["checkpoint"]
    if sha256_file(n_path) != n_scope["checkpoint_sha256"]:
        raise ValueError("N checkpoint changed")
    actor_n, n_manifest = actor_training.load_actor_checkpoint(n_path, bank, device=device)
    if n_manifest["selected_update"] != n_scope["selected_update"]:
        raise ValueError("N checkpoint update changed")
    actor_n.eval().requires_grad_(False)

    predictor_path = context.path("predictor")
    if sha256_file(predictor_path) != scope["models"]["P_all_sha256"]:
        raise ValueError("P_all checkpoint changed")
    predictor, _ = load_predictor_checkpoint(predictor_path, device=device)
    predictor.eval().requires_grad_(False)
    hashes = {
        "C": state_dict_sha256(actor_c),
        "N": state_dict_sha256(actor_n),
        "P_all": state_dict_sha256(predictor),
    }
    return FrozenInputs(
        bank=bank,
        features=features,
        actor_c=actor_c,
        actor_n=actor_n,
        predictor=predictor,
        cell_costs=_cell_costs(bank),
        device=device,
        parameter_hashes=hashes,
    )


__all__ = ["FrozenInputs", "load_frozen_inputs", "state_dict_sha256"]

