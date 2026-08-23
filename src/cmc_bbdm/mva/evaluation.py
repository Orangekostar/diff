"""Pure A3 hypothesis evaluation and terminal decision."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .statistics import BootstrapEffect


@dataclass(frozen=True, slots=True)
class A3GateInputs:
    uniform_low_mae: float
    mechanical_low_mae: float
    uniform_domain_low_mae: tuple[float, ...]
    mechanical_domain_low_mae: tuple[float, ...]
    reconstruction_minus_mechanical_auebc: BootstrapEffect
    appearance_minus_mechanical_auebc: BootstrapEffect
    uniform_auebc: float
    random_median_auebc: float
    mechanical_auebc: float
    uniform_b5: float | None
    random_median_b5: float | None
    mechanical_b5: float | None


@dataclass(frozen=True, slots=True)
class A3GateDecision:
    status: str
    h1_pass: bool
    h2_pass: bool
    h3_pass: bool
    h4_pass: bool
    h1_relative_improvement: float
    h1_improved_domains: int
    h4_relative_auebc_improvement: float
    h4_auebc_pass: bool
    h4_b5_saving: float | None
    h4_b5_pass: bool


def _finite(value: object, label: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(output):
        raise ValueError(f"{label} must be finite")
    return output


def _optional_budget(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    output = _finite(value, label)
    if not 0.0 < output <= 1.0:
        raise ValueError(f"{label} must be in (0,1]")
    return output


def evaluate_a3_gate(inputs: A3GateInputs) -> A3GateDecision:
    """Apply H1-H4 exactly and return the only permitted terminal status."""

    if type(inputs) is not A3GateInputs:
        raise ValueError("issued A3 gate inputs are required")
    uniform = _finite(inputs.uniform_low_mae, "uniform low-budget MAE")
    mechanical = _finite(inputs.mechanical_low_mae, "mechanical low-budget MAE")
    if uniform <= 0.0:
        raise ValueError("uniform low-budget MAE must be positive")
    uniform_domain = tuple(
        _finite(value, "uniform domain MAE") for value in inputs.uniform_domain_low_mae
    )
    mechanical_domain = tuple(
        _finite(value, "mechanical domain MAE")
        for value in inputs.mechanical_domain_low_mae
    )
    if len(uniform_domain) != 6 or len(mechanical_domain) != 6:
        raise ValueError("H1 requires exactly six domain MAEs")
    h1_relative = float((uniform - mechanical) / uniform)
    h1_domains = sum(
        candidate < baseline
        for baseline, candidate in zip(uniform_domain, mechanical_domain, strict=True)
    )
    h1 = h1_relative >= 0.05 and h1_domains >= 4

    reconstruction = inputs.reconstruction_minus_mechanical_auebc
    appearance = inputs.appearance_minus_mechanical_auebc
    if (
        type(reconstruction) is not BootstrapEffect
        or type(appearance) is not BootstrapEffect
    ):
        raise ValueError("H2/H3 require bootstrap effects")
    h2 = reconstruction.point_estimate > 0.0 and reconstruction.lower > 0.0
    h3 = appearance.point_estimate > 0.0 and appearance.lower > 0.0

    uniform_area = _finite(inputs.uniform_auebc, "uniform AUEBC")
    random_area = _finite(inputs.random_median_auebc, "random AUEBC")
    mechanical_area = _finite(inputs.mechanical_auebc, "mechanical AUEBC")
    stronger_area = min(uniform_area, random_area)
    if stronger_area <= 0.0:
        raise ValueError("reference AUEBC must be positive")
    relative_area = float((stronger_area - mechanical_area) / stronger_area)
    h4_area = relative_area >= 0.10

    uniform_b5 = _optional_budget(inputs.uniform_b5, "uniform B5")
    random_b5 = _optional_budget(inputs.random_median_b5, "random B5")
    mechanical_b5 = _optional_budget(inputs.mechanical_b5, "mechanical B5")
    reference_budgets = tuple(
        value for value in (uniform_b5, random_b5) if value is not None
    )
    b5_saving = (
        None
        if mechanical_b5 is None or not reference_budgets
        else float(1.0 - mechanical_b5 / min(reference_budgets))
    )
    h4_b5 = b5_saving is not None and b5_saving >= 0.25
    h4 = h4_area or h4_b5
    status = "MVA_ORACLE_GO" if all((h1, h2, h3, h4)) else "MVA_ORACLE_NO_GO"
    return A3GateDecision(
        status=status,
        h1_pass=h1,
        h2_pass=h2,
        h3_pass=h3,
        h4_pass=h4,
        h1_relative_improvement=h1_relative,
        h1_improved_domains=h1_domains,
        h4_relative_auebc_improvement=relative_area,
        h4_auebc_pass=h4_area,
        h4_b5_saving=b5_saving,
        h4_b5_pass=h4_b5,
    )


__all__ = ["A3GateDecision", "A3GateInputs", "evaluate_a3_gate"]
