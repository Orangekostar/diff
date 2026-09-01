"""Observable task-conditioned inspection policy research stage."""

from .contracts import CAIContextMode, G1PolicyState, TaskTokenMode
from .crossfit import (
    CrossfitCAIAssessorFit,
    CrossfitPriorFit,
    CrossfitRoster,
    G1CrossfitError,
    build_crossfit_roster,
    fit_crossfit_cai_assessor,
    fit_crossfit_source_prior,
)
from .features import (
    build_policy_state,
    canonical_action_from_slot,
    canonical_slot,
    decision_type,
)
from .policy_model import SharedActionMLP, StructuredInspectionPolicy
from .warm_start import (
    DEPLOYMENT_INITIAL_NOMINAL_BUDGET,
    PRIMARY_WARM_START_CELLS,
    PRIMARY_WARM_START_K,
    SENSITIVITY_WARM_START_K,
    DeploymentGeometryAudit,
    WarmStartAudit,
    apply_warm_start,
    audit_deployment_geometry,
    build_deployment_grid,
    warm_start_audit,
)

__all__ = [
    "DEPLOYMENT_INITIAL_NOMINAL_BUDGET",
    "PRIMARY_WARM_START_CELLS",
    "PRIMARY_WARM_START_K",
    "SENSITIVITY_WARM_START_K",
    "CAIContextMode",
    "CrossfitCAIAssessorFit",
    "CrossfitPriorFit",
    "CrossfitRoster",
    "DeploymentGeometryAudit",
    "G1CrossfitError",
    "G1PolicyState",
    "SharedActionMLP",
    "StructuredInspectionPolicy",
    "TaskTokenMode",
    "WarmStartAudit",
    "apply_warm_start",
    "audit_deployment_geometry",
    "build_crossfit_roster",
    "build_deployment_grid",
    "build_policy_state",
    "canonical_action_from_slot",
    "canonical_slot",
    "decision_type",
    "fit_crossfit_cai_assessor",
    "fit_crossfit_source_prior",
    "warm_start_audit",
]
