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
from .policy_training import hard_behavior_cloning_loss
from .teacher import (
    PrivilegedTeacherLabel,
    authorize_source_teacher,
    cai_teacher_label,
    field_teacher_label,
)
from .teacher_bank import (
    ContinuationPolicy,
    materialize_label_independent_states,
    materialize_oracle_checkpoint_states,
)
from .utility_distillation import (
    soft_utility_distillation_loss,
    teacher_distribution,
)
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
    "ContinuationPolicy",
    "CrossfitCAIAssessorFit",
    "CrossfitPriorFit",
    "CrossfitRoster",
    "DeploymentGeometryAudit",
    "G1CrossfitError",
    "G1PolicyState",
    "PrivilegedTeacherLabel",
    "SharedActionMLP",
    "StructuredInspectionPolicy",
    "TaskTokenMode",
    "WarmStartAudit",
    "apply_warm_start",
    "audit_deployment_geometry",
    "authorize_source_teacher",
    "build_crossfit_roster",
    "build_deployment_grid",
    "build_policy_state",
    "cai_teacher_label",
    "canonical_action_from_slot",
    "canonical_slot",
    "decision_type",
    "field_teacher_label",
    "fit_crossfit_cai_assessor",
    "fit_crossfit_source_prior",
    "hard_behavior_cloning_loss",
    "materialize_label_independent_states",
    "materialize_oracle_checkpoint_states",
    "soft_utility_distillation_loss",
    "teacher_distribution",
    "warm_start_audit",
]
