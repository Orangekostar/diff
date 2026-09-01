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
from .dagger import (
    DAGGER_MAX_NEW_STATES,
    REGISTERED_DAGGER_ITERATIONS,
    DaggerSourceError,
    DaggerVisitedState,
    equal_specimen_task_weights,
    select_source_relabels,
    trajectory_quantile_indices,
)
from .features import (
    build_policy_state,
    canonical_action_from_slot,
    canonical_slot,
    decision_type,
)
from .policy_model import SharedActionMLP, StructuredInspectionPolicy
from .policy_training import hard_behavior_cloning_loss
from .privileged_awr import (
    AAWRAuthorization,
    AAWRSourceEvidence,
    PrivilegedValueNetwork,
    advantage_weighted_actor_loss,
    authorize_conditional_aawr,
)
from .stopping_policy import (
    SourceStopLabel,
    StopThresholdSelection,
    build_source_stop_label,
    observable_stop_loss,
    select_conservative_stop_threshold,
)
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
    "DAGGER_MAX_NEW_STATES",
    "DEPLOYMENT_INITIAL_NOMINAL_BUDGET",
    "PRIMARY_WARM_START_CELLS",
    "PRIMARY_WARM_START_K",
    "REGISTERED_DAGGER_ITERATIONS",
    "SENSITIVITY_WARM_START_K",
    "AAWRAuthorization",
    "AAWRSourceEvidence",
    "CAIContextMode",
    "ContinuationPolicy",
    "CrossfitCAIAssessorFit",
    "CrossfitPriorFit",
    "CrossfitRoster",
    "DaggerSourceError",
    "DaggerVisitedState",
    "DeploymentGeometryAudit",
    "G1CrossfitError",
    "G1PolicyState",
    "PrivilegedTeacherLabel",
    "PrivilegedValueNetwork",
    "SharedActionMLP",
    "SourceStopLabel",
    "StopThresholdSelection",
    "StructuredInspectionPolicy",
    "TaskTokenMode",
    "WarmStartAudit",
    "advantage_weighted_actor_loss",
    "apply_warm_start",
    "audit_deployment_geometry",
    "authorize_conditional_aawr",
    "authorize_source_teacher",
    "build_crossfit_roster",
    "build_deployment_grid",
    "build_policy_state",
    "build_source_stop_label",
    "cai_teacher_label",
    "canonical_action_from_slot",
    "canonical_slot",
    "decision_type",
    "equal_specimen_task_weights",
    "field_teacher_label",
    "fit_crossfit_cai_assessor",
    "fit_crossfit_source_prior",
    "hard_behavior_cloning_loss",
    "materialize_label_independent_states",
    "materialize_oracle_checkpoint_states",
    "observable_stop_loss",
    "select_conservative_stop_threshold",
    "select_source_relabels",
    "soft_utility_distillation_loss",
    "teacher_distribution",
    "trajectory_quantile_indices",
    "warm_start_audit",
]
