"""Typed identities for the CAI active-image v2 protocol."""

from __future__ import annotations

from enum import StrEnum

INITIAL_PROPOSAL_RULE = "HIGHEST_RELIABLE_CONFIDENCE_C0_THEN_UNLOCK_V1"
ACTOR_STATE_PROTOCOL = "VISIBLE_ORDERED_HISTORY_BUDGET_V1"


class Method(StrEnum):
    VLM_CAI_FEEDBACK_AGENT = "VLM_CAI_FEEDBACK_AGENT"
    NO_VLM_FEEDBACK = "NO_VLM_FEEDBACK"
    VLM_OPEN_LOOP = "VLM_OPEN_LOOP"
    LEARNED_STATIC = "LEARNED_STATIC"
    SERPENTINE = "SERPENTINE"
    CENTER_FIRST = "CENTER_FIRST"
    GEOMETRY_SPREAD = "GEOMETRY_SPREAD"
    RANDOM = "RANDOM"

    @property
    def uses_vlm(self) -> bool:
        return self in {
            Method.VLM_CAI_FEEDBACK_AGENT,
            Method.VLM_OPEN_LOOP,
        }

    @property
    def uses_feedback(self) -> bool:
        return self in {
            Method.VLM_CAI_FEEDBACK_AGENT,
            Method.NO_VLM_FEEDBACK,
        }

    @property
    def learned(self) -> bool:
        return self in {
            Method.VLM_CAI_FEEDBACK_AGENT,
            Method.NO_VLM_FEEDBACK,
            Method.VLM_OPEN_LOOP,
            Method.LEARNED_STATIC,
        }


__all__ = ["ACTOR_STATE_PROTOCOL", "INITIAL_PROPOSAL_RULE", "Method"]
