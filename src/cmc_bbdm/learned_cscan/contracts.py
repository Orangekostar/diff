"""Typed contracts for the learned C-scan study."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Split(StrEnum):
    TRAIN = "TRAIN"
    VALID = "VALID"
    TEST = "TEST"


class Task(StrEnum):
    LOCATE = "LOCATE"
    CHARACTERIZE = "CHARACTERIZE"


class ReferenceStatus(StrEnum):
    ALGORITHM_DERIVED_NOT_REVIEWED = "ALGORITHM_DERIVED_NOT_REVIEWED"
    EXPERT_REVIEWED = "EXPERT_REVIEWED"


class ReviewState(StrEnum):
    PENDING = "pending"
    REVIEWED = "reviewed"


@dataclass(frozen=True, slots=True)
class ReferenceEvidence:
    status: ReferenceStatus
    review_state: ReviewState
    reviewer_alias: str | None

    def __post_init__(self) -> None:
        if self.status is ReferenceStatus.EXPERT_REVIEWED:
            if self.review_state is not ReviewState.REVIEWED or not self.reviewer_alias:
                raise ValueError("expert reference requires review provenance")
        elif self.review_state is not ReviewState.PENDING or self.reviewer_alias is not None:
            raise ValueError("algorithm-derived reference cannot claim review provenance")

    @property
    def formal_eligible(self) -> bool:
        return (
            self.status is ReferenceStatus.EXPERT_REVIEWED
            and self.review_state is ReviewState.REVIEWED
            and bool(self.reviewer_alias)
        )


@dataclass(frozen=True, slots=True)
class ReferenceScore:
    reference_eligible: bool
    proxy_success: bool
    proxy_scope: str
    formal_success: bool | None


def score_reference_evidence(
    evidence: ReferenceEvidence, *, proxy_success: bool
) -> ReferenceScore:
    if type(evidence) is not ReferenceEvidence or type(proxy_success) is not bool:
        raise TypeError("typed reference evidence and proxy outcome are required")
    eligible = evidence.formal_eligible
    return ReferenceScore(
        reference_eligible=eligible,
        proxy_success=proxy_success,
        proxy_scope=(
            "REVIEWED_REFERENCE"
            if eligible
            else "SAME_READER_SELF_CONSISTENCY"
        ),
        formal_success=proxy_success if eligible else None,
    )


__all__ = [
    "ReferenceEvidence",
    "ReferenceScore",
    "ReferenceStatus",
    "ReviewState",
    "Split",
    "Task",
    "score_reference_evidence",
]
