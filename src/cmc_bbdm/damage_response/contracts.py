from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

PRIMARY_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "74t7kcdgkr": 45,
        "cgtnjyggtm": 49,
        "w68dtmpfyf": 43,
        "xcmzfsbd9t": 59,
        "yfxyg8jm46": 42,
        "ykhs7s2dck": 38,
    }
)

LOAD_KN_PER_VOLT = 25.0
DISPLACEMENT_MM_PER_VOLT = 1.0
POST_CAI_IMAGE_INPUT_FORBIDDEN = True


class InputRole(Enum):
    """Scientific role and deployment eligibility of a named input."""

    LAMINATE = ("laminate", True)
    GEOMETRY = ("geometry", True)
    SURFACE_PROFILE = ("surface_profile", True)
    CSCAN = ("cscan", True)
    IMPACT_ENERGY = ("impact_energy", False)
    IMPACTOR = ("impactor", False)
    TRUE_CAI_TRACE = ("true_cai_trace", False)
    DERIVED_RESPONSE = ("derived_response", False)
    TRUE_PEAK_STRENGTH = ("true_peak_strength", False)
    POST_CAI_IMAGE = ("post_cai_image", False)

    def __init__(self, input_name: str, deployable: bool) -> None:
        self.input_name = input_name
        self.deployable = deployable


_ROLES_BY_NAME = {role.input_name: role for role in InputRole}
_FORBIDDEN_ROLES = frozenset(
    {
        InputRole.TRUE_CAI_TRACE,
        InputRole.DERIVED_RESPONSE,
        InputRole.TRUE_PEAK_STRENGTH,
        InputRole.POST_CAI_IMAGE,
    }
)
_PRIVILEGED_ROLES = frozenset({InputRole.IMPACT_ENERGY, InputRole.IMPACTOR})


def validate_input_names(
    names: Iterable[str], *, allow_privileged: bool = False
) -> tuple[str, ...]:
    """Validate an explicit feature view without silently changing its members."""

    validated = tuple(names)
    if not validated:
        raise ValueError("at least one input name is required")
    if len(set(validated)) != len(validated):
        raise ValueError("duplicate input names are forbidden")

    for name in validated:
        role = _ROLES_BY_NAME.get(name)
        if role is None:
            raise ValueError(f"unknown input name: {name!r}")
        if role in _FORBIDDEN_ROLES:
            detail = "post-CAI" if role is InputRole.POST_CAI_IMAGE else "true response"
            raise ValueError(f"forbidden {detail} input: {name}")
        if role in _PRIVILEGED_ROLES and not allow_privileged:
            raise ValueError(f"privileged sensitivity input is not deployable: {name}")
    return validated


class StageStatus(str, Enum):
    P0_GO = "P0_GO"
    P0_NO_GO = "P0_NO_GO"
    P0_REQUIRES_HUMAN_REVIEW = "P0_REQUIRES_HUMAN_REVIEW"
    P1_GO = "P1_GO"
    RESPONSE_BEYOND_STRENGTH_NO_GO = "RESPONSE_BEYOND_STRENGTH_NO_GO"
    NOT_RUN_NOT_AUTHORIZED = "NOT_RUN_NOT_AUTHORIZED"


@dataclass(frozen=True)
class P0GateFacts:
    exact_identity_pairing_possible: bool
    exact_pair_counts: Mapping[str, int]
    identity_guessed: bool
    all_sources_hash_bound: bool
    peak_reconciliation_passed: bool
    missing_primary_channel_fractions: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "exact_pair_counts", MappingProxyType(dict(self.exact_pair_counts))
        )
        object.__setattr__(
            self,
            "missing_primary_channel_fractions",
            MappingProxyType(dict(self.missing_primary_channel_fractions)),
        )


@dataclass(frozen=True)
class GateDecision:
    status: StageStatus
    reasons: tuple[str, ...]


def evaluate_p0_gate(facts: P0GateFacts) -> GateDecision:
    """Apply preregistered P0 stop and review criteria with NO-GO precedence."""

    no_go: list[str] = []
    review: list[str] = []
    expected_domains = set(PRIMARY_COUNTS)
    count_domains = set(facts.exact_pair_counts)
    channel_domains = set(facts.missing_primary_channel_fractions)

    if not facts.exact_identity_pairing_possible:
        no_go.append("exact identity pairing is not possible")
    if facts.identity_guessed:
        no_go.append("guessed identity was used")
    if not facts.all_sources_hash_bound:
        no_go.append("source identity is not hash-bound")
    if not facts.peak_reconciliation_passed:
        no_go.append("published peak reconciliation failed")

    missing_count_domains = sorted(expected_domains - count_domains)
    if missing_count_domains:
        no_go.append(
            "missing primary domain pair counts: " + ", ".join(missing_count_domains)
        )
    missing_channel_domains = sorted(expected_domains - channel_domains)
    if missing_channel_domains:
        no_go.append(
            "missing primary domain channel facts: "
            + ", ".join(missing_channel_domains)
        )
    unexpected_domains = sorted(
        (count_domains | channel_domains) - expected_domains
    )
    if unexpected_domains:
        no_go.append("unexpected domain facts: " + ", ".join(unexpected_domains))

    for domain in sorted(expected_domains & count_domains):
        count = facts.exact_pair_counts[domain]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            no_go.append(f"invalid exact-pair count for {domain}: {count!r}")
        elif count < 20:
            review.append(f"{domain} has fewer than 20 exact pairs: {count}")

    for domain in sorted(expected_domains & channel_domains):
        fraction = facts.missing_primary_channel_fractions[domain]
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not 0.0 <= fraction <= 1.0
        ):
            no_go.append(f"invalid missing-channel fraction for {domain}: {fraction!r}")
        elif fraction > 0.2:
            review.append(
                f"{domain} has more than 20% missing primary channels: {fraction:.6f}"
            )

    if no_go:
        return GateDecision(StageStatus.P0_NO_GO, tuple(no_go))
    if review:
        return GateDecision(StageStatus.P0_REQUIRES_HUMAN_REVIEW, tuple(review))
    return GateDecision(StageStatus.P0_GO, ())
