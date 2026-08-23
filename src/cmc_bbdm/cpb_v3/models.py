"""Fold-safe tabular candidates for the CPB v3 protocol.

The module keeps feature assembly, preprocessing, and inner selection explicit.  In
particular, no helper in this file is allowed to fit a transform on rows outside
the indices supplied by its caller.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import io
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Any
from weakref import ReferenceType, ref

import numpy as np

from .config import SURFACE_METADATA_NAMES, SURFACE_PROFILE_STAT_NAMES
from .embeddings import (
    EMBEDDING_DIMENSION,
    FoldLocalPCA,
    fit_embedding_pca,
    transform_embedding_pca,
    valid_pca_dimensions,
    validate_fold_local_pca,
)
from .embeddings import (
    PCA_DIMENSIONS as EMBEDDING_PCA_DIMENSIONS,
)
from .embeddings import (
    FeatureValidationError as EmbeddingFeatureValidationError,
)
from .embeddings import _state_hash as _embedding_state_hash
from .morphology import MORPHOLOGY_FEATURE_NAMES, MORPHOLOGY_UNITS

# V2's registered baseline has 13 metadata variables and 21 profile statistics.
METADATA_FEATURE_COUNT = 13
SURFACE_FEATURE_COUNT = 21
SCALAR_FEATURE_COUNT = 3
MORPHOLOGY_FEATURE_COUNT = 64
FROZEN_EMBEDDING_FEATURE_COUNT = EMBEDDING_DIMENSION
A_FEATURE_COUNT = METADATA_FEATURE_COUNT + SURFACE_FEATURE_COUNT
B_SCALAR_FEATURE_COUNT = A_FEATURE_COUNT + SCALAR_FEATURE_COUNT
RIDGE_ALPHA = 10.0
PCA_DIMENSIONS = tuple(EMBEDDING_PCA_DIMENSIONS)
PRIMARY_CANDIDATE_ORDER = ("B_morph", "B_frozen", "B_combined")
BASELINE_CANDIDATE_ORDER = ("A_surface", "B_scalar")
PCA_TIE_TOLERANCE = 1e-12
_FROZEN_CANDIDATES = frozenset({"B_frozen", "B_combined", "I_frozen", "I_combined"})
_PRIMARY_OUTER_CANDIDATES_ORDER = PRIMARY_CANDIDATE_ORDER
_PRIMARY_OUTER_CANDIDATES = frozenset(PRIMARY_CANDIDATE_ORDER)
_BASELINE_OUTER_CANDIDATES = frozenset(BASELINE_CANDIDATE_ORDER)
_INTERNAL_OUTER_CANDIDATES = frozenset({"I_morph", "I_frozen", "I_combined"})
_OUTER_CANDIDATES = (
    _PRIMARY_OUTER_CANDIDATES | _BASELINE_OUTER_CANDIDATES | _INTERNAL_OUTER_CANDIDATES
)
_OUTER_SELECTION_ALIASES = frozenset({"B_field_selected", "I_field_selected"})


class _DuplicateJSONKeyError(ValueError):
    pass


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, item in pairs:
        if key in payload:
            raise _DuplicateJSONKeyError(f"duplicate JSON key: {key}")
        payload[key] = item
    return payload

# The dataclasses are frozen and their arrays are read-only, but Python callers
# can still bypass both with ``object.__setattr__`` and ``setflags``.  Keep the
# construction digest outside each instance so rewriting instance fields and
# recomputing their advertised hash cannot make a state valid again.
_STATE_AUTHORITY: dict[int, tuple[ReferenceType[object], str]] = {}
_PCA_AUTHORITY: dict[
    int, tuple[ReferenceType[object], str, tuple[str, ...], tuple[str, ...]]
] = {}
_BUNDLE_AUTHORITY: dict[int, tuple[ReferenceType[object], int, str, bool]] = {}
_MATRIX_AUTHORITY: dict[int, tuple[ReferenceType[object], ReferenceType[object], str, str]] = {}
_RESPONSE_AUTHORITY: dict[int, tuple[ReferenceType[object], ReferenceType[object], str]] = {}
_BUNDLE_BY_STATE: dict[str, ReferenceType[object]] = {}
_MATRIX_BY_STATE: dict[str, ReferenceType[object]] = {}
_PCA_BUNDLE_AUTHORITY: dict[
    int,
    tuple[
        ReferenceType[object],
        ReferenceType[object],
        str,
        str,
        ReferenceType[object] | None,
        ReferenceType[object] | None,
        str,
        str,
    ],
] = {}
_OUTER_AUTHORITY: dict[
    int,
    tuple[
        ReferenceType[object],
        ReferenceType[object],
        ReferenceType[object],
        ReferenceType[object],
        ReferenceType[object],
        ReferenceType[object],
        str,
    ],
] = {}
_TEST_AUTHORITY = object()
_BUNDLE_CONSTRUCTION_CONTEXT: tuple[object, bool] | None = None
_MATRIX_CONSTRUCTION_CONTEXT: FeatureBundle | None = None
_RESPONSE_CONSTRUCTION_CONTEXT: FeatureBundle | None = None
_OUTER_CONSTRUCTION_CONTEXT: tuple[
    object, object, object, object, object
] | None = None


def _fold_state_constructor_runtime() -> tuple[Callable[[object], bool], object]:
    capability = object()

    def validate(candidate: object) -> bool:
        return candidate is capability

    return validate, capability


(
    _validate_preprocessor_constructor_capability,
    _preprocessor_fit_capability,
) = _fold_state_constructor_runtime()
(
    _validate_ridge_constructor_capability,
    _ridge_fit_capability,
) = _fold_state_constructor_runtime()
del _fold_state_constructor_runtime


def _fold_state_issuance_runtime(
    constructor: Callable[..., object],
    capability: object,
    issuer_codes: frozenset[object],
) -> Callable[..., object]:
    def issue(**fields: object) -> object:
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        if caller is None or caller.f_code not in issuer_codes:
            _raise_validation("fold state issuance requires the fit implementation")
        return constructor(**fields, _fit_capability=capability)

    return issue


def _register_state(value: object, digest: str) -> None:
    try:
        identity = id(value)
        _STATE_AUTHORITY[identity] = (
            ref(value, lambda _dead, identity=identity: _STATE_AUTHORITY.pop(
                identity, None
            )),
            digest,
        )
    except TypeError:
        # Non-weak-referenceable inputs (notably V3Data/V3DataView) are never
        # registered as model states.  Their authority is checked through the
        # typed source object retained by a production FeatureBundle.
        return


def _authority_digest(value: object, *, name: str) -> str:
    record = _STATE_AUTHORITY.get(id(value))
    if record is None or record[0]() is not value:
        _raise_validation(f"{name} has no registered construction authority")
    return record[1]


def _assert_authority(value: object, current: str, *, name: str) -> None:
    if _authority_digest(value, name=name) != current:
        _raise_validation(f"{name} construction authority does not match current state")


def _source_state(source: object, *, name: str) -> str:
    validate = getattr(source, "validate", None)
    state = getattr(source, "state_hash", None)
    if not callable(validate) or not isinstance(state, str) or len(state) != 64:
        _raise_validation(f"{name} has no validated source state")
    if _typed_v3_source(source):
        try:
            from .data import validate_issued_data_authority

            issued_state = validate_issued_data_authority(source)
        except Exception as error:
            raise FeatureValidationError(
                f"{name} is not a loader-issued V3Data/V3DataView authority"
            ) from error
        if issued_state != state:
            _raise_validation(f"{name} source state is invalid")
        return issued_state
    try:
        validate(state)
    except TypeError:
        try:
            validate()
        except Exception as error:
            raise FeatureValidationError(f"{name} source state is invalid") from error
    except Exception as error:
        raise FeatureValidationError(f"{name} source state is invalid") from error
    return state


def _typed_v3_source(value: object) -> bool:
    try:
        from .data import V3Data, V3DataView
    except (ImportError, ModuleNotFoundError):
        return False
    return isinstance(value, (V3Data, V3DataView))


def _register_bundle_authority(
    bundle: FeatureBundle, source: object, *, production: bool
) -> None:
    if production and not _typed_v3_source(source):
        _raise_validation("production feature bundles require a typed V3Data/V3DataView")
    source_state = _source_state(source, name="feature bundle source")
    identity = id(bundle)
    _BUNDLE_AUTHORITY[identity] = (
        ref(bundle, lambda _dead, identity=identity: _BUNDLE_AUTHORITY.pop(identity, None)),
        id(source),
        source_state,
        production,
    )
    _remember_bundle_state(bundle)


def _register_test_bundle_authority(bundle: FeatureBundle) -> None:
    identity = id(bundle)
    _BUNDLE_AUTHORITY[identity] = (
        ref(bundle, lambda _dead, identity=identity: _BUNDLE_AUTHORITY.pop(identity, None)),
        id(_TEST_AUTHORITY),
        "",
        False,
    )
    _remember_bundle_state(bundle)


def _remember_bundle_state(bundle: FeatureBundle) -> None:
    """Keep only a weak source lookup for trusted matrix deserialization."""

    state = bundle.state_sha256
    existing = _BUNDLE_BY_STATE.get(state)
    if existing is not None and existing() is not None:
        return

    def remove(dead: ReferenceType[object], *, state: str = state) -> None:
        if _BUNDLE_BY_STATE.get(state) is dead:
            _BUNDLE_BY_STATE.pop(state, None)

    try:
        _BUNDLE_BY_STATE[state] = ref(bundle, remove)
    except TypeError:
        return


def _resolve_bundle_state(state: object) -> FeatureBundle:
    if (
        type(state) is not str
        or len(state) != 64
        or any(value not in "0123456789abcdef" for value in state)
    ):
        _raise_validation("serialized feature matrix source digest is invalid")
    record = _BUNDLE_BY_STATE.get(state)
    bundle = None if record is None else record()
    if not isinstance(bundle, FeatureBundle):
        _raise_validation("serialized feature matrix source authority is unavailable")
    validate_feature_bundle(bundle)
    if bundle.state_sha256 != state:
        _raise_validation("serialized feature matrix source state changed")
    return bundle


def _remember_matrix_state(matrix: FeatureMatrix) -> None:
    state = matrix.state_sha256
    existing = _MATRIX_BY_STATE.get(state)
    if existing is not None and existing() is not None:
        return

    def remove(dead: ReferenceType[object], *, state: str = state) -> None:
        if _MATRIX_BY_STATE.get(state) is dead:
            _MATRIX_BY_STATE.pop(state, None)

    try:
        _MATRIX_BY_STATE[state] = ref(matrix, remove)
    except TypeError:
        return


def _resolve_matrix_state(state: object) -> FeatureMatrix | None:
    if (
        type(state) is not str
        or len(state) != 64
        or any(value not in "0123456789abcdef" for value in state)
    ):
        _raise_validation("serialized feature matrix state digest is invalid")
    record = _MATRIX_BY_STATE.get(state)
    matrix = None if record is None else record()
    if matrix is not None and not isinstance(matrix, FeatureMatrix):
        _raise_validation("serialized feature matrix authority is invalid")
    return matrix


def _mark_test_bundle(bundle: FeatureBundle) -> FeatureBundle:
    object.__setattr__(bundle, "_source_authority", _TEST_AUTHORITY)
    object.__setattr__(bundle, "source_state_sha256", "test-only")
    _register_test_bundle_authority(bundle)
    return bundle


def _bundle_authority_record(bundle: FeatureBundle) -> tuple[object, str, bool]:
    record = _BUNDLE_AUTHORITY.get(id(bundle))
    if record is None or record[0]() is not bundle:
        return (None, "", False)
    source_identity, expected_state, production = record[1:]
    source = getattr(bundle, "_source_authority", None)
    if id(source) != source_identity:
        _raise_validation("feature bundle source authority identity changed")
    if production:
        current_state = _source_state(source, name="feature bundle source")
        if current_state != expected_state:
            _raise_validation("feature bundle source authority state changed")
    return source, expected_state, production


def _bundle_authority_digest(bundle: FeatureBundle) -> str:
    record = _bundle_authority_record(bundle)
    if record[0] is None:
        _raise_validation("feature bundle has no registered source authority")
    return bundle.state_sha256


def _require_production_bundle(bundle: FeatureBundle) -> tuple[object, str]:
    source, state, production = _bundle_authority_record(bundle)
    if source is None or not production:
        _raise_validation(
            "production selectors require a FeatureBundle issued from V3Data/V3DataView authority"
        )
    return source, state


def _construct_bundle(
    *,
    source: object | None,
    production: bool,
    **values: object,
) -> FeatureBundle:
    global _BUNDLE_CONSTRUCTION_CONTEXT
    previous = _BUNDLE_CONSTRUCTION_CONTEXT
    _BUNDLE_CONSTRUCTION_CONTEXT = (source, production)
    try:
        return FeatureBundle(**values)  # type: ignore[arg-type]
    finally:
        _BUNDLE_CONSTRUCTION_CONTEXT = previous


def _strict_string_tuple(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        _raise_validation(f"{name} must be a sequence of exact strings")
    try:
        result = tuple(value)
    except (TypeError, ValueError) as error:
        raise FeatureValidationError(
            f"{name} must be a sequence of exact strings"
        ) from error
    if any(type(item) is not str or not item for item in result):
        _raise_validation(f"{name} must contain only non-empty exact strings")
    return result


@dataclass(frozen=True)
class ResponseVector:
    """Immutable response bound to one validated bundle and its row identity."""

    values: np.ndarray
    sample_ids: tuple[str, ...]
    domain_ids: tuple[str, ...]
    source_sha256: str
    state_sha256: str
    _source_authority: object = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if _RESPONSE_CONSTRUCTION_CONTEXT is None:
            _raise_validation(
                "ResponseVector must be issued by a trusted response factory"
            )
        values = _as_vector(self.values, "response values")
        sample_ids = _strict_string_tuple(self.sample_ids, name="response sample IDs")
        domain_ids = _strict_string_tuple(self.domain_ids, name="response domain IDs")
        if values.size != len(sample_ids) or values.size != len(domain_ids):
            _raise_validation("response identity does not align with values")
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.source_sha256)
        ):
            _raise_validation("response source digest is invalid")
        immutable = _readonly_array(values)
        state = _state_hash(immutable, sample_ids, domain_ids, self.source_sha256)
        if self.state_sha256 != state:
            _raise_validation("response state hash mismatch")
        object.__setattr__(self, "values", immutable)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "domain_ids", domain_ids)
        object.__setattr__(self, "state_sha256", state)
        _register_state(self, state)


def make_response_vector(
    bundle: FeatureBundle, values: Sequence[float] | np.ndarray | None = None
) -> ResponseVector:
    _require_production_bundle(bundle)
    validate_feature_bundle(bundle)
    target = bundle._target_values if values is None else values
    return _make_response_vector(bundle, target, require_real_target=True)


def _make_response_vector(
    bundle: FeatureBundle,
    values: Sequence[float] | np.ndarray,
    *,
    require_real_target: bool = False,
) -> ResponseVector:
    if not isinstance(bundle, FeatureBundle):
        _raise_validation("response source must be a FeatureBundle")
    validate_feature_bundle(bundle)
    vector = _as_vector(values, "target/response values")
    if vector.size != bundle.n_rows:
        _raise_validation("response values must align with the feature bundle")
    if require_real_target and not np.array_equal(vector, bundle._target_values):
        _raise_validation(
            "production response must match the authoritative V3 target vector"
        )
    source = _bundle_authority_digest(bundle)
    state = _state_hash(vector, bundle.sample_ids, bundle.domain_ids, source)
    global _RESPONSE_CONSTRUCTION_CONTEXT
    previous = _RESPONSE_CONSTRUCTION_CONTEXT
    _RESPONSE_CONSTRUCTION_CONTEXT = bundle
    try:
        response = ResponseVector(
            values=vector,
            sample_ids=bundle.sample_ids,
            domain_ids=bundle.domain_ids,
            source_sha256=source,
            state_sha256=state,
        )
    finally:
        _RESPONSE_CONSTRUCTION_CONTEXT = previous
    object.__setattr__(response, "_source_authority", bundle)
    response_identity = id(response)
    _RESPONSE_AUTHORITY[id(response)] = (
        ref(
            response,
            lambda _dead: _RESPONSE_AUTHORITY.pop(
                response_identity, None
            ),
        ),
        ref(
            bundle,
            lambda _dead: _RESPONSE_AUTHORITY.pop(
                response_identity, None
            ),
        ),
        state,
    )
    return response


def make_test_response_vector(
    bundle: FeatureBundle, values: Sequence[float] | np.ndarray
) -> ResponseVector:
    """Test-only adapter for synthetic bundles; production callers use real V3 data."""

    if not isinstance(bundle, FeatureBundle):
        _raise_validation("response source must be a FeatureBundle")
    source, _state, production = _bundle_authority_record(bundle)
    if source is None or production:
        _raise_validation("test response adapter requires a test-only feature bundle")
    return _make_response_vector(bundle, values)


def validate_response_vector(response: ResponseVector, bundle: FeatureBundle) -> bool:
    if not isinstance(response, ResponseVector):
        _raise_validation("response must be a typed ResponseVector")
    validate_feature_bundle(bundle)
    response_record = _RESPONSE_AUTHORITY.get(id(response))
    if response_record is None or response_record[0]() is not response:
        _raise_validation("response has no registered construction authority")
    source_bundle = response_record[1]()
    if source_bundle is not bundle:
        _raise_validation("response source authority does not match feature bundle")
    _source, _source_state_value, production = _bundle_authority_record(bundle)
    if _source is None:
        _raise_validation("response source bundle has no registered authority")
    _assert_authority(
        response,
        _state_hash(
            response.values,
            response.sample_ids,
            response.domain_ids,
            response.source_sha256,
        ),
        name="response",
    )
    if response.source_sha256 != _bundle_authority_digest(bundle):
        _raise_validation("response source does not match feature bundle authority")
    if (
        response.sample_ids != bundle.sample_ids
        or response.domain_ids != bundle.domain_ids
    ):
        _raise_validation("response identity does not match feature bundle")
    if response.state_sha256 != _state_hash(
        response.values,
        response.sample_ids,
        response.domain_ids,
        response.source_sha256,
    ):
        _raise_validation("response state hash mismatch")
    if production and not np.array_equal(response.values, bundle._target_values):
        _raise_validation(
            "response targets do not match the authoritative V3 target vector"
        )
    return True


def _response_training_digest(
    response: ResponseVector, train_indices: Sequence[int]
) -> str:
    indices = np.asarray(tuple(int(index) for index in train_indices), dtype=np.int64)
    return _state_hash(
        response.values[indices],
        tuple(response.sample_ids[index] for index in indices),
        tuple(response.domain_ids[index] for index in indices),
    )


SCALAR_FEATURE_NAMES = (
    "area_mm2",
    "height_mm",
    "width_mm",
)
SCALAR_FEATURE_UNITS = ("mm^2", "mm", "mm")

_METADATA_FEATURE_NAMES = tuple(SURFACE_METADATA_NAMES)
_SURFACE_FEATURE_NAMES = tuple(SURFACE_PROFILE_STAT_NAMES)
_SURFACE_FEATURE_UNITS = ("1",) * SURFACE_FEATURE_COUNT
_MORPHOLOGY_NAMES = tuple(MORPHOLOGY_FEATURE_NAMES)
_MORPHOLOGY_UNITS = tuple(MORPHOLOGY_UNITS[name] for name in _MORPHOLOGY_NAMES)
_FROZEN_NAMES = tuple(f"frozen_pca_{index}" for index in range(max(PCA_DIMENSIONS)))

METADATA_FEATURE_NAMES = _METADATA_FEATURE_NAMES
SURFACE_FEATURE_NAMES = _SURFACE_FEATURE_NAMES


class FeatureValidationError(ValueError):
    """Raised when a registered feature or fold contract is invalid."""


class _PcaRankUnavailable(Exception):
    """Internal marker for a registered PCA dimension unavailable in one fold."""


@dataclass(frozen=True)
class _TestOnlyPCA:
    """PCA adapter for synthetic test bundles, isolated from loader PCA authority."""

    mean_: np.ndarray
    components_: np.ndarray
    explained_variance_: np.ndarray
    singular_values_: np.ndarray
    n_components_: int
    n_features_in_: int
    fit_sample_ids: tuple[str, ...]
    fit_domain_ids: tuple[str, ...]
    n_samples_in_: int
    state_sha256: str = field(init=False)
    _mean_backing: bytes = field(init=False, repr=False)
    _components_backing: bytes = field(init=False, repr=False)
    _explained_variance_backing: bytes = field(init=False, repr=False)
    _singular_values_backing: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mean = _readonly_array(self.mean_)
        components = _readonly_array(self.components_)
        explained_variance = _readonly_array(self.explained_variance_)
        singular_values = _readonly_array(self.singular_values_)
        if mean.ndim != 1 or components.ndim != 2:
            _raise_validation("test-only PCA arrays have invalid dimensions")
        if explained_variance.ndim != 1 or singular_values.ndim != 1:
            _raise_validation("test-only PCA arrays have invalid dimensions")
        if components.shape != (self.n_components_, self.n_features_in_):
            _raise_validation("test-only PCA component shape is invalid")
        if mean.shape != (self.n_features_in_,):
            _raise_validation("test-only PCA mean shape is invalid")
        if explained_variance.shape != (self.n_components_,) or singular_values.shape != (
            self.n_components_,
        ):
            _raise_validation("test-only PCA spectrum shape is invalid")
        if type(self.n_components_) is not int or self.n_components_ not in PCA_DIMENSIONS:
            _raise_validation("test-only PCA dimension is invalid")
        if type(self.n_features_in_) is not int or self.n_features_in_ != FROZEN_EMBEDDING_FEATURE_COUNT:
            _raise_validation("test-only PCA feature count is invalid")
        if type(self.n_samples_in_) is not int or self.n_samples_in_ <= self.n_components_:
            _raise_validation("test-only PCA sample count is invalid")
        ids = _ids(
            self.fit_sample_ids,
            length=self.n_samples_in_,
            name="fit_sample_ids",
        )
        domains = _ids(
            self.fit_domain_ids,
            length=self.n_samples_in_,
            name="fit_domain_ids",
        )
        for name, array in (
            ("mean", mean),
            ("components", components),
            ("explained variance", explained_variance),
            ("singular values", singular_values),
        ):
            if not np.all(np.isfinite(array)):
                _raise_validation(f"test-only PCA {name} must be finite")
        object.__setattr__(self, "mean_", mean)
        object.__setattr__(self, "components_", components)
        object.__setattr__(self, "explained_variance_", explained_variance)
        object.__setattr__(self, "singular_values_", singular_values)
        object.__setattr__(self, "fit_sample_ids", ids)
        object.__setattr__(self, "fit_domain_ids", domains)
        object.__setattr__(self, "_mean_backing", mean.tobytes(order="C"))
        object.__setattr__(self, "_components_backing", components.tobytes(order="C"))
        object.__setattr__(
            self,
            "_explained_variance_backing",
            explained_variance.tobytes(order="C"),
        )
        object.__setattr__(
            self,
            "_singular_values_backing",
            singular_values.tobytes(order="C"),
        )
        object.__setattr__(
            self,
            "state_sha256",
            _state_hash(
                mean,
                components,
                explained_variance,
                singular_values,
                self.n_components_,
                self.n_features_in_,
                self.n_samples_in_,
                ids,
                domains,
            ),
        )


def _raise_validation(message: str) -> None:
    raise FeatureValidationError(message)


def _as_matrix(
    value: Any,
    name: str,
    *,
    rows: int | None = None,
    columns: int | None = None,
    allow_nan: bool = True,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2:
        _raise_validation(f"{name} must be a two-dimensional matrix")
    if np.iscomplexobj(raw) or raw.dtype.kind in {"b", "O", "S", "U"}:
        _raise_validation(f"{name} cannot be object, string, complex, or boolean")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise FeatureValidationError(f"{name} must be numeric") from error
    if rows is not None and result.shape[0] != rows:
        _raise_validation(f"{name} row count does not match the feature bundle")
    if columns is not None and result.shape[1] != columns:
        _raise_validation(f"{name} must have exactly {columns} columns")
    if np.any(np.isinf(result)) or (not allow_nan and np.any(np.isnan(result))):
        _raise_validation(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _as_vector(
    value: Any,
    name: str,
    *,
    allow_nan: bool = False,
) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise FeatureValidationError(f"{name} must be numeric") from error
    if raw.ndim != 1:
        _raise_validation(f"{name} must be a one-dimensional vector")
    kind = raw.dtype.kind
    if kind == "b":
        _raise_validation(f"{name} cannot be boolean")
    if kind == "c":
        _raise_validation(f"{name} cannot be complex")
    if kind in "SU" or kind not in "iufO":
        _raise_validation(f"{name} must be real numeric")
    if kind == "O":
        for item in raw:
            if isinstance(item, (bool, np.bool_)):
                _raise_validation(f"{name} cannot contain boolean values")
            if isinstance(item, (complex, np.complexfloating)):
                _raise_validation(f"{name} cannot contain complex values")
            if not isinstance(item, (int, float, np.integer, np.floating)):
                _raise_validation(f"{name} must contain only real numeric values")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise FeatureValidationError(f"{name} must be numeric") from error
    if np.any(np.isinf(result)) or (not allow_nan and np.any(np.isnan(result))):
        _raise_validation(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _readonly_array(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _ids(
    value: Sequence[str] | np.ndarray | None, *, length: int, name: str
) -> tuple[str, ...]:
    if value is None:
        _raise_validation(f"{name} is required")
    if isinstance(value, (str, bytes)):
        _raise_validation(f"{name} must be a sequence of strings")
    result = tuple(value)
    if len(result) != length or any(
        not isinstance(item, str) or not item for item in result
    ):
        _raise_validation(f"{name} must align with the fitted rows")
    if len(set(result)) != len(result) and name.endswith("sample_ids"):
        _raise_validation(f"{name} must be unique")
    return result


def _indices(
    value: Sequence[int] | np.ndarray | None, *, length: int, name: str
) -> np.ndarray:
    if value is None:
        result = np.arange(length, dtype=np.int64)
    else:
        raw = np.asarray(value)
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            _raise_validation(f"{name} must be a one-dimensional integer index array")
        result = np.asarray(raw, dtype=np.int64)
    if result.size == 0:
        _raise_validation(f"{name} cannot be empty")
    if np.any(result < 0) or np.any(result >= length):
        _raise_validation(f"{name} contains an out-of-range row")
    if len(np.unique(result)) != len(result):
        _raise_validation(f"{name} must not contain duplicate rows")
    return result


_MAX_SERIALIZED_JSON_BYTES = 64 * 1024 * 1024
_MAX_SERIALIZED_BASE64_CHARS = 64 * 1024 * 1024
_MAX_SERIALIZED_ARRAYS = 32
_MAX_SERIALIZED_ARRAY_NDIM = 4
_MAX_SERIALIZED_ARRAY_ELEMENTS = 10_000_000
_MAX_SERIALIZED_ARRAY_BYTES = 64 * 1024 * 1024
_MAX_SERIALIZED_NPY_HEADER_BYTES = 1 * 1024 * 1024


@dataclass
class _ArrayDecodeBudget:
    count: int = 0

    def claim(self, name: str) -> None:
        self.count += 1
        if self.count > _MAX_SERIALIZED_ARRAYS:
            _raise_validation(f"serialized {name} array count exceeds the safety cap")


def _require_json_schema(
    payload: Mapping[str, object],
    *,
    name: str,
    schema: Mapping[str, object],
) -> None:
    if not isinstance(payload, Mapping):
        _raise_validation(f"serialized {name} payload is invalid")
    try:
        keys = set(payload)
    except (TypeError, ValueError) as error:
        raise FeatureValidationError(f"serialized {name} payload is invalid") from error
    expected = set(schema)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        _raise_validation(
            f"serialized {name} schema mismatch (missing={missing}, unknown={unknown})"
        )
    for key, expected_type in schema.items():
        value = payload[key]
        if expected_type is object:
            continue
        if expected_type is float:
            valid = type(value) is float
        elif isinstance(expected_type, tuple):
            valid = any(
                (type(value) is item if item in (int, float, str, bool) else isinstance(value, item))
                for item in expected_type
            )
        elif expected_type is int:
            valid = type(value) is int
        elif expected_type is str:
            valid = type(value) is str
        elif expected_type is list:
            valid = type(value) is list
        else:
            valid = isinstance(value, expected_type)
        if not valid:
            _raise_validation(f"serialized {name} field {key!r} has an invalid type")


def _require_string_list(value: object, *, name: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        _raise_validation(f"serialized {name} must be a list of strings")
    return value  # type: ignore[return-value]


def _decode_base64_bytes(value: object, *, name: str) -> bytes:
    if type(value) is not str or len(value) > _MAX_SERIALIZED_BASE64_CHARS:
        _raise_validation(f"serialized {name} payload is too large or invalid")
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError, TypeError) as error:
        raise FeatureValidationError(f"serialized {name} payload is invalid") from error
    if len(raw) > _MAX_SERIALIZED_ARRAY_BYTES:
        _raise_validation(f"serialized {name} payload exceeds the safety cap")
    return raw


def _array_digest(digest: Any, label: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(label.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))


def _state_hash(*parts: Any) -> str:
    digest = hashlib.sha256()
    for index, part in enumerate(parts):
        if isinstance(part, np.ndarray):
            _array_digest(digest, f"array:{index}", part)
        else:
            digest.update(f"part:{index}:".encode("ascii"))
            digest.update(
                json.dumps(
                    part, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                ).encode("ascii")
            )
    return digest.hexdigest()


def _encode_array(value: np.ndarray) -> str:
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _decode_array(
    value: object,
    *,
    name: str,
    budget: _ArrayDecodeBudget | None = None,
    expected_dtype: np.dtype[Any] | None = None,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if budget is None:
        budget = _ArrayDecodeBudget()
    budget.claim(name)
    raw = _decode_base64_bytes(value, name=f"{name} array")
    if len(raw) < 10:
        _raise_validation(f"serialized {name} array is invalid")
    if raw[:2] == b"PK":
        _raise_validation(f"serialized {name} array must be an NPY payload")
    stream = io.BytesIO(raw)
    try:
        major, _minor = np.lib.format.read_magic(stream)
        if major == 1:
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(
                stream, max_header_size=_MAX_SERIALIZED_NPY_HEADER_BYTES
            )
        elif major == 2 or major == 3:
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(
                stream, max_header_size=_MAX_SERIALIZED_NPY_HEADER_BYTES
            )
        else:
            _raise_validation(f"serialized {name} array version is unsupported")
        header_end = stream.tell()
        if header_end > _MAX_SERIALIZED_NPY_HEADER_BYTES:
            _raise_validation(f"serialized {name} array header exceeds the safety cap")
        shape = tuple(shape)
        if len(shape) > _MAX_SERIALIZED_ARRAY_NDIM or any(
            type(item) is not int
            or item < 0
            or item > _MAX_SERIALIZED_ARRAY_ELEMENTS
            for item in shape
        ):
            _raise_validation(f"serialized {name} array shape is invalid")
        elements = 1
        for dimension in shape:
            elements *= dimension
            if elements > _MAX_SERIALIZED_ARRAY_ELEMENTS:
                _raise_validation(f"serialized {name} array elements exceed the safety cap")
        dtype = np.dtype(dtype)
        if dtype.hasobject or dtype.kind not in "biufc?":
            _raise_validation(f"serialized {name} array dtype is invalid")
        if elements * dtype.itemsize > _MAX_SERIALIZED_ARRAY_BYTES:
            _raise_validation(f"serialized {name} array bytes exceed the safety cap")
        if expected_dtype is not None and dtype != np.dtype(expected_dtype):
            _raise_validation(f"serialized {name} array dtype does not match trusted state")
        if expected_shape is not None and shape != tuple(expected_shape):
            _raise_validation(f"serialized {name} array shape does not match trusted state")
        array = np.load(
            io.BytesIO(raw),
            allow_pickle=False,
            max_header_size=_MAX_SERIALIZED_NPY_HEADER_BYTES,
        )
    except FeatureValidationError:
        raise
    except (ValueError, TypeError, OSError, EOFError, MemoryError) as error:
        raise FeatureValidationError(f"serialized {name} array is invalid") from error
    if not isinstance(array, np.ndarray):
        _raise_validation(f"serialized {name} array is invalid")
    result = np.asarray(array)
    if result.ndim > _MAX_SERIALIZED_ARRAY_NDIM:
        _raise_validation(f"serialized {name} array dimensions exceed the safety cap")
    if expected_dtype is not None and result.dtype != np.dtype(expected_dtype):
        _raise_validation(f"serialized {name} array dtype does not match trusted state")
    if expected_shape is not None and result.shape != tuple(expected_shape):
        _raise_validation(f"serialized {name} array shape does not match trusted state")
    return result


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_SERIALIZED_JSON_BYTES:
        _raise_validation("serialized payload exceeds the safety cap")
    return encoded


def _load_json_bytes(value: bytes, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        _raise_validation(f"serialized {name} payload is invalid")
    if len(value) > _MAX_SERIALIZED_JSON_BYTES:
        _raise_validation(f"serialized {name} payload exceeds the safety cap")

    try:
        payload = json.loads(
            bytes(value).decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except _DuplicateJSONKeyError as error:
        raise FeatureValidationError(f"serialized {name} payload has duplicate JSON keys") from error
    except (UnicodeDecodeError, json.JSONDecodeError, MemoryError) as error:
        raise FeatureValidationError(f"serialized {name} payload is invalid") from error
    if not isinstance(payload, Mapping):
        _raise_validation(f"serialized {name} payload is invalid")
    return payload


def _safe_embedding_error(error: Exception) -> FeatureValidationError:
    if isinstance(error, EmbeddingFeatureValidationError):
        return FeatureValidationError(str(error))
    return FeatureValidationError(str(error))


@dataclass(frozen=True)
class FeatureBundle:
    """All registered row-aligned blocks used by a CPB v3 fold."""

    metadata: np.ndarray
    surface_stats: np.ndarray
    scalar_internal: np.ndarray
    morphology: np.ndarray
    frozen_embedding: np.ndarray
    sample_ids: tuple[str, ...]
    domain_ids: tuple[str, ...]
    state_sha256: str = field(init=False)
    block_contract: Mapping[str, Mapping[str, object]] = field(
        init=False, repr=False, compare=False
    )
    _metadata_backing: bytes = field(init=False, repr=False, compare=False)
    _surface_backing: bytes = field(init=False, repr=False, compare=False)
    _scalar_backing: bytes = field(init=False, repr=False, compare=False)
    _morphology_backing: bytes = field(init=False, repr=False, compare=False)
    _frozen_backing: bytes = field(init=False, repr=False, compare=False)
    _source_authority: object = field(init=False, repr=False, compare=False)
    source_state_sha256: str = field(init=False, repr=False, compare=False)
    _target_values: np.ndarray = field(init=False, repr=False, compare=False)
    target_state_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        construction = _BUNDLE_CONSTRUCTION_CONTEXT
        source, production = (
            (_TEST_AUTHORITY, False) if construction is None else construction
        )
        if production and not _typed_v3_source(source):
            _raise_validation(
                "production feature bundles require a typed V3Data/V3DataView"
            )
        metadata = _as_matrix(
            self.metadata,
            "metadata",
            columns=METADATA_FEATURE_COUNT,
            allow_nan=True,
        )
        rows = metadata.shape[0]
        if rows == 0:
            _raise_validation("feature bundle cannot be empty")
        surface_stats = _as_matrix(
            self.surface_stats,
            "surface_stats",
            rows=rows,
            columns=SURFACE_FEATURE_COUNT,
            allow_nan=True,
        )
        scalar_internal = _as_matrix(
            self.scalar_internal,
            "scalar_internal",
            rows=rows,
            columns=SCALAR_FEATURE_COUNT,
            allow_nan=True,
        )
        morphology = _as_matrix(
            self.morphology,
            "morphology",
            rows=rows,
            columns=MORPHOLOGY_FEATURE_COUNT,
            allow_nan=True,
        )
        frozen_embedding = _as_matrix(
            self.frozen_embedding,
            "frozen_embedding",
            rows=rows,
            columns=FROZEN_EMBEDDING_FEATURE_COUNT,
            allow_nan=True,
        )
        if isinstance(self.sample_ids, (str, bytes)) or isinstance(
            self.domain_ids, (str, bytes)
        ):
            _raise_validation("sample_ids and domain_ids must be sequences of strings")
        sample_ids = tuple(self.sample_ids)
        domain_ids = tuple(self.domain_ids)
        if len(sample_ids) != rows or len(domain_ids) != rows:
            _raise_validation(
                "sample_ids and domain_ids must align with all feature rows"
            )
        if any(
            not isinstance(value, str) or not value for value in sample_ids + domain_ids
        ):
            _raise_validation("sample_ids and domain_ids must be non-empty strings")
        if len(set(sample_ids)) != rows:
            _raise_validation("sample_ids must be unique")
        metadata = _readonly_array(metadata)
        surface_stats = _readonly_array(surface_stats)
        scalar_internal = _readonly_array(scalar_internal)
        morphology = _readonly_array(morphology)
        frozen_embedding = _readonly_array(frozen_embedding)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "surface_stats", surface_stats)
        object.__setattr__(self, "scalar_internal", scalar_internal)
        object.__setattr__(self, "morphology", morphology)
        object.__setattr__(self, "frozen_embedding", frozen_embedding)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "domain_ids", domain_ids)
        object.__setattr__(self, "_source_authority", source)
        source_state = (
            _source_state(source, name="feature bundle source")
            if production
            else "test-only"
        )
        object.__setattr__(self, "source_state_sha256", source_state)
        if production:
            source_sample_ids = _strict_string_tuple(
                np.asarray(source.sample_ids).tolist(),
                name="feature bundle source sample IDs",
            )
            source_domain_ids = _strict_string_tuple(
                np.asarray(source.dataset_ids).tolist(),
                name="feature bundle source domain IDs",
            )
            if source_sample_ids != sample_ids or source_domain_ids != domain_ids:
                _raise_validation(
                    "feature bundle row identity does not match its V3 source"
                )
            target_values = _as_vector(
                source.cai_ratio,
                "feature bundle source targets",
            )
            if target_values.size != rows:
                _raise_validation(
                    "feature bundle source targets do not align with feature rows"
                )
            target_values = _readonly_array(target_values)
            target_state = _state_hash(
                target_values, sample_ids, domain_ids, source_state
            )
        else:
            target_values = _readonly_array(np.empty(0, dtype=np.float64))
            target_state = "test-only"
        object.__setattr__(self, "_target_values", target_values)
        object.__setattr__(self, "target_state_sha256", target_state)
        object.__setattr__(self, "_metadata_backing", metadata.tobytes(order="C"))
        object.__setattr__(self, "_surface_backing", surface_stats.tobytes(order="C"))
        object.__setattr__(self, "_scalar_backing", scalar_internal.tobytes(order="C"))
        object.__setattr__(self, "_morphology_backing", morphology.tobytes(order="C"))
        object.__setattr__(self, "_frozen_backing", frozen_embedding.tobytes(order="C"))
        object.__setattr__(
            self,
            "state_sha256",
            _state_hash(
                metadata,
                surface_stats,
                scalar_internal,
                morphology,
                frozen_embedding,
                sample_ids,
                domain_ids,
            ),
        )
        block_contract = {
            "metadata": {
                "feature_names": _METADATA_FEATURE_NAMES,
                "units": ("1",) * METADATA_FEATURE_COUNT,
                "role": "metadata",
                "sample_ids": sample_ids,
                "source_sha256": _state_hash(metadata),
            },
            "surface_stats": {
                "feature_names": _SURFACE_FEATURE_NAMES,
                "units": _SURFACE_FEATURE_UNITS,
                "role": "surface_stats",
                "sample_ids": sample_ids,
                "source_sha256": _state_hash(surface_stats),
            },
            "scalar_internal": {
                "feature_names": SCALAR_FEATURE_NAMES,
                "units": SCALAR_FEATURE_UNITS,
                "role": "scalar_internal",
                "sample_ids": sample_ids,
                "source_sha256": _state_hash(scalar_internal),
            },
            "morphology": {
                "feature_names": _MORPHOLOGY_NAMES,
                "units": _MORPHOLOGY_UNITS,
                "role": "morphology",
                "sample_ids": sample_ids,
                "source_sha256": _state_hash(morphology),
            },
            "frozen_embedding": {
                "feature_names": tuple(
                    f"frozen_embedding_{index}"
                    for index in range(FROZEN_EMBEDDING_FEATURE_COUNT)
                ),
                "units": ("1",) * FROZEN_EMBEDDING_FEATURE_COUNT,
                "role": "frozen_embedding",
                "sample_ids": sample_ids,
                "source_sha256": _state_hash(frozen_embedding),
            },
        }
        object.__setattr__(
            self,
            "block_contract",
            MappingProxyType(
                {
                    name: MappingProxyType(dict(contract))
                    for name, contract in block_contract.items()
                }
            ),
        )
        _register_state(self, self.state_sha256)
        if production:
            _register_bundle_authority(self, source, production=True)
        else:
            _register_test_bundle_authority(self)

    @property
    def n_rows(self) -> int:
        validate_feature_bundle(self)
        return int(self.metadata.shape[0])


def make_test_feature_bundle(**values: object) -> FeatureBundle:
    """Create an explicitly test-only synthetic bundle."""

    return _mark_test_bundle(FeatureBundle(**values))


def make_feature_bundle_from_v3_data(
    data: object,
    *,
    morphology: np.ndarray,
    frozen_embedding: np.ndarray,
) -> FeatureBundle:
    """Issue a production bundle from a validated V3Data or V3DataView source."""

    if not _typed_v3_source(data):
        _raise_validation("production feature bundles require a typed V3Data/V3DataView")
    source_state = _source_state(data, name="V3 data source")
    bundle = _construct_bundle(
        source=data,
        production=True,
        metadata=np.asarray(data.metadata13),
        surface_stats=np.asarray(data.profile_stats21),
        scalar_internal=np.asarray(data.scalar_internal3),
        morphology=morphology,
        frozen_embedding=frozen_embedding,
        sample_ids=tuple(np.asarray(data.sample_ids).tolist()),
        domain_ids=tuple(np.asarray(data.dataset_ids).tolist()),
    )
    if bundle.source_state_sha256 != source_state:
        _raise_validation("feature bundle source state changed during construction")
    return bundle


make_production_feature_bundle = make_feature_bundle_from_v3_data


def make_response_vector_from_v3_data(
    data: object,
    bundle: FeatureBundle,
) -> ResponseVector:
    """Issue a response only when the bundle is bound to this exact V3 source."""

    if not _typed_v3_source(data):
        _raise_validation("response source requires a typed V3Data/V3DataView")
    source, _state, production = _bundle_authority_record(bundle)
    if not production or source is not data:
        _raise_validation("response source authority does not match feature bundle")
    return make_response_vector(bundle)


def validate_feature_bundle(bundle: FeatureBundle) -> bool:
    """Validate immutable feature arrays and the complete bundle state digest."""

    if not isinstance(bundle, FeatureBundle):
        _raise_validation("feature bundle state has an invalid type")
    current = _state_hash(
        bundle.metadata,
        bundle.surface_stats,
        bundle.scalar_internal,
        bundle.morphology,
        bundle.frozen_embedding,
        bundle.sample_ids,
        bundle.domain_ids,
    )
    _assert_authority(bundle, current, name="feature bundle")
    source, source_state, production = _bundle_authority_record(bundle)
    if source is None:
        _raise_validation("feature bundle has no registered source authority")
    if production and bundle.source_state_sha256 != source_state:
        _raise_validation("feature bundle source state digest changed")
    if production:
        source_sample_ids = _strict_string_tuple(
            np.asarray(source.sample_ids).tolist(),
            name="feature bundle source sample IDs",
        )
        source_domain_ids = _strict_string_tuple(
            np.asarray(source.dataset_ids).tolist(),
            name="feature bundle source domain IDs",
        )
        if (source_sample_ids, source_domain_ids) != (
            bundle.sample_ids,
            bundle.domain_ids,
        ):
            _raise_validation("feature bundle row identity does not match source")
        source_targets = _as_vector(
            source.cai_ratio, "feature bundle source targets"
        )
        if not np.array_equal(source_targets, bundle._target_values):
            _raise_validation("feature bundle target authority changed")
        expected_target_state = _state_hash(
            bundle._target_values,
            bundle.sample_ids,
            bundle.domain_ids,
            source_state,
        )
        if bundle.target_state_sha256 != expected_target_state:
            _raise_validation("feature bundle target state hash mismatch")
    elif bundle.target_state_sha256 != "test-only":
        _raise_validation("test-only feature bundle target authority is invalid")
    expected_contract = {
        "metadata": (
            _METADATA_FEATURE_NAMES,
            ("1",) * METADATA_FEATURE_COUNT,
            "metadata",
        ),
        "surface_stats": (
            _SURFACE_FEATURE_NAMES,
            _SURFACE_FEATURE_UNITS,
            "surface_stats",
        ),
        "scalar_internal": (
            SCALAR_FEATURE_NAMES,
            SCALAR_FEATURE_UNITS,
            "scalar_internal",
        ),
        "morphology": (_MORPHOLOGY_NAMES, _MORPHOLOGY_UNITS, "morphology"),
        "frozen_embedding": (
            tuple(
                f"frozen_embedding_{index}"
                for index in range(FROZEN_EMBEDDING_FEATURE_COUNT)
            ),
            ("1",) * FROZEN_EMBEDDING_FEATURE_COUNT,
            "frozen_embedding",
        ),
    }
    if set(bundle.block_contract) != set(expected_contract):
        _raise_validation("feature bundle block contract changed")
    for name, (names, units, role) in expected_contract.items():
        item = bundle.block_contract[name]
        expected_source = {
            "metadata": bundle.metadata,
            "surface_stats": bundle.surface_stats,
            "scalar_internal": bundle.scalar_internal,
            "morphology": bundle.morphology,
            "frozen_embedding": bundle.frozen_embedding,
        }[name]
        if (
            tuple(item.get("feature_names", ())) != names
            or tuple(item.get("units", ())) != units
            or item.get("role") != role
            or tuple(item.get("sample_ids", ())) != bundle.sample_ids
            or item.get("source_sha256") != _state_hash(expected_source)
        ):
            _raise_validation("feature bundle block contract changed")
    arrays = (
        (bundle.metadata, bundle._metadata_backing),
        (bundle.surface_stats, bundle._surface_backing),
        (bundle.scalar_internal, bundle._scalar_backing),
        (bundle.morphology, bundle._morphology_backing),
        (bundle.frozen_embedding, bundle._frozen_backing),
    )
    if any(array.flags.writeable for array, _ in arrays):
        _raise_validation("feature bundle state arrays must be read-only")
    if any(array.tobytes(order="C") != backing for array, backing in arrays):
        _raise_validation("feature bundle state backing bytes changed")
    expected = current
    if expected != bundle.state_sha256:
        _raise_validation("feature bundle state hash mismatch")
    return True


@dataclass(frozen=True)
class FeatureMatrix:
    """A typed feature block with its registered names, units, and block labels."""

    name: str
    matrix: np.ndarray
    feature_names: tuple[str, ...]
    units: tuple[str, ...]
    blocks: tuple[str, ...]
    sample_ids: tuple[str, ...] = ()
    domain_ids: tuple[str, ...] = ()
    source_sha256: str = ""
    authority: str = "test"
    state_sha256: str = field(init=False)
    source_state_sha256: str = field(init=False, repr=False, compare=False)
    _backing_bytes: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        matrix = _as_matrix(self.matrix, f"{self.name}.matrix", allow_nan=True)
        names = tuple(str(value) for value in self.feature_names)
        units = tuple(str(value) for value in self.units)
        if matrix.shape[1] != len(names) or len(names) != len(units):
            _raise_validation(f"{self.name} feature names and units do not align")
        if len(set(names)) != len(names):
            _raise_validation(f"{self.name} feature names are not unique")
        blocks = tuple(str(value) for value in self.blocks)
        if not blocks:
            _raise_validation(f"{self.name} has no registered blocks")
        sample_ids = _strict_string_tuple(
            self.sample_ids, name=f"{self.name} sample IDs"
        )
        if len(sample_ids) != matrix.shape[0]:
            _raise_validation(f"{self.name} sample IDs must align with rows")
        domain_ids = _strict_string_tuple(
            self.domain_ids, name=f"{self.name} domain IDs"
        )
        if len(domain_ids) != matrix.shape[0]:
            _raise_validation(f"{self.name} domain IDs must align with rows")
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.source_sha256)
        ):
            _raise_validation(f"{self.name} source digest is invalid")
        if self.authority not in {"test", "bundle"}:
            _raise_validation(f"{self.name} construction authority is invalid")
        context = _MATRIX_CONSTRUCTION_CONTEXT
        source_state = (
            context.source_state_sha256 if context is not None else "test-only"
        )
        if context is not None and self.authority == "bundle":
            if self.source_sha256 != context.state_sha256:
                _raise_validation(
                    f"{self.name} matrix source digest does not match its bundle"
                )
            if sample_ids != context.sample_ids or domain_ids != context.domain_ids:
                _raise_validation(
                    f"{self.name} matrix row identity does not match its bundle"
                )
        _validate_registered_matrix_contract(self.name, matrix, names, units, blocks)
        immutable_matrix = _readonly_array(matrix)
        object.__setattr__(self, "matrix", immutable_matrix)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "domain_ids", domain_ids)
        object.__setattr__(self, "source_state_sha256", source_state)
        object.__setattr__(self, "_backing_bytes", immutable_matrix.tobytes(order="C"))
        object.__setattr__(
            self,
            "state_sha256",
            _state_hash(
                self.name,
                immutable_matrix,
                names,
                units,
                blocks,
                sample_ids,
                domain_ids,
                self.source_sha256,
                self.authority,
                source_state,
            ),
        )
        _register_state(self, self.state_sha256)

    @property
    def values(self) -> np.ndarray:
        validate_feature_matrix(self)
        return self.matrix

    @property
    def X(self) -> np.ndarray:
        validate_feature_matrix(self)
        return self.matrix

    @property
    def n_features(self) -> int:
        validate_feature_matrix(self)
        return int(self.matrix.shape[1])


MatrixContract = FeatureMatrix


def _validate_registered_matrix_contract(
    name: str,
    matrix: np.ndarray,
    feature_names: Sequence[str],
    units: Sequence[str],
    blocks: Sequence[str],
) -> None:
    names = tuple(feature_names)
    unit_values = tuple(units)
    block_values = tuple(blocks)
    fixed = {
        "A_surface": (
            _METADATA_FEATURE_NAMES + _SURFACE_FEATURE_NAMES,
            ("1",) * A_FEATURE_COUNT,
            ("metadata", "surface_stats"),
        ),
        "B_scalar": (
            _METADATA_FEATURE_NAMES + _SURFACE_FEATURE_NAMES + SCALAR_FEATURE_NAMES,
            ("1",) * A_FEATURE_COUNT + SCALAR_FEATURE_UNITS,
            ("A_surface", "scalar_internal"),
        ),
        "B_morph": (
            _METADATA_FEATURE_NAMES + _SURFACE_FEATURE_NAMES + _MORPHOLOGY_NAMES,
            ("1",) * A_FEATURE_COUNT + _MORPHOLOGY_UNITS,
            ("A_surface", "morphology"),
        ),
        "I_morph": (
            _METADATA_FEATURE_NAMES + _MORPHOLOGY_NAMES,
            ("1",) * METADATA_FEATURE_COUNT + _MORPHOLOGY_UNITS,
            ("metadata", "morphology"),
        ),
    }
    expected = fixed.get(name)
    if expected is not None:
        if (names, unit_values, block_values) != expected:
            _raise_validation(f"{name} does not use the registered canonical columns")
        return
    if name not in {"B_frozen", "B_combined", "I_frozen", "I_combined"}:
        return
    if name in {"B_frozen", "B_combined"}:
        prefix_names = _METADATA_FEATURE_NAMES + _SURFACE_FEATURE_NAMES
        prefix_units = ("1",) * A_FEATURE_COUNT
        prefix_blocks = ("A_surface",)
        if name == "B_combined":
            prefix_names += _MORPHOLOGY_NAMES
            prefix_units += _MORPHOLOGY_UNITS
            prefix_blocks += ("morphology",)
    else:
        prefix_names = _METADATA_FEATURE_NAMES
        prefix_units = ("1",) * METADATA_FEATURE_COUNT
        prefix_blocks = ("metadata",)
        if name == "I_combined":
            prefix_names += _MORPHOLOGY_NAMES
            prefix_units += _MORPHOLOGY_UNITS
            prefix_blocks += ("morphology",)
    dimension = matrix.shape[1] - len(prefix_names)
    if dimension not in PCA_DIMENSIONS:
        _raise_validation(f"{name} does not have a registered PCA dimension")
    expected = (
        prefix_names + _FROZEN_NAMES[:dimension],
        prefix_units + ("1",) * dimension,
        prefix_blocks + ("frozen_embedding",),
    )
    if (names, unit_values, block_values) != expected:
        _raise_validation(f"{name} does not use the registered canonical columns")


def validate_feature_matrix(matrix: FeatureMatrix) -> bool:
    """Reject mutable or byte/state-tampered feature matrices."""

    if not isinstance(matrix, FeatureMatrix):
        _raise_validation("feature matrix state has an invalid type")
    current = _state_hash(
        matrix.name,
        matrix.matrix,
        matrix.feature_names,
        matrix.units,
        matrix.blocks,
    matrix.sample_ids,
        matrix.domain_ids,
        matrix.source_sha256,
        matrix.authority,
        matrix.source_state_sha256,
    )
    _assert_authority(matrix, current, name="feature matrix")
    matrix_record = _MATRIX_AUTHORITY.get(id(matrix))
    if matrix_record is None or matrix_record[0]() is not matrix:
        _raise_validation("feature matrix has no registered construction authority")
    source_bundle = matrix_record[1]()
    if source_bundle is None:
        _raise_validation("feature matrix source bundle authority is unavailable")
    validate_feature_bundle(source_bundle)
    if matrix.source_sha256 != source_bundle.state_sha256:
        _raise_validation("feature matrix source digest does not match source bundle")
    if matrix.source_state_sha256 != source_bundle.source_state_sha256:
        _raise_validation("feature matrix source authority state does not match bundle")
    if (
        matrix.sample_ids != source_bundle.sample_ids
        or matrix.domain_ids != source_bundle.domain_ids
    ):
        _raise_validation("feature matrix row identity does not match source bundle")
    if matrix_record[2] != source_bundle.state_sha256:
        _raise_validation("feature matrix source bundle state changed")
    if matrix_record[3] != source_bundle.source_state_sha256:
        _raise_validation("feature matrix source authority state changed")
    if matrix.authority != "bundle":
        _raise_validation("feature matrix has no production bundle authority")
    _validate_registered_matrix_contract(
        matrix.name,
        matrix.matrix,
        matrix.feature_names,
        matrix.units,
        matrix.blocks,
    )
    if matrix.matrix.flags.writeable:
        _raise_validation("feature matrix state array must be read-only")
    if matrix.matrix.tobytes(order="C") != matrix._backing_bytes:
        _raise_validation("feature matrix backing bytes were tampered")
    expected = current
    if expected != matrix.state_sha256:
        _raise_validation("feature matrix state hash mismatch")
    return True


def serialize_feature_matrix(matrix: FeatureMatrix) -> bytes:
    validate_feature_matrix(matrix)
    return _json_bytes(
        {
            "version": 1,
            "name": matrix.name,
            "matrix": _encode_array(matrix.matrix),
            "feature_names": list(matrix.feature_names),
            "units": list(matrix.units),
            "blocks": list(matrix.blocks),
            "sample_ids": list(matrix.sample_ids),
            "domain_ids": list(matrix.domain_ids),
            "source_sha256": matrix.source_sha256,
            "source_state_sha256": matrix.source_state_sha256,
            "authority": matrix.authority,
            "state_sha256": matrix.state_sha256,
        }
    )


def deserialize_feature_matrix(
    value: bytes, *, source_bundle: FeatureBundle | None = None
) -> FeatureMatrix:
    try:
        payload = _load_json_bytes(value, name="feature matrix")
        _require_json_schema(
            payload,
            name="feature matrix",
            schema={
                "version": int,
                "name": str,
                "matrix": str,
                "feature_names": list,
                "units": list,
                "blocks": list,
                "sample_ids": list,
                "domain_ids": list,
                "source_sha256": str,
                "source_state_sha256": str,
                "authority": str,
                "state_sha256": str,
            },
        )
        feature_names = _require_string_list(
            payload["feature_names"], name="feature matrix feature_names"
        )
        units = _require_string_list(payload["units"], name="feature matrix units")
        blocks = _require_string_list(payload["blocks"], name="feature matrix blocks")
        serialized_sample_ids = _require_string_list(
            payload["sample_ids"], name="feature matrix sample_ids"
        )
        serialized_domain_ids = _require_string_list(
            payload["domain_ids"], name="feature matrix domain_ids"
        )
        if payload.get("version") != 1:
            _raise_validation("serialized feature matrix version is unsupported")
        if source_bundle is None:
            source_bundle = _resolve_bundle_state(payload.get("source_sha256"))
        elif not isinstance(source_bundle, FeatureBundle):
            _raise_validation("serialized feature matrix source bundle is invalid")
        validate_feature_bundle(source_bundle)
        if payload.get("source_sha256") != source_bundle.state_sha256:
            _raise_validation("serialized feature matrix source digest does not match bundle")
        if payload.get("authority") != "bundle":
            _raise_validation("serialized feature matrix authority is invalid")
        if payload.get("source_state_sha256") != source_bundle.source_state_sha256:
            _raise_validation(
                "serialized feature matrix source authority state does not match bundle"
            )
        trusted_matrix = _resolve_matrix_state(payload.get("state_sha256"))
        if trusted_matrix is None:
            try:
                trusted_matrix = assemble_feature_matrices(source_bundle).get(
                    payload.get("name")
                )
            except Exception as error:
                raise FeatureValidationError(
                    "serialized feature matrix has no trusted creation authority"
                ) from error
        if not isinstance(trusted_matrix, FeatureMatrix):
            _raise_validation(
                "serialized feature matrix has no trusted creation authority"
            )
        validate_feature_matrix(trusted_matrix)
        trusted_record = _MATRIX_AUTHORITY.get(id(trusted_matrix))
        if trusted_record is None or trusted_record[1]() is not source_bundle:
            _raise_validation(
                "serialized feature matrix source authority does not match bundle"
            )
        sample_ids = tuple(serialized_sample_ids)
        domain_ids = tuple(serialized_domain_ids)
        if sample_ids != source_bundle.sample_ids or domain_ids != source_bundle.domain_ids:
            _raise_validation("serialized feature matrix row identity does not match bundle")
        decoded_matrix = _decode_array(
            payload["matrix"],
            name="feature matrix",
            expected_dtype=trusted_matrix.matrix.dtype,
            expected_shape=trusted_matrix.matrix.shape,
        )
        if (
            payload["name"] != trusted_matrix.name
            or tuple(feature_names) != trusted_matrix.feature_names
            or tuple(units) != trusted_matrix.units
            or tuple(blocks) != trusted_matrix.blocks
            or sample_ids != trusted_matrix.sample_ids
            or domain_ids != trusted_matrix.domain_ids
            or not np.array_equal(
                decoded_matrix, trusted_matrix.matrix, equal_nan=True
            )
        ):
            _raise_validation(
                "serialized feature matrix payload differs from trusted creation state"
            )
        global _MATRIX_CONSTRUCTION_CONTEXT
        previous = _MATRIX_CONSTRUCTION_CONTEXT
        _MATRIX_CONSTRUCTION_CONTEXT = source_bundle
        try:
            matrix = FeatureMatrix(
                name=payload["name"],  # type: ignore[arg-type]
                matrix=decoded_matrix,
                feature_names=tuple(feature_names),
                units=tuple(units),
                blocks=tuple(blocks),
                sample_ids=sample_ids,
                domain_ids=domain_ids,
                source_sha256=source_bundle.state_sha256,
                authority="bundle",
            )
        finally:
            _MATRIX_CONSTRUCTION_CONTEXT = previous
        _register_matrix_authority(matrix, source_bundle)
        if matrix.state_sha256 != payload.get("state_sha256"):
            _raise_validation("serialized feature matrix state hash mismatch")
        validate_feature_matrix(matrix)
        return matrix
    except FeatureValidationError:
        raise
    except Exception as error:
        raise FeatureValidationError("serialized feature matrix payload is invalid") from error


def _block(
    name: str,
    matrix: np.ndarray,
    feature_names: Sequence[str],
    units: Sequence[str],
    blocks: Sequence[str],
    *,
    sample_ids: Sequence[str],
    domain_ids: Sequence[str],
    source_sha256: str,
    authority: str = "bundle",
) -> FeatureMatrix:
    global _MATRIX_CONSTRUCTION_CONTEXT
    bundle = _MATRIX_CONSTRUCTION_CONTEXT
    if bundle is None:
        _raise_validation("feature matrix requires a trusted bundle factory")
    previous = _MATRIX_CONSTRUCTION_CONTEXT
    try:
        result = FeatureMatrix(
        name=name,
        matrix=matrix,
        feature_names=tuple(feature_names),
        units=tuple(units),
        blocks=tuple(blocks),
        sample_ids=tuple(sample_ids),
        domain_ids=tuple(domain_ids),
        source_sha256=source_sha256,
        authority=authority,
        )
    finally:
        _MATRIX_CONSTRUCTION_CONTEXT = previous
    _register_matrix_authority(result, bundle)
    return result


def _register_matrix_authority(matrix: FeatureMatrix, bundle: FeatureBundle) -> None:
    matrix_identity = id(matrix)
    _MATRIX_AUTHORITY[id(matrix)] = (
        ref(matrix, lambda _dead: _MATRIX_AUTHORITY.pop(matrix_identity, None)),
        ref(bundle, lambda _dead: _MATRIX_AUTHORITY.pop(matrix_identity, None)),
        bundle.state_sha256,
        bundle.source_state_sha256,
    )
    _remember_matrix_state(matrix)


def _pca_names(dimension: int) -> tuple[str, ...]:
    if dimension not in PCA_DIMENSIONS:
        _raise_validation("PCA dimension must be one of 8, 16, or 32")
    return _FROZEN_NAMES[:dimension]


def _validate_pca_dimension(dimension: int) -> int:
    if not isinstance(dimension, (int, np.integer)) or isinstance(
        dimension, (bool, np.bool_)
    ):
        _raise_validation("PCA dimension must be an integer")
    dimension = int(dimension)
    if dimension not in PCA_DIMENSIONS:
        _raise_validation("PCA dimension must be one of 8, 16, or 32")
    return dimension


def _pca_state_digest(
    model: FoldLocalPCA | _TestOnlyPCA,
    fit_sample_ids: tuple[str, ...],
    fit_domain_ids: tuple[str, ...],
) -> str:
    return _state_hash(
        model.mean_,
        model.components_,
        model.explained_variance_,
        model.singular_values_,
        model.n_components_,
        model.n_features_in_,
        model.n_samples_in_,
        model.fit_sample_ids,
        model.fit_domain_ids,
        fit_sample_ids,
        fit_domain_ids,
    )


def _training_embedding_digest(
    bundle: FeatureBundle,
    fit_sample_ids: Sequence[str],
    fit_domain_ids: Sequence[str],
) -> str:
    """Digest the exact embedding rows used by a registered PCA fit."""

    fit_samples = tuple(fit_sample_ids)
    sample_ids = _ids(
        fit_samples,
        length=len(fit_samples),
        name="PCA fit_sample_ids",
    )
    domain_ids = _ids(
        fit_domain_ids,
        length=len(sample_ids),
        name="PCA fit_domain_ids",
    )
    positions = {sample_id: index for index, sample_id in enumerate(bundle.sample_ids)}
    try:
        indices = np.asarray([positions[sample_id] for sample_id in sample_ids])
    except KeyError as error:
        _raise_validation("PCA fit sample IDs are not registered in the bundle")
        raise AssertionError from error
    expected_domains = tuple(bundle.domain_ids[index] for index in indices)
    if expected_domains != domain_ids:
        _raise_validation("PCA fit domains do not match the bundle embedding rows")
    return _state_hash(
        bundle.frozen_embedding[indices],
        sample_ids,
        domain_ids,
        bundle.source_state_sha256,
    )


def _raw_training_embedding_digest(
    bundle: FeatureBundle,
    fit_sample_ids: Sequence[str],
) -> str:
    positions = {sample_id: index for index, sample_id in enumerate(bundle.sample_ids)}
    try:
        indices = np.asarray([positions[sample_id] for sample_id in fit_sample_ids])
    except KeyError as error:
        raise FeatureValidationError(
            "PCA fit sample IDs are not registered in the bundle"
        ) from error
    return _embedding_state_hash(bundle.frozen_embedding[indices])


def _fit_test_only_pca(
    values: np.ndarray,
    *,
    n_components: int,
    fit_sample_ids: Sequence[str],
    fit_domain_ids: Sequence[str],
) -> _TestOnlyPCA:
    value = _as_matrix(
        values,
        "test-only PCA training embeddings",
        columns=FROZEN_EMBEDDING_FEATURE_COUNT,
        allow_nan=False,
    )
    if value.shape[0] <= n_components:
        raise _PcaRankUnavailable(n_components)
    mean = np.mean(value, axis=0, dtype=np.float64)
    centered = value - mean
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            _u, singular_values, components = np.linalg.svd(
                centered, full_matrices=False
            )
            if not np.all(np.isfinite(singular_values)) or not np.all(
                np.isfinite(components)
            ):
                _raise_validation("test-only PCA SVD values must be finite")
    except _PcaRankUnavailable:
        raise
    except (FloatingPointError, np.linalg.LinAlgError, RuntimeError, ValueError) as error:
        raise FeatureValidationError("test-only PCA fitting failed safely") from error
    try:
        valid_dimensions = valid_pca_dimensions(value, PCA_DIMENSIONS)
    except EmbeddingFeatureValidationError as error:
        raise FeatureValidationError(str(error)) from error
    if n_components not in valid_dimensions:
        raise _PcaRankUnavailable(n_components)
    components = np.asarray(components[:n_components], dtype=np.float64).copy()
    for row in range(components.shape[0]):
        pivot = int(np.argmax(np.abs(components[row])))
        if components[row, pivot] < 0.0:
            components[row] *= -1.0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            explained_variance = (singular_values[:n_components] ** 2) / max(
                value.shape[0] - 1, 1
            )
    except (FloatingPointError, RuntimeError, ValueError) as error:
        raise FeatureValidationError(
            "test-only PCA explained variance is not finite"
        ) from error
    if not np.all(np.isfinite(explained_variance)):
        raise FeatureValidationError("test-only PCA explained variance is not finite")
    return _TestOnlyPCA(
        mean_=mean,
        components_=components,
        explained_variance_=explained_variance,
        singular_values_=singular_values[:n_components],
        n_components_=n_components,
        n_features_in_=value.shape[1],
        fit_sample_ids=tuple(fit_sample_ids),
        fit_domain_ids=tuple(fit_domain_ids),
        n_samples_in_=value.shape[0],
    )


def _register_pca_model(
    model: FoldLocalPCA | _TestOnlyPCA,
    fit_sample_ids: Sequence[str],
    fit_domain_ids: Sequence[str],
    bundle: FeatureBundle | None = None,
    *,
    training_authority: object | None = None,
    parent_authority: object | None = None,
    training_embedding_digest: str | None = None,
) -> None:
    sample_ids = _ids(fit_sample_ids, length=model.n_samples_in_, name="fit_sample_ids")
    domain_ids = _ids(fit_domain_ids, length=model.n_samples_in_, name="fit_domain_ids")
    identity = id(model)

    def remove(_dead: ReferenceType[object]) -> None:
        if _PCA_AUTHORITY.get(identity, (None,))[0] is _dead:
            _PCA_AUTHORITY.pop(identity, None)

    _PCA_AUTHORITY[id(model)] = (
        ref(model, remove),
        _pca_state_digest(model, sample_ids, domain_ids),
        sample_ids,
        domain_ids,
    )
    if bundle is not None:
        validate_feature_bundle(bundle)
        embedding_digest = _training_embedding_digest(
            bundle, sample_ids, domain_ids
        )
        if isinstance(model, FoldLocalPCA):
            expected_raw_digest = _raw_training_embedding_digest(bundle, sample_ids)
            if model.fit_embeddings_sha256 != expected_raw_digest:
                _raise_validation(
                    "PCA model training embedding digest does not match bundle"
                )
        if (
            training_embedding_digest is not None
            and training_embedding_digest != embedding_digest
        ):
            _raise_validation("PCA training embedding digest does not match bundle")
        model_identity = id(model)
        try:
            outer_reference = (
                None
                if training_authority is None
                else ref(
                    training_authority,
                    lambda _dead: _PCA_BUNDLE_AUTHORITY.pop(model_identity, None),
                )
            )
            parent_reference = (
                None
                if parent_authority is None
                else ref(
                    parent_authority,
                    lambda _dead: _PCA_BUNDLE_AUTHORITY.pop(model_identity, None),
                )
            )
        except TypeError as error:
            raise FeatureValidationError(
                "PCA authority objects must support weak identity binding"
            ) from error
        outer_state = (
            ""
            if training_authority is None
            else _source_state(training_authority, name="PCA training authority")
        )
        parent_state = (
            ""
            if parent_authority is None
            else _source_state(parent_authority, name="PCA parent authority")
        )
        _PCA_BUNDLE_AUTHORITY[model_identity] = (
            ref(model, lambda _dead: _PCA_BUNDLE_AUTHORITY.pop(model_identity, None)),
            ref(bundle, lambda _dead: _PCA_BUNDLE_AUTHORITY.pop(model_identity, None)),
            bundle.source_state_sha256,
            embedding_digest,
            outer_reference,
            parent_reference,
            outer_state,
            parent_state,
        )


def _validate_pca_bundle_authority(
    model: FoldLocalPCA | _TestOnlyPCA,
    bundle: FeatureBundle,
    *,
    training_authority: object | None = None,
    parent_authority: object | None = None,
    training_embedding_digest: str | None = None,
) -> None:
    record = _PCA_BUNDLE_AUTHORITY.get(id(model))
    if record is None or record[0]() is not model or record[1]() is not bundle:
        _raise_validation("PCA model has no matching feature bundle authority")
    validate_feature_bundle(bundle)
    if record[2] != bundle.source_state_sha256:
        _raise_validation("PCA feature bundle source authority state changed")
    authority = _PCA_AUTHORITY.get(id(model))
    if authority is None or authority[0]() is not model:
        _raise_validation("PCA model has no registered construction authority")
    expected_embedding_digest = _training_embedding_digest(
        bundle, authority[2], authority[3]
    )
    if record[3] != expected_embedding_digest:
        _raise_validation("PCA training embedding authority changed")
    if isinstance(model, FoldLocalPCA):
        expected_raw_digest = _raw_training_embedding_digest(bundle, authority[2])
        if model.fit_embeddings_sha256 != expected_raw_digest:
            _raise_validation("PCA model training embedding authority changed")
    if (
        training_embedding_digest is not None
        and training_embedding_digest != expected_embedding_digest
    ):
        _raise_validation("PCA training embedding digest does not match bundle")
    for supplied, expected_ref, expected_state, name in (
        (training_authority, record[4], record[6], "PCA training authority"),
        (parent_authority, record[5], record[7], "PCA parent authority"),
    ):
        expected = None if expected_ref is None else expected_ref()
        if supplied is not None and expected is not supplied:
            _raise_validation(f"{name} identity does not match PCA authority")
        if expected is not None and _source_state(expected, name=name) != expected_state:
            _raise_validation(f"{name} state changed")


def _validate_pca_model(
    model: FoldLocalPCA | _TestOnlyPCA,
    dimension: int | None = None,
    *,
    fit_sample_ids: Sequence[str] | None = None,
    fit_domain_ids: Sequence[str] | None = None,
    bundle: FeatureBundle | None = None,
    training_authority: object | None = None,
    parent_authority: object | None = None,
    training_embedding_digest: str | None = None,
) -> FoldLocalPCA | _TestOnlyPCA:
    if isinstance(model, _TestOnlyPCA):
        if model.n_features_in_ != FROZEN_EMBEDDING_FEATURE_COUNT:
            _raise_validation("PCA model is not fitted to 512-D frozen embeddings")
        if dimension is not None and model.n_components_ != dimension:
            _raise_validation("PCA model dimension does not match the requested dimension")
        authority = _PCA_AUTHORITY.get(id(model))
        if authority is None or authority[0]() is not model:
            _raise_validation("PCA model has no registered construction authority")
        if fit_sample_ids is not None or fit_domain_ids is not None:
            if fit_sample_ids is None or fit_domain_ids is None:
                _raise_validation("PCA fit IDs and domains are both required")
            sample_ids = _ids(
                fit_sample_ids, length=model.n_samples_in_, name="fit_sample_ids"
            )
            domain_ids = _ids(
                fit_domain_ids, length=model.n_samples_in_, name="fit_domain_ids"
            )
            if (sample_ids, domain_ids) != authority[2:]:
                _raise_validation("PCA fit identity does not match its authority")
        if authority[1] != _pca_state_digest(model, authority[2], authority[3]):
            _raise_validation("PCA construction authority does not match current state")
        if bundle is not None:
            _validate_pca_bundle_authority(
                model,
                bundle,
                training_authority=training_authority,
                parent_authority=parent_authority,
                training_embedding_digest=training_embedding_digest,
            )
        return model
    if not isinstance(model, FoldLocalPCA):
        _raise_validation("PCA model has an invalid type")
    if model.n_features_in_ != FROZEN_EMBEDDING_FEATURE_COUNT:
        _raise_validation("PCA model is not fitted to 512-D frozen embeddings")
    if dimension is not None and model.n_components_ != dimension:
        _raise_validation("PCA model dimension does not match the requested dimension")
    _validate_pca_dimension(model.n_components_)
    try:
        validate_fold_local_pca(model)
    except Exception as error:
        raise _safe_embedding_error(error) from error
    authority = _PCA_AUTHORITY.get(id(model))
    if authority is None or authority[0]() is not model:
        _raise_validation("PCA model has no registered construction authority")
    if fit_sample_ids is not None or fit_domain_ids is not None:
        if fit_sample_ids is None or fit_domain_ids is None:
            _raise_validation("PCA fit IDs and domains are both required")
        sample_ids = _ids(
            fit_sample_ids, length=model.n_samples_in_, name="fit_sample_ids"
        )
        domain_ids = _ids(
            fit_domain_ids, length=model.n_samples_in_, name="fit_domain_ids"
        )
        if (sample_ids, domain_ids) != authority[2:]:
            _raise_validation("PCA fit identity does not match its authority")
    if authority[1] != _pca_state_digest(model, authority[2], authority[3]):
        _raise_validation("PCA construction authority does not match current state")
    if bundle is not None:
        _validate_pca_bundle_authority(
            model,
            bundle,
            training_authority=training_authority,
            parent_authority=parent_authority,
            training_embedding_digest=training_embedding_digest,
        )
    return model


def _transform_pca_model(
    model: FoldLocalPCA | _TestOnlyPCA, values: np.ndarray
) -> np.ndarray:
    if isinstance(model, _TestOnlyPCA):
        value = _as_matrix(
            values,
            "test-only PCA query embeddings",
            columns=FROZEN_EMBEDDING_FEATURE_COUNT,
            allow_nan=False,
        )
        output = (value - model.mean_) @ model.components_.T
        if not np.all(np.isfinite(output)):
            _raise_validation("test-only PCA output must be finite")
        return output
    try:
        return transform_embedding_pca(model, values)
    except Exception as error:
        raise _safe_embedding_error(error) from error


def _validate_issued_pca_view(
    authority: object,
    fit_sample_ids: Sequence[str],
    fit_domain_ids: Sequence[str],
    *,
    name: str,
    expected_domain_count: int,
) -> str:
    if not _typed_v3_source(authority):
        _raise_validation(f"{name} must be an issued V3DataView authority")
    try:
        from .data import V3DataView

        if type(authority) is not V3DataView:
            _raise_validation(f"{name} must be an issued V3DataView authority")
        authority_state = _source_state(authority, name=name)
        authority_ids = tuple(np.asarray(authority.sample_ids).tolist())
        authority_domains = tuple(np.asarray(authority.dataset_ids).tolist())
    except FeatureValidationError:
        raise
    except Exception as error:
        raise FeatureValidationError(f"{name} is malformed") from error
    expected_ids = tuple(fit_sample_ids)
    expected_domains = tuple(fit_domain_ids)
    if (authority_ids, authority_domains) != (expected_ids, expected_domains):
        _raise_validation(f"{name} identity does not match PCA training rows")
    if len(set(authority_domains)) != expected_domain_count:
        _raise_validation(
            f"{name} must contain exactly {expected_domain_count} training domains"
        )
    return authority_state


def _resolve_outer_pca_authority(
    bundle: FeatureBundle,
    fit_indices: np.ndarray,
    fit_sample_ids: Sequence[str],
    fit_domain_ids: Sequence[str],
    *,
    outer_train_authority: object | None,
    heldout_domain: str | None,
) -> tuple[object, str, str]:
    source, _source_state_value, production = _bundle_authority_record(bundle)
    if not production or source is None:
        _raise_validation("production PCA requires a V3 source authority")
    selected_domains = tuple(fit_domain_ids)
    resolved_heldout = heldout_domain
    if resolved_heldout is None:
        _raise_validation("production PCA requires one explicit heldout outer domain")
    if type(resolved_heldout) is not str or not resolved_heldout:
        _raise_validation("PCA heldout domain must be a non-empty string")
    authority = outer_train_authority
    if authority is None:
        try:
            authority = source.subset(fit_indices)
        except Exception as error:
            raise FeatureValidationError(
                "production PCA could not issue the outer-training V3DataView"
            ) from error
    authority_state = _validate_issued_pca_view(
        authority,
        fit_sample_ids,
        fit_domain_ids,
        name="outer-training PCA authority",
        expected_domain_count=5,
    )
    _validate_pca_view_matches_bundle(
        bundle,
        authority,
        name="outer-training PCA authority",
    )
    if resolved_heldout in set(selected_domains):
        _raise_validation("PCA heldout domain occurs in outer training")
    return authority, resolved_heldout, authority_state


def _validate_pca_view_matches_bundle(
    bundle: FeatureBundle,
    authority: object,
    *,
    name: str,
) -> None:
    """Ensure an issued view carries the exact values from this bundle source."""

    source, source_state, production = _bundle_authority_record(bundle)
    if not production or source is None:
        _raise_validation("production PCA requires a V3 source authority")
    source_ids = tuple(np.asarray(source.sample_ids).tolist())
    authority_ids = tuple(np.asarray(authority.sample_ids).tolist())
    positions = {sample_id: index for index, sample_id in enumerate(source_ids)}
    try:
        source_indices = np.asarray([positions[item] for item in authority_ids])
    except KeyError as error:
        raise FeatureValidationError(
            f"{name} contains a sample outside the feature bundle source"
        ) from error
    source_domains = tuple(np.asarray(source.dataset_ids)[source_indices].tolist())
    authority_domains = tuple(np.asarray(authority.dataset_ids).tolist())
    if source_domains != authority_domains:
        _raise_validation(f"{name} domains do not match the feature bundle source")
    for field_name in (
        "metadata13",
        "profile_stats21",
        "scalar_internal3",
        "cai_ratio",
    ):
        source_values = np.asarray(getattr(source, field_name))[source_indices]
        authority_values = np.asarray(getattr(authority, field_name))
        if not np.array_equal(source_values, authority_values):
            _raise_validation(f"{name} values do not match the feature bundle source")
    if _source_state(source, name="feature bundle source") != source_state:
        _raise_validation("feature bundle source authority state changed")


def _validate_outer_split_authorities(
    bundle: FeatureBundle,
    selection: CandidateSelection,
    outer_train_authority: object,
    outer_test_authority: object,
) -> tuple[np.ndarray, np.ndarray, str, str, str, str]:
    """Validate the issued five-domain train and one-domain test views."""

    try:
        from .data import V3DataView
    except (ImportError, ModuleNotFoundError) as error:
        raise FeatureValidationError("V3DataView authority is unavailable") from error
    if type(outer_train_authority) is not V3DataView or not _typed_v3_source(
        outer_train_authority
    ):
        _raise_validation(
            "outer_train_authority must be an issued five-domain V3DataView"
        )
    if type(outer_test_authority) is not V3DataView or not _typed_v3_source(
        outer_test_authority
    ):
        _raise_validation(
            "outer_test_authority must be an issued held-out V3DataView"
        )
    train_indices = np.asarray(selection.outer_train_indices, dtype=np.int64)
    train_indices = np.sort(
        _indices(train_indices, length=bundle.n_rows, name="selection outer train")
    )
    sample_to_index = {sample_id: index for index, sample_id in enumerate(bundle.sample_ids)}
    try:
        test_indices = np.asarray(
            [sample_to_index[sample_id] for sample_id in selection.outer_test_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise FeatureValidationError("selection outer test ID is not in the bundle") from error
    canonical_test_order = np.argsort(test_indices, kind="stable")
    test_indices = test_indices[canonical_test_order]
    expected_train_ids = tuple(bundle.sample_ids[index] for index in train_indices)
    expected_train_domains = tuple(bundle.domain_ids[index] for index in train_indices)
    expected_test_ids = tuple(bundle.sample_ids[index] for index in test_indices)
    expected_test_domains = tuple(bundle.domain_ids[index] for index in test_indices)
    if (selection.outer_test_ids, selection.outer_test_domains) != (
        expected_test_ids,
        expected_test_domains,
    ):
        _raise_validation("selection outer test identity is not canonical")
    if len(set(expected_train_domains)) != 5 or len(set(expected_test_domains)) != 1:
        _raise_validation("outer split must contain five training and one test domain")
    if set(expected_train_ids) | set(expected_test_ids) != set(bundle.sample_ids):
        _raise_validation("outer split does not cover the feature bundle")
    train_state = _validate_issued_pca_view(
        outer_train_authority,
        expected_train_ids,
        expected_train_domains,
        name="outer training authority",
        expected_domain_count=5,
    )
    test_state = _validate_issued_pca_view(
        outer_test_authority,
        expected_test_ids,
        expected_test_domains,
        name="outer test authority",
        expected_domain_count=1,
    )
    _validate_pca_view_matches_bundle(
        bundle, outer_train_authority, name="outer training authority"
    )
    _validate_pca_view_matches_bundle(
        bundle, outer_test_authority, name="outer test authority"
    )
    return train_indices, test_indices, expected_test_domains[0], train_state, test_state, _source_state(
        bundle._source_authority, name="feature bundle source"
    )


def _resolve_inner_pca_authorities(
    bundle: FeatureBundle,
    fit_indices: np.ndarray,
    fit_sample_ids: Sequence[str],
    fit_domain_ids: Sequence[str],
    *,
    inner_train_authority: object | None,
    parent_outer_authority: object | None,
    heldout_domain: str | None,
    inner_query_domain: str | None,
) -> tuple[object, object, str, str]:
    if parent_outer_authority is None:
        _raise_validation(
            "production inner PCA requires an issued parent outer-training view"
        )
    if not _typed_v3_source(parent_outer_authority):
        _raise_validation(
            "parent outer-training PCA authority must be an issued V3DataView"
        )
    parent_ids = tuple(np.asarray(parent_outer_authority.sample_ids).tolist())
    parent_domains = tuple(np.asarray(parent_outer_authority.dataset_ids).tolist())
    _validate_issued_pca_view(
        parent_outer_authority,
        parent_ids,
        parent_domains,
        name="parent outer-training PCA authority",
        expected_domain_count=5,
    )
    _validate_pca_view_matches_bundle(
        bundle,
        parent_outer_authority,
        name="parent outer-training PCA authority",
    )
    if heldout_domain is None:
        _raise_validation("production inner PCA requires the outer heldout domain")
    if heldout_domain in set(parent_domains):
        _raise_validation("outer heldout domain occurs in parent outer training")
    if (
        type(inner_query_domain) is not str
        or not inner_query_domain
        or inner_query_domain == heldout_domain
        or inner_query_domain not in set(parent_domains)
    ):
        _raise_validation(
            "production inner PCA requires an explicit inner query domain"
        )
    parent_positions = {sample_id: index for index, sample_id in enumerate(parent_ids)}
    try:
        relative_indices = np.asarray(
            [parent_positions[sample_id] for sample_id in fit_sample_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise FeatureValidationError(
            "inner PCA rows are not contained in the parent outer view"
        ) from error
    expected_inner = parent_outer_authority.subset(relative_indices)
    authority = expected_inner if inner_train_authority is None else inner_train_authority
    _validate_issued_pca_view(
        authority,
        fit_sample_ids,
        fit_domain_ids,
        name="inner-training PCA authority",
        expected_domain_count=4,
    )
    _validate_pca_view_matches_bundle(
        bundle,
        authority,
        name="inner-training PCA authority",
    )
    parent_state = _source_state(
        parent_outer_authority, name="parent outer-training PCA authority"
    )
    return authority, parent_outer_authority, heldout_domain, parent_state


def _fit_production_pca(
    bundle: FeatureBundle,
    train_embeddings: np.ndarray,
    *,
    n_components: int,
    fit_sample_ids: Sequence[str],
    fit_domain_ids: Sequence[str],
    training_authority: object,
    heldout_domain: str,
    parent_outer_authority: object | None = None,
    inner_query_domain: str | None = None,
) -> FoldLocalPCA:
    """Call the embeddings authority API; never substitute test-only SVD."""

    fit_indices = np.asarray(
        [
            {sample_id: index for index, sample_id in enumerate(bundle.sample_ids)}[
                sample_id
            ]
            for sample_id in fit_sample_ids
        ],
        dtype=np.int64,
    )
    expected_embeddings = np.ascontiguousarray(bundle.frozen_embedding[fit_indices])
    supplied_embeddings = np.asarray(train_embeddings)
    if supplied_embeddings.shape != expected_embeddings.shape or not np.array_equal(
        supplied_embeddings, expected_embeddings
    ):
        _raise_validation("production PCA training embeddings are not bundle-authorized")

    kwargs: dict[str, object] = {
        "n_components": n_components,
        "fit_sample_ids": fit_sample_ids,
        "fit_domain_ids": fit_domain_ids,
        "outer_train_authority": (
            parent_outer_authority
            if parent_outer_authority is not None
            else training_authority
        ),
        "heldout_domain": heldout_domain,
    }
    if parent_outer_authority is not None:
        if inner_query_domain is None:
            _raise_validation(
                "production inner PCA requires an explicit inner query domain"
            )
        parameters = inspect.signature(fit_embedding_pca).parameters
        required = {
            "inner_train_authority",
            "parent_outer_authority",
            "inner_query_domain",
        }
        if not required.issubset(parameters):
            _raise_validation(
                "embeddings PCA API lacks the issued inner/parent authority contract"
            )
        kwargs["inner_train_authority"] = training_authority
        kwargs["parent_outer_authority"] = parent_outer_authority
        kwargs["inner_query_domain"] = inner_query_domain
    try:
        model = fit_embedding_pca(train_embeddings, **kwargs)
    except EmbeddingFeatureValidationError as error:
        if str(error) == f"PCA components exceed the training fold rank: {n_components}":
            raise _PcaRankUnavailable(n_components) from error
        raise _safe_embedding_error(error) from error
    except (TypeError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        raise _safe_embedding_error(error) from error
    if not isinstance(model, FoldLocalPCA):
        _raise_validation("embeddings PCA authority returned an invalid model")
    expected_digest = _embedding_state_hash(np.asarray(train_embeddings))
    if model.fit_embeddings_sha256 != expected_digest:
        _raise_validation("PCA model training embedding digest does not match input")
    return model


def _assemble_from_components(
    name: str,
    components: Sequence[tuple[str, np.ndarray, Sequence[str], Sequence[str]]],
    *,
    sample_ids: Sequence[str],
    domain_ids: Sequence[str],
    source_sha256: str,
    authority: str = "bundle",
) -> FeatureMatrix:
    # The enclosing assembly installs the bundle before issuing each matrix.
    if _MATRIX_CONSTRUCTION_CONTEXT is None:
        _raise_validation("feature matrix requires a trusted bundle factory")
    values = np.column_stack([component[1] for component in components])
    names: list[str] = []
    units: list[str] = []
    blocks: list[str] = []
    for block_name, _value, block_names, block_units in components:
        names.extend(block_names)
        units.extend(block_units)
        blocks.append(block_name)
    return _block(
        name,
        values,
        names,
        units,
        blocks,
        sample_ids=sample_ids,
        domain_ids=domain_ids,
        source_sha256=source_sha256,
        authority=authority,
    )


def _pca_transform_rows(
    bundle: FeatureBundle,
    indices: np.ndarray,
    *,
    fit_indices: np.ndarray | None,
    pca_dimension: int | None,
    pca_model: FoldLocalPCA | None,
    fit_sample_ids: Sequence[str] | None = None,
    fit_domain_ids: Sequence[str] | None = None,
    outer_train_authority: object | None = None,
    heldout_domain: str | None = None,
    parent_outer_authority: object | None = None,
    inner_train_authority: object | None = None,
    inner_query_domain: str | None = None,
) -> tuple[np.ndarray, FoldLocalPCA | _TestOnlyPCA | None]:
    if pca_model is not None:
        if bundle.source_state_sha256 == "test-only":
            if any(
                authority is not None
                for authority in (
                    outer_train_authority,
                    inner_train_authority,
                    parent_outer_authority,
                )
            ) or heldout_domain is not None:
                _raise_validation(
                    "test-only PCA replay cannot use production data authorities"
                )
        elif parent_outer_authority is not None or inner_train_authority is not None:
            if fit_indices is None:
                _raise_validation("inner PCA replay requires fit indices")
            authority, parent, resolved_heldout, _inner_state = (
                _resolve_inner_pca_authorities(
                    bundle,
                    fit_indices,
                    tuple(bundle.sample_ids[index] for index in fit_indices),
                    tuple(bundle.domain_ids[index] for index in fit_indices),
                    inner_train_authority=inner_train_authority,
                    parent_outer_authority=parent_outer_authority,
                    heldout_domain=heldout_domain,
                    inner_query_domain=inner_query_domain,
                )
            )
            if not isinstance(pca_model, FoldLocalPCA):
                _raise_validation("production PCA replay model has an invalid type")
            if pca_model.heldout_domain != resolved_heldout:
                _raise_validation("PCA replay heldout domain does not match authority")
            if pca_model.outer_train_state_sha256 != _source_state(
                parent, name="parent outer-training PCA authority"
            ):
                _raise_validation("PCA replay parent authority state changed")
            outer_train_authority = authority
            parent_outer_authority = parent
        else:
            if fit_indices is None:
                _raise_validation("PCA replay requires fit indices")
            authority, resolved_heldout, authority_state = _resolve_outer_pca_authority(
                bundle,
                fit_indices,
                tuple(bundle.sample_ids[index] for index in fit_indices),
                tuple(bundle.domain_ids[index] for index in fit_indices),
                outer_train_authority=outer_train_authority,
                heldout_domain=heldout_domain,
            )
            if not isinstance(pca_model, FoldLocalPCA):
                _raise_validation("production PCA replay model has an invalid type")
            if pca_model.heldout_domain != resolved_heldout:
                _raise_validation("PCA replay heldout domain does not match authority")
            if pca_model.outer_train_state_sha256 != authority_state:
                _raise_validation("PCA replay authority state changed")
            outer_train_authority = authority
        training_digest = (
            _training_embedding_digest(
                bundle,
                tuple(bundle.sample_ids[index] for index in fit_indices),
                tuple(bundle.domain_ids[index] for index in fit_indices),
            )
            if fit_indices is not None
            else None
        )
        model = _validate_pca_model(
            pca_model,
            pca_dimension,
            fit_sample_ids=fit_sample_ids,
            fit_domain_ids=fit_domain_ids,
            bundle=bundle,
            training_authority=outer_train_authority or inner_train_authority,
            parent_authority=parent_outer_authority,
            training_embedding_digest=training_digest,
        )
    else:
        if pca_dimension is None:
            _raise_validation(
                "a PCA dimension or fitted PCA model is required for frozen candidates"
            )
        dimension = _validate_pca_dimension(pca_dimension)
        if fit_indices is None:
            _raise_validation("fit_indices are required for fold-local PCA")
        if fit_sample_ids is None or fit_domain_ids is None:
            _raise_validation("fit_sample_ids and fit_domain_ids are required for PCA")
        selected_sample_ids = tuple(
            bundle.sample_ids[index] for index in fit_indices
        )
        selected_domain_ids = tuple(bundle.domain_ids[index] for index in fit_indices)
        registration_training_authority: object | None = None
        registration_parent_authority: object | None = None
        registration_embedding_digest = _training_embedding_digest(
            bundle, selected_sample_ids, selected_domain_ids
        )
        if bundle.source_state_sha256 == "test-only":
            if any(
                authority is not None
                for authority in (
                    outer_train_authority,
                    inner_train_authority,
                    parent_outer_authority,
                )
            ) or heldout_domain is not None:
                _raise_validation(
                    "test-only PCA cannot use production data authorities"
                )
            model = _fit_test_only_pca(
                bundle.frozen_embedding[fit_indices],
                n_components=dimension,
                fit_sample_ids=selected_sample_ids,
                fit_domain_ids=selected_domain_ids,
            )
        else:
            if parent_outer_authority is not None or inner_train_authority is not None:
                authority, parent, resolved_heldout, _inner_state = (
                    _resolve_inner_pca_authorities(
                        bundle,
                        fit_indices,
                        selected_sample_ids,
                        selected_domain_ids,
                        inner_train_authority=inner_train_authority,
                        parent_outer_authority=parent_outer_authority,
                        heldout_domain=heldout_domain,
                        inner_query_domain=inner_query_domain,
                    )
                )
                model = _fit_production_pca(
                    bundle,
                    bundle.frozen_embedding[fit_indices],
                    n_components=dimension,
                    fit_sample_ids=selected_sample_ids,
                    fit_domain_ids=selected_domain_ids,
                    training_authority=authority,
                    heldout_domain=resolved_heldout,
                    parent_outer_authority=parent,
                    inner_query_domain=inner_query_domain,
                )
                registration_training_authority = authority
                registration_parent_authority = parent
            else:
                authority, resolved_heldout, _authority_state = (
                    _resolve_outer_pca_authority(
                        bundle,
                        fit_indices,
                        selected_sample_ids,
                        selected_domain_ids,
                        outer_train_authority=outer_train_authority,
                        heldout_domain=heldout_domain,
                    )
                )
                model = _fit_production_pca(
                    bundle,
                    bundle.frozen_embedding[fit_indices],
                    n_components=dimension,
                    fit_sample_ids=selected_sample_ids,
                    fit_domain_ids=selected_domain_ids,
                    training_authority=authority,
                    heldout_domain=resolved_heldout,
                )
                registration_training_authority = authority
        _register_pca_model(
            model,
            selected_sample_ids,
            selected_domain_ids,
            bundle,
            training_authority=registration_training_authority,
            parent_authority=registration_parent_authority,
            training_embedding_digest=registration_embedding_digest,
        )
    transformed = _transform_pca_model(model, bundle.frozen_embedding[indices])
    return np.asarray(transformed, dtype=np.float64), model


def assemble_feature_matrices(
    bundle: FeatureBundle,
    *,
    fit_indices: Sequence[int] | np.ndarray | None = None,
    fit_sample_ids: Sequence[str] | None = None,
    fit_domain_ids: Sequence[str] | None = None,
    pca_dimension: int | None = None,
    pca_model: FoldLocalPCA | None = None,
    outer_train_authority: object | None = None,
    heldout_domain: str | None = None,
) -> Mapping[str, FeatureMatrix]:
    """Assemble all registered candidates from one row-aligned feature bundle.

    When a PCA dimension is requested, ``fit_indices`` identify the only rows used
    to fit it.  The returned matrices are for all rows and are intended for a
    diagnostic assembly path; fold fitting uses the private row-sliced helper below.
    """

    if not isinstance(bundle, FeatureBundle):
        _raise_validation("bundle must be a FeatureBundle")
    rows = bundle.n_rows
    all_indices = np.arange(rows, dtype=np.int64)
    selected_fit = (
        None
        if fit_indices is None
        else _indices(fit_indices, length=rows, name="fit_indices")
    )
    requires_frozen = pca_dimension is not None or pca_model is not None
    frozen = None
    selected_fit_sample_ids: tuple[str, ...] | None = None
    selected_fit_domain_ids: tuple[str, ...] | None = None
    if requires_frozen:
        if selected_fit is None:
            _raise_validation("fit_indices are required for fold-local PCA")
        if fit_sample_ids is None or fit_domain_ids is None:
            _raise_validation("fit_sample_ids and fit_domain_ids are required for PCA")
        selected_fit_sample_ids = _ids(
            fit_sample_ids,
            length=selected_fit.size,
            name="fit_sample_ids",
        )
        selected_fit_domain_ids = _ids(
            fit_domain_ids,
            length=selected_fit.size,
            name="fit_domain_ids",
        )
        expected_sample_ids = tuple(bundle.sample_ids[index] for index in selected_fit)
        expected_domain_ids = tuple(bundle.domain_ids[index] for index in selected_fit)
        if selected_fit_sample_ids != expected_sample_ids:
            _raise_validation("fit_sample_ids are not the exact bundle subset")
        if selected_fit_domain_ids != expected_domain_ids:
            _raise_validation("fit_domain_ids are not the exact bundle subset")
        frozen, _ = _pca_transform_rows(
            bundle,
            all_indices,
            fit_indices=selected_fit,
            pca_dimension=pca_dimension,
            pca_model=pca_model,
            fit_sample_ids=selected_fit_sample_ids,
            fit_domain_ids=selected_fit_domain_ids,
            outer_train_authority=outer_train_authority,
            heldout_domain=heldout_domain,
        )

    global _MATRIX_CONSTRUCTION_CONTEXT
    previous_matrix_context = _MATRIX_CONSTRUCTION_CONTEXT
    _MATRIX_CONSTRUCTION_CONTEXT = bundle

    def assemble(
        name: str,
        components: Sequence[tuple[str, np.ndarray, Sequence[str], Sequence[str]]],
    ) -> FeatureMatrix:
        return _assemble_from_components(
            name,
            components,
            sample_ids=bundle.sample_ids,
            domain_ids=bundle.domain_ids,
            source_sha256=bundle.state_sha256,
            authority="bundle",
        )

    metadata = bundle.metadata
    surface = bundle.surface_stats
    scalar = bundle.scalar_internal
    morphology = bundle.morphology
    a = assemble(
        "A_surface",
        (
            (
                "metadata",
                metadata,
                _METADATA_FEATURE_NAMES,
                ("1",) * METADATA_FEATURE_COUNT,
            ),
            ("surface_stats", surface, _SURFACE_FEATURE_NAMES, _SURFACE_FEATURE_UNITS),
        ),
    )
    b_scalar = assemble(
        "B_scalar",
        (
            ("A_surface", a.matrix, a.feature_names, a.units),
            ("scalar_internal", scalar, SCALAR_FEATURE_NAMES, SCALAR_FEATURE_UNITS),
        ),
    )
    b_morph = assemble(
        "B_morph",
        (
            ("A_surface", a.matrix, a.feature_names, a.units),
            ("morphology", morphology, _MORPHOLOGY_NAMES, _MORPHOLOGY_UNITS),
        ),
    )
    matrices: dict[str, FeatureMatrix] = {
        "A_surface": a,
        "B_scalar": b_scalar,
        "B_morph": b_morph,
        "I_morph": assemble(
            "I_morph",
            (
                (
                    "metadata",
                    metadata,
                    _METADATA_FEATURE_NAMES,
                    ("1",) * METADATA_FEATURE_COUNT,
                ),
                ("morphology", morphology, _MORPHOLOGY_NAMES, _MORPHOLOGY_UNITS),
            ),
        ),
    }
    if frozen is not None:
        names = _pca_names(int(frozen.shape[1]))
        units = ("1",) * frozen.shape[1]
        b_frozen = assemble(
            "B_frozen",
            (
                ("A_surface", a.matrix, a.feature_names, a.units),
                ("frozen_embedding", frozen, names, units),
            ),
        )
        b_combined = assemble(
            "B_combined",
            (
                ("A_surface", a.matrix, a.feature_names, a.units),
                ("morphology", morphology, _MORPHOLOGY_NAMES, _MORPHOLOGY_UNITS),
                ("frozen_embedding", frozen, names, units),
            ),
        )
        matrices.update(
            {
                "B_frozen": b_frozen,
                "B_combined": b_combined,
                "I_frozen": assemble(
                    "I_frozen",
                    (
                        (
                            "metadata",
                            metadata,
                            _METADATA_FEATURE_NAMES,
                            ("1",) * METADATA_FEATURE_COUNT,
                        ),
                        ("frozen_embedding", frozen, names, units),
                    ),
                ),
                "I_combined": assemble(
                    "I_combined",
                    (
                        (
                            "metadata",
                            metadata,
                            _METADATA_FEATURE_NAMES,
                            ("1",) * METADATA_FEATURE_COUNT,
                        ),
                        (
                            "morphology",
                            morphology,
                            _MORPHOLOGY_NAMES,
                            _MORPHOLOGY_UNITS,
                        ),
                        ("frozen_embedding", frozen, names, units),
                    ),
                ),
            }
        )
    _MATRIX_CONSTRUCTION_CONTEXT = previous_matrix_context
    return MappingProxyType(matrices)


@dataclass(frozen=True)
class FoldPreprocessor:
    """Fold-local median-safe (mean) imputer followed by StandardScaler."""

    imputer_statistics_: np.ndarray
    mean_: np.ndarray
    scale_: np.ndarray
    fit_sample_ids: tuple[str, ...]
    fit_domain_ids: tuple[str, ...]
    n_features_in_: int
    state_sha256: str
    _fit_capability: InitVar[object | None] = None

    def __post_init__(self, _fit_capability: object | None) -> None:
        imputer = _as_vector(self.imputer_statistics_, "imputer statistics")
        mean = _as_vector(self.mean_, "feature means")
        scale = _as_vector(self.scale_, "feature scales")
        if not (imputer.shape == mean.shape == scale.shape):
            _raise_validation("preprocessor state feature shapes do not align")
        sample_ids = _strict_string_tuple(self.fit_sample_ids, name="fit_sample_ids")
        domain_ids = _strict_string_tuple(self.fit_domain_ids, name="fit_domain_ids")
        if not sample_ids or not domain_ids or len(sample_ids) != len(domain_ids):
            _raise_validation("preprocessor fit IDs and domains are required")
        if len(set(sample_ids)) != len(sample_ids):
            _raise_validation("preprocessor fit sample IDs must be unique")
        if type(self.n_features_in_) is not int or self.n_features_in_ != mean.size:
            _raise_validation("preprocessor n_features_in_ is inconsistent")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            _raise_validation("preprocessor scale must be finite and strictly positive")
        if not _validate_preprocessor_constructor_capability(_fit_capability):
            _raise_validation("preprocessor requires a fit issuer authority")
        immutable_imputer = _readonly_array(imputer)
        immutable_mean = _readonly_array(mean)
        immutable_scale = _readonly_array(scale)
        object.__setattr__(self, "imputer_statistics_", immutable_imputer)
        object.__setattr__(self, "mean_", immutable_mean)
        object.__setattr__(self, "scale_", immutable_scale)
        object.__setattr__(self, "fit_sample_ids", sample_ids)
        object.__setattr__(self, "fit_domain_ids", domain_ids)
        state = _state_hash(
            immutable_imputer,
            immutable_mean,
            immutable_scale,
            sample_ids,
            domain_ids,
            self.n_features_in_,
        )
        object.__setattr__(self, "state_sha256", state)
        _register_state(self, state)

    @property
    def statistics_(self) -> np.ndarray:
        return self.imputer_statistics_

    @property
    def feature_mean_(self) -> np.ndarray:
        return self.mean_

    @property
    def feature_scale_(self) -> np.ndarray:
        return self.scale_

    def transform(self, matrix: np.ndarray | FeatureMatrix) -> np.ndarray:
        validate_fold_preprocessor_state(self)
        if isinstance(matrix, FeatureMatrix):
            validate_feature_matrix(matrix)
        value = matrix.matrix if isinstance(matrix, FeatureMatrix) else matrix
        value = _as_matrix(
            value, "query features", columns=self.n_features_in_, allow_nan=True
        )
        if np.any(np.isinf(value)):
            _raise_validation("query features must not contain infinity")
        imputed = np.where(np.isnan(value), self.imputer_statistics_, value)
        output = (imputed - self.mean_) / self.scale_
        if not np.all(np.isfinite(output)):
            _raise_validation("scaled query features must be finite")
        return np.asarray(output, dtype=np.float64)


def _preprocessor_state(model: FoldPreprocessor) -> str:
    return _state_hash(
        model.imputer_statistics_,
        model.mean_,
        model.scale_,
        model.fit_sample_ids,
        model.fit_domain_ids,
        model.n_features_in_,
    )


def validate_fold_preprocessor_state(model: FoldPreprocessor) -> bool:
    if not isinstance(model, FoldPreprocessor):
        _raise_validation("preprocessor state has an invalid type")
    current = _preprocessor_state(model)
    _assert_authority(model, current, name="preprocessor")
    if any(
        array.flags.writeable
        for array in (
            model.imputer_statistics_,
            model.mean_,
            model.scale_,
        )
    ):
        _raise_validation("preprocessor state arrays must be read-only")
    if not np.all(np.isfinite(model.imputer_statistics_)):
        _raise_validation("preprocessor state must be finite")
    if not np.all(np.isfinite(model.scale_)) or np.any(model.scale_ <= 0.0):
        _raise_validation("preprocessor scale must be finite and strictly positive")
    if current != model.state_sha256:
        _raise_validation("preprocessor state hash mismatch")
    return True


def fit_fold_preprocessor(
    matrix: np.ndarray | FeatureMatrix,
    *,
    fit_sample_ids: Sequence[str] | None = None,
    fit_domain_ids: Sequence[str] | None = None,
) -> FoldPreprocessor:
    if isinstance(matrix, FeatureMatrix):
        validate_feature_matrix(matrix)
    value = matrix.matrix if isinstance(matrix, FeatureMatrix) else matrix
    value = _as_matrix(value, "fit features", allow_nan=True)
    if value.shape[0] == 0:
        _raise_validation("fit features cannot be empty")
    imputer = np.nanmean(value, axis=0)
    if not np.all(np.isfinite(imputer)):
        _raise_validation("every fitted feature must contain one finite value")
    filled = np.where(np.isnan(value), imputer, value)
    mean = np.mean(filled, axis=0, dtype=np.float64)
    scale = np.std(filled, axis=0, dtype=np.float64, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        _raise_validation("preprocessor scale must be finite and strictly positive")
    sample_ids = _ids(fit_sample_ids, length=value.shape[0], name="fit_sample_ids")
    domain_ids = _ids(fit_domain_ids, length=value.shape[0], name="fit_domain_ids")
    state = _state_hash(
        imputer,
        mean,
        scale,
        sample_ids,
        domain_ids,
        int(value.shape[1]),
    )
    return _issue_fold_preprocessor(
        imputer_statistics_=np.asarray(imputer, dtype=np.float64),
        mean_=np.asarray(mean, dtype=np.float64),
        scale_=np.asarray(scale, dtype=np.float64),
        fit_sample_ids=sample_ids,
        fit_domain_ids=domain_ids,
        n_features_in_=int(value.shape[1]),
        state_sha256=state,
    )


_issue_fold_preprocessor = _fold_state_issuance_runtime(
    FoldPreprocessor,
    _preprocessor_fit_capability,
    frozenset({fit_fold_preprocessor.__code__}),
)


fit_fold_scaler = fit_fold_preprocessor


@dataclass(frozen=True)
class FoldRidgeModel:
    preprocessor: FoldPreprocessor
    alpha: float
    coef_: np.ndarray
    intercept_: float
    fit_sample_ids: tuple[str, ...]
    fit_domain_ids: tuple[str, ...]
    state_sha256: str
    _fit_capability: InitVar[object | None] = None

    def __post_init__(self, _fit_capability: object | None) -> None:
        validate_fold_preprocessor_state(self.preprocessor)
        coef = _as_vector(self.coef_, "ridge coefficients")
        if coef.size != self.preprocessor.n_features_in_:
            _raise_validation("ridge coefficients do not match n_features_in_")
        if not np.isfinite(self.intercept_):
            _raise_validation("ridge intercept must be finite")
        sample_ids = tuple(self.fit_sample_ids)
        domain_ids = tuple(self.fit_domain_ids)
        if (sample_ids, domain_ids) != (
            self.preprocessor.fit_sample_ids,
            self.preprocessor.fit_domain_ids,
        ):
            _raise_validation("ridge fit IDs do not match preprocessor fit IDs")
        if not _validate_ridge_constructor_capability(_fit_capability):
            _raise_validation("Ridge model requires a fit issuer authority")
        immutable_coef = _readonly_array(coef)
        object.__setattr__(self, "coef_", immutable_coef)
        object.__setattr__(self, "fit_sample_ids", sample_ids)
        object.__setattr__(self, "fit_domain_ids", domain_ids)
        state = _state_hash(
            self.preprocessor.state_sha256,
            RIDGE_ALPHA,
            immutable_coef,
            float(self.intercept_),
            sample_ids,
            domain_ids,
            self.preprocessor.n_features_in_,
        )
        object.__setattr__(self, "state_sha256", state)
        _register_state(self, state)

    @property
    def coef(self) -> np.ndarray:
        return self.coef_

    @property
    def intercept(self) -> float:
        return self.intercept_

    def predict(self, matrix: np.ndarray | FeatureMatrix) -> np.ndarray:
        validate_fold_ridge_state(self)
        transformed = self.preprocessor.transform(matrix)
        output = transformed @ self.coef_ + self.intercept_
        if not np.all(np.isfinite(output)):
            _raise_validation("ridge predictions must be finite")
        return np.asarray(output, dtype=np.float64)


def fit_fold_ridge(
    train_x: np.ndarray | FeatureMatrix,
    train_y: Sequence[float] | np.ndarray,
    *,
    alpha: float = RIDGE_ALPHA,
    fit_sample_ids: Sequence[str] | None = None,
    fit_domain_ids: Sequence[str] | None = None,
) -> FoldRidgeModel:
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(
        alpha, (int, float, np.integer, np.floating)
    ):
        _raise_validation("alpha must be the registered numeric value 10.0")
    if float(alpha) != RIDGE_ALPHA:
        _raise_validation("alpha must equal the registered Ridge alpha 10.0")
    if isinstance(train_x, FeatureMatrix):
        validate_feature_matrix(train_x)
    value = train_x.matrix if isinstance(train_x, FeatureMatrix) else train_x
    x = _as_matrix(value, "train features", allow_nan=True)
    try:
        raw_y = np.asarray(train_y)
    except (TypeError, ValueError) as error:
        raise FeatureValidationError("train response must be numeric") from error
    if raw_y.ndim == 2 and raw_y.shape[1] == 1:
        raw_y = raw_y[:, 0]
    y = _as_vector(raw_y, "train response")
    if y.shape[0] != x.shape[0]:
        _raise_validation(
            "train response must be a finite vector aligned with features"
        )
    preprocessor = fit_fold_preprocessor(
        x,
        fit_sample_ids=fit_sample_ids,
        fit_domain_ids=fit_domain_ids,
    )
    transformed = preprocessor.transform(x)
    x_mean = np.mean(transformed, axis=0, dtype=np.float64)
    y_mean = float(np.mean(y, dtype=np.float64))
    centered_x = transformed - x_mean
    centered_y = y - y_mean
    gram = centered_x.T @ centered_x
    rhs = centered_x.T @ centered_y
    regularized = gram + RIDGE_ALPHA * np.eye(x.shape[1], dtype=np.float64)
    try:
        coef = np.linalg.solve(regularized, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(regularized, rhs, rcond=None)[0]
    intercept = y_mean - float(x_mean @ coef)
    if not np.all(np.isfinite(coef)) or not np.isfinite(intercept):
        _raise_validation("ridge fit produced non-finite parameters")
    sample_ids = preprocessor.fit_sample_ids
    domain_ids = preprocessor.fit_domain_ids
    state = _state_hash(
        preprocessor.state_sha256,
        RIDGE_ALPHA,
        np.asarray(coef, dtype=np.float64),
        float(intercept),
        sample_ids,
        domain_ids,
        preprocessor.n_features_in_,
    )
    return _issue_fold_ridge(
        preprocessor=preprocessor,
        alpha=RIDGE_ALPHA,
        coef_=np.asarray(coef, dtype=np.float64),
        intercept_=float(intercept),
        fit_sample_ids=sample_ids,
        fit_domain_ids=domain_ids,
        state_sha256=state,
    )


_issue_fold_ridge = _fold_state_issuance_runtime(
    FoldRidgeModel,
    _ridge_fit_capability,
    frozenset({fit_fold_ridge.__code__}),
)
del _preprocessor_fit_capability
del _ridge_fit_capability


fit_ridge = fit_fold_ridge


def _outer_fitted_state(model: OuterFittedCandidate) -> str:
    return _state_hash(
        "outer-fitted-candidate",
        model.candidate,
        model.selection_label,
        model.pca_dimension,
        model.alpha,
        model.ridge_model.state_sha256,
        None if model.pca_model is None else model.pca_model.state_sha256,
        model.fit_sample_ids,
        model.fit_domain_ids,
        model.test_sample_ids,
        model.test_domain_ids,
        model.predictions,
        model.bundle_state_sha256,
        model.response_training_state_sha256,
        model.selection_state_sha256,
        model.feature_bundle_state_sha256,
        model.source_authority_state_sha256,
        model.response_state_sha256,
        model.outer_train_authority_state_sha256,
        model.outer_test_authority_state_sha256,
    )


@dataclass(frozen=True)
class OuterFittedCandidate:
    """Production outer-fold refit state and ordered held-out predictions."""

    candidate: str
    selection_label: str
    pca_dimension: int | None
    alpha: float
    ridge_model: FoldRidgeModel
    pca_model: FoldLocalPCA | None
    fit_sample_ids: tuple[str, ...]
    fit_domain_ids: tuple[str, ...]
    test_sample_ids: tuple[str, ...]
    test_domain_ids: tuple[str, ...]
    predictions: np.ndarray
    bundle_state_sha256: str
    response_training_state_sha256: str
    selection_state_sha256: str
    feature_bundle_state_sha256: str
    source_authority_state_sha256: str
    response_state_sha256: str
    outer_train_authority_state_sha256: str
    outer_test_authority_state_sha256: str
    state_sha256: str

    def __post_init__(self) -> None:
        context = _OUTER_CONSTRUCTION_CONTEXT
        if context is None:
            _raise_validation(
                "OuterFittedCandidate must be issued by the production outer factory"
            )
        candidate = self.candidate
        if candidate not in _OUTER_CANDIDATES:
            _raise_validation("outer fitted candidate is not registered")
        if self.selection_label not in {
            "B_field_selected",
            "I_field_selected",
            *_BASELINE_OUTER_CANDIDATES,
        }:
            _raise_validation("outer fitted selection label is not registered")
        if self.selection_label == "B_field_selected" and candidate not in _PRIMARY_OUTER_CANDIDATES:
            _raise_validation("outer fitted candidate does not match B selection")
        if self.selection_label == "I_field_selected" and candidate not in _INTERNAL_OUTER_CANDIDATES:
            _raise_validation("outer fitted candidate does not match I selection")
        if self.selection_label in _BASELINE_OUTER_CANDIDATES and candidate != self.selection_label:
            _raise_validation("outer fitted candidate does not match baseline selection")
        if isinstance(self.alpha, (bool, np.bool_)) or float(self.alpha) != RIDGE_ALPHA:
            _raise_validation("outer fitted Ridge alpha must equal 10.0")
        if self.pca_dimension is not None:
            _validate_pca_dimension(self.pca_dimension)
        if candidate in _FROZEN_CANDIDATES and self.pca_dimension is None:
            _raise_validation("frozen outer candidate requires a PCA dimension")
        if candidate not in _FROZEN_CANDIDATES and self.pca_dimension is not None:
            _raise_validation("non-frozen outer candidate cannot carry PCA")
        fit_ids = _strict_string_tuple(self.fit_sample_ids, name="outer fit sample IDs")
        fit_domains = _strict_string_tuple(self.fit_domain_ids, name="outer fit domains")
        test_ids = _strict_string_tuple(self.test_sample_ids, name="outer test sample IDs")
        test_domains = _strict_string_tuple(self.test_domain_ids, name="outer test domains")
        if len(fit_ids) != len(fit_domains) or len(test_ids) != len(test_domains):
            _raise_validation("outer fitted IDs and domains are not aligned")
        if not fit_ids or not test_ids or len(set(fit_ids)) != len(fit_ids):
            _raise_validation("outer fitted row identities are invalid")
        if len(set(test_ids)) != len(test_ids) or set(fit_ids) & set(test_ids):
            _raise_validation("outer fitted train and test identities overlap")
        predictions = _as_vector(self.predictions, "outer predictions")
        if predictions.size != len(test_ids):
            _raise_validation("outer predictions are not ordered with held-out IDs")
        validate_fold_ridge_state(self.ridge_model)
        if self.ridge_model.alpha != RIDGE_ALPHA:
            _raise_validation("outer fitted Ridge alpha is not registered")
        if self.ridge_model.fit_sample_ids != fit_ids or self.ridge_model.fit_domain_ids != fit_domains:
            _raise_validation("outer Ridge fit identity does not match outer state")
        if self.pca_model is not None:
            if not isinstance(self.pca_model, FoldLocalPCA):
                _raise_validation("outer production PCA state has an invalid type")
            _validate_pca_model(self.pca_model, self.pca_dimension)
            if self.pca_model.fit_sample_ids != fit_ids or self.pca_model.fit_domain_ids != fit_domains:
                _raise_validation("outer PCA fit identity does not match outer state")
        for name, value in (
            ("bundle", self.bundle_state_sha256),
            ("response", self.response_training_state_sha256),
            ("selection", self.selection_state_sha256),
            ("feature bundle", self.feature_bundle_state_sha256),
            ("source authority", self.source_authority_state_sha256),
            ("response", self.response_state_sha256),
            ("outer train authority", self.outer_train_authority_state_sha256),
            ("outer test authority", self.outer_test_authority_state_sha256),
        ):
            if type(value) is not str or len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                _raise_validation(f"{name} state digest is invalid")
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "selection_label", self.selection_label)
        object.__setattr__(self, "alpha", RIDGE_ALPHA)
        object.__setattr__(self, "fit_sample_ids", fit_ids)
        object.__setattr__(self, "fit_domain_ids", fit_domains)
        object.__setattr__(self, "test_sample_ids", test_ids)
        object.__setattr__(self, "test_domain_ids", test_domains)
        object.__setattr__(self, "predictions", _readonly_array(predictions))
        expected = _outer_fitted_state(self)
        if self.state_sha256 == "0" * 64:
            object.__setattr__(self, "state_sha256", expected)
        elif self.state_sha256 != expected:
            _raise_validation("outer fitted state hash mismatch")
        _register_state(self, expected)

    @property
    def prediction(self) -> np.ndarray:
        return self.predictions

    @property
    def selected_candidate(self) -> str:
        return self.candidate

    def __copy__(self) -> OuterFittedCandidate:
        raise TypeError("production outer fitted candidates cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> OuterFittedCandidate:
        raise TypeError("production outer fitted candidates cannot be copied")

    def __reduce__(self) -> Any:
        raise TypeError("production outer fitted candidates cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("production outer fitted candidates cannot be serialized")


def _validate_metric_inputs(
    target: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
    domain_ids: Sequence[str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    target_value = _as_vector(target, "metric target")
    prediction_value = _as_vector(prediction, "metric prediction")
    if target_value.shape != prediction_value.shape or not target_value.size:
        _raise_validation("metric values must have matching non-empty lengths")
    if isinstance(domain_ids, (str, bytes)):
        _raise_validation("domain IDs must be a sequence of exact strings")
    try:
        domains = tuple(domain_ids)
    except (TypeError, ValueError) as error:
        raise FeatureValidationError(
            "domain IDs must be a sequence of exact strings"
        ) from error
    if len(domains) != target_value.size or any(
        not isinstance(value, str) or not value for value in domains
    ):
        _raise_validation("domain IDs must be non-empty exact strings")
    return target_value, prediction_value, domains


def domain_mae(
    target: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
    domain_ids: Sequence[str] | np.ndarray,
) -> dict[str, float]:
    target_value, prediction_value, domains = _validate_metric_inputs(
        target, prediction, domain_ids
    )
    try:
        with np.errstate(over="raise", invalid="raise"):
            differences = target_value - prediction_value
            if not np.all(np.isfinite(differences)):
                _raise_validation("metric differences must be finite")
            errors = np.abs(differences)
            if not np.all(np.isfinite(errors)):
                _raise_validation("metric absolute errors must be finite")
    except FloatingPointError as error:
        raise FeatureValidationError("metric arithmetic overflow") from error
    result: dict[str, float] = {}
    for domain in dict.fromkeys(domains):
        mask = np.asarray([value == domain for value in domains], dtype=bool)
        value = _stable_finite_mean(errors[mask], name="metric mean")
        if not np.isfinite(value):
            _raise_validation("metric mean must be finite")
        result[domain] = value
    return result


def equal_domain_mae(
    target: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
    domain_ids: Sequence[str] | np.ndarray,
) -> float:
    values = domain_mae(target, prediction, domain_ids)
    return _stable_finite_mean(
        np.asarray(tuple(values.values()), dtype=np.float64), name="equal-domain metric mean"
    )


def _stable_finite_mean(values: np.ndarray, *, name: str) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        _raise_validation(f"{name} must contain finite non-empty values")
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        normalized = values / scale
        mean_normalized = float(np.mean(normalized, dtype=np.float64))
    result = scale * mean_normalized
    if not np.isfinite(result):
        _raise_validation(f"{name} arithmetic overflow")
    return float(result)


@dataclass(frozen=True)
class FoldProvenance:
    candidate: str
    pca_dimension: int | None
    fold_index: int
    fit_indices: tuple[int, ...]
    query_indices: tuple[int, ...]
    fit_sample_ids: tuple[str, ...]
    query_sample_ids: tuple[str, ...]
    fit_domains: tuple[str, ...]
    query_domains: tuple[str, ...]
    targets: np.ndarray
    predictions: np.ndarray
    domain_mae: tuple[tuple[str, float], ...]
    fit_state_sha256: str
    state_sha256: str

    def __post_init__(self) -> None:
        targets = _as_vector(self.targets, "OOF targets")
        predictions = _as_vector(self.predictions, "OOF predictions")
        if targets.shape != predictions.shape:
            _raise_validation("OOF targets and predictions must align")
        fit_indices = tuple(self.fit_indices)
        query_indices = tuple(self.query_indices)
        if any(
            type(value) is not int or value < 0 for value in fit_indices + query_indices
        ):
            _raise_validation("OOF indices must be non-negative exact integers")
        fit_sample_ids = _strict_string_tuple(
            self.fit_sample_ids, name="OOF fit sample IDs"
        )
        query_sample_ids = _strict_string_tuple(
            self.query_sample_ids, name="OOF query sample IDs"
        )
        fit_domains = _strict_string_tuple(self.fit_domains, name="OOF fit domains")
        query_domains = _strict_string_tuple(
            self.query_domains, name="OOF query domains"
        )
        if len(fit_indices) != len(fit_sample_ids) or len(fit_indices) != len(
            fit_domains
        ):
            _raise_validation("OOF fit identity does not align")
        if (
            len(query_indices) != len(query_sample_ids)
            or len(query_indices) != targets.size
        ):
            _raise_validation("OOF query sample identity does not align")
        if len(query_domains) != targets.size:
            _raise_validation("OOF query domains must align with targets")
        object.__setattr__(self, "targets", _readonly_array(targets))
        object.__setattr__(self, "predictions", _readonly_array(predictions))
        object.__setattr__(self, "fit_indices", fit_indices)
        object.__setattr__(self, "query_indices", query_indices)
        object.__setattr__(self, "fit_sample_ids", fit_sample_ids)
        object.__setattr__(self, "query_sample_ids", query_sample_ids)
        object.__setattr__(self, "fit_domains", fit_domains)
        object.__setattr__(self, "query_domains", query_domains)
        object.__setattr__(
            self,
            "domain_mae",
            tuple((name, float(value)) for name, value in self.domain_mae),
        )
        if any(
            type(name) is not str or not name or not np.isfinite(value)
            for name, value in self.domain_mae
        ):
            _raise_validation("OOF domain MAE must contain finite exact values")
        _register_state(self, _fold_provenance_state(self))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "pca_dimension": self.pca_dimension,
            "fold_index": self.fold_index,
            "fit_indices": list(self.fit_indices),
            "query_indices": list(self.query_indices),
            "fit_sample_ids": list(self.fit_sample_ids),
            "query_sample_ids": list(self.query_sample_ids),
            "fit_domains": list(self.fit_domains),
            "query_domains": list(self.query_domains),
            "targets": self.targets.tolist(),
            "predictions": self.predictions.tolist(),
            "domain_mae": dict(self.domain_mae),
            "fit_state_sha256": self.fit_state_sha256,
            "state_sha256": self.state_sha256,
        }


@dataclass(frozen=True)
class DimensionScore:
    """All inner OOF evidence for one registered PCA dimension."""

    pca_dimension: int | None
    inner_equal_domain_mae: float
    inner_domain_mae: tuple[tuple[str, float], ...]
    fold_provenance: tuple[FoldProvenance, ...]
    state_sha256: str

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.inner_equal_domain_mae)):
            _raise_validation("dimension score must be finite")
        domain_mae_values = tuple(
            (name, float(value)) for name, value in self.inner_domain_mae
        )
        if any(
            type(name) is not str or not name or not np.isfinite(value)
            for name, value in domain_mae_values
        ):
            _raise_validation("dimension domain MAE must be finite exact values")
        object.__setattr__(self, "inner_domain_mae", domain_mae_values)
        object.__setattr__(self, "fold_provenance", tuple(self.fold_provenance))
        _register_state(self, _dimension_score_state(self))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pca_dimension": self.pca_dimension,
            "inner_equal_domain_mae": self.inner_equal_domain_mae,
            "inner_domain_mae": dict(self.inner_domain_mae),
            "fold_provenance": [record.to_dict() for record in self.fold_provenance],
            "state_sha256": self.state_sha256,
        }


@dataclass(frozen=True)
class CandidateScore:
    candidate: str
    pca_dimension: int | None
    inner_equal_domain_mae: float
    inner_domain_mae: tuple[tuple[str, float], ...]
    fold_provenance: tuple[FoldProvenance, ...]
    dimension_records: tuple[DimensionScore, ...]
    state_sha256: str
    unavailable_pca_dimensions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.candidate) is not str or not self.candidate:
            _raise_validation("candidate score name must be an exact string")
        if not np.isfinite(float(self.inner_equal_domain_mae)):
            _raise_validation("candidate score must be finite")
        domain_mae_values = tuple(
            (name, float(value)) for name, value in self.inner_domain_mae
        )
        if any(
            type(name) is not str or not name or not np.isfinite(value)
            for name, value in domain_mae_values
        ):
            _raise_validation("candidate domain MAE must be finite exact values")
        object.__setattr__(self, "inner_domain_mae", domain_mae_values)
        object.__setattr__(self, "fold_provenance", tuple(self.fold_provenance))
        object.__setattr__(self, "dimension_records", tuple(self.dimension_records))
        unavailable = tuple(
            _validate_pca_dimension(value) for value in self.unavailable_pca_dimensions
        )
        if len(set(unavailable)) != len(unavailable):
            _raise_validation(
                "candidate score has duplicate unavailable PCA dimensions"
            )
        object.__setattr__(self, "unavailable_pca_dimensions", unavailable)
        _register_state(self, _candidate_score_state(self))

    @property
    def pca_records(self) -> tuple[DimensionScore, ...]:
        return self.dimension_records

    @property
    def score(self) -> float:
        return self.inner_equal_domain_mae

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "pca_dimension": self.pca_dimension,
            "inner_equal_domain_mae": self.inner_equal_domain_mae,
            "inner_domain_mae": dict(self.inner_domain_mae),
            "fold_provenance": [record.to_dict() for record in self.fold_provenance],
            "dimension_records": [
                record.to_dict() for record in self.dimension_records
            ],
            "unavailable_pca_dimensions": list(self.unavailable_pca_dimensions),
            "state_sha256": self.state_sha256,
        }


@dataclass(frozen=True)
class CandidateSelection:
    selected_name: str
    selected_pca_dimension: int | None
    inner_equal_domain_mae: float
    candidate_order: tuple[str, ...]
    candidate_scores: Mapping[str, CandidateScore]
    fold_provenance: tuple[FoldProvenance, ...]
    state_sha256: str
    outer_train_indices: tuple[int, ...]
    outer_test_ids: tuple[str, ...]
    outer_test_domains: tuple[str, ...]
    bundle_state_sha256: str
    response_state_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_order", tuple(self.candidate_order))
        object.__setattr__(
            self,
            "candidate_scores",
            MappingProxyType(dict(self.candidate_scores)),
        )
        object.__setattr__(self, "fold_provenance", tuple(self.fold_provenance))
        object.__setattr__(
            self,
            "outer_train_indices",
            tuple(int(value) for value in self.outer_train_indices),
        )
        test_ids = tuple(self.outer_test_ids)
        test_domains = tuple(self.outer_test_domains)
        if any(
            type(value) is not str or not value for value in test_ids + test_domains
        ):
            _raise_validation("outer test IDs and domains must be exact strings")
        if not test_ids or len(test_ids) != len(test_domains):
            _raise_validation("outer test IDs and domains must be aligned")
        object.__setattr__(self, "outer_test_ids", test_ids)
        object.__setattr__(self, "outer_test_domains", test_domains)
        if (
            not isinstance(self.bundle_state_sha256, str)
            or len(self.bundle_state_sha256) != 64
        ):
            _raise_validation("selection bundle authority digest is invalid")
        if (
            not isinstance(self.response_state_sha256, str)
            or len(self.response_state_sha256) != 64
        ):
            _raise_validation("selection response authority digest is invalid")
        if not np.isfinite(float(self.inner_equal_domain_mae)):
            _raise_validation("selection MAE must be finite")
        _register_state(self, _selection_state(self))

    @property
    def selected_candidate(self) -> str:
        return self.selected_name

    @property
    def score(self) -> float:
        return self.inner_equal_domain_mae

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_name": self.selected_name,
            "selected_pca_dimension": self.selected_pca_dimension,
            "inner_equal_domain_mae": self.inner_equal_domain_mae,
            "candidate_order": list(self.candidate_order),
            "candidate_scores": {
                name: score.to_dict() for name, score in self.candidate_scores.items()
            },
            "fold_provenance": [record.to_dict() for record in self.fold_provenance],
            "outer_train_indices": list(self.outer_train_indices),
            "outer_test_ids": list(self.outer_test_ids),
            "outer_test_domains": list(self.outer_test_domains),
            "bundle_state_sha256": self.bundle_state_sha256,
            "response_state_sha256": self.response_state_sha256,
            "state_sha256": self.state_sha256,
        }


def _candidate_components(
    bundle: FeatureBundle,
    candidate: str,
    indices: np.ndarray,
    frozen: np.ndarray | None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    a = np.column_stack((bundle.metadata[indices], bundle.surface_stats[indices]))
    if candidate == "A_surface":
        return (
            a,
            _METADATA_FEATURE_NAMES + _SURFACE_FEATURE_NAMES,
            ("1",) * A_FEATURE_COUNT,
        )
    if candidate == "B_scalar":
        matrix = np.column_stack((a, bundle.scalar_internal[indices]))
        return (
            matrix,
            _METADATA_FEATURE_NAMES + _SURFACE_FEATURE_NAMES + SCALAR_FEATURE_NAMES,
            (("1",) * A_FEATURE_COUNT + SCALAR_FEATURE_UNITS),
        )
    if candidate == "B_morph":
        matrix = np.column_stack((a, bundle.morphology[indices]))
        return (
            matrix,
            _METADATA_FEATURE_NAMES + _SURFACE_FEATURE_NAMES + _MORPHOLOGY_NAMES,
            (("1",) * A_FEATURE_COUNT + _MORPHOLOGY_UNITS),
        )
    if candidate == "B_frozen":
        if frozen is None:
            _raise_validation("B_frozen requires a fold-local PCA transform")
        names = _pca_names(frozen.shape[1])
        matrix = np.column_stack((a, frozen))
        return (
            matrix,
            _METADATA_FEATURE_NAMES + _SURFACE_FEATURE_NAMES + names,
            (("1",) * A_FEATURE_COUNT + ("1",) * frozen.shape[1]),
        )
    if candidate == "B_combined":
        if frozen is None:
            _raise_validation("B_combined requires a fold-local PCA transform")
        names = _pca_names(frozen.shape[1])
        matrix = np.column_stack((a, bundle.morphology[indices], frozen))
        return (
            matrix,
            _METADATA_FEATURE_NAMES
            + _SURFACE_FEATURE_NAMES
            + _MORPHOLOGY_NAMES
            + names,
            (("1",) * A_FEATURE_COUNT + _MORPHOLOGY_UNITS + ("1",) * frozen.shape[1]),
        )
    if candidate == "I_morph":
        return (
            np.column_stack((bundle.metadata[indices], bundle.morphology[indices])),
            _METADATA_FEATURE_NAMES + _MORPHOLOGY_NAMES,
            (("1",) * METADATA_FEATURE_COUNT + _MORPHOLOGY_UNITS),
        )
    if candidate == "I_frozen":
        if frozen is None:
            _raise_validation("I_frozen requires a fold-local PCA transform")
        return (
            np.column_stack((bundle.metadata[indices], frozen)),
            _METADATA_FEATURE_NAMES + _pca_names(frozen.shape[1]),
            (("1",) * METADATA_FEATURE_COUNT + ("1",) * frozen.shape[1]),
        )
    if candidate == "I_combined":
        if frozen is None:
            _raise_validation("I_combined requires a fold-local PCA transform")
        names = _pca_names(frozen.shape[1])
        return (
            np.column_stack(
                (bundle.metadata[indices], bundle.morphology[indices], frozen)
            ),
            _METADATA_FEATURE_NAMES + _MORPHOLOGY_NAMES + names,
            (
                ("1",) * METADATA_FEATURE_COUNT
                + _MORPHOLOGY_UNITS
                + ("1",) * frozen.shape[1]
            ),
        )
    _raise_validation(f"candidate is not registered: {candidate}")


def _candidate_matrix(
    bundle: FeatureBundle,
    candidate: str,
    indices: np.ndarray,
    *,
    pca_model: FoldLocalPCA | _TestOnlyPCA | None,
) -> np.ndarray:
    frozen = None
    if candidate in {"B_frozen", "B_combined", "I_frozen", "I_combined"}:
        if pca_model is None:
            _raise_validation("frozen candidate requires a fold-local PCA model")
        _validate_pca_model(pca_model, bundle=bundle)
        frozen = _transform_pca_model(pca_model, bundle.frozen_embedding[indices])
    matrix, _names, _units = _candidate_components(bundle, candidate, indices, frozen)
    return _as_matrix(matrix, f"{candidate} matrix", allow_nan=True)


def _fit_inner_fold(
    bundle: FeatureBundle,
    target: np.ndarray,
    candidate: str,
    fit_idx: np.ndarray,
    query_idx: np.ndarray,
    *,
    pca_dimension: int | None,
    fold_index: int,
    inner_train_authority: object | None = None,
    parent_outer_authority: object | None = None,
    heldout_domain: str | None = None,
    inner_query_domain: str | None = None,
) -> tuple[np.ndarray, FoldProvenance]:
    pca_model = None
    if candidate in {"B_frozen", "B_combined", "I_frozen", "I_combined"}:
        if pca_dimension is None:
            _raise_validation("frozen candidate requires a PCA dimension")
        fit_sample_ids = tuple(bundle.sample_ids[index] for index in fit_idx)
        fit_domain_ids = tuple(bundle.domain_ids[index] for index in fit_idx)
        registration_training_authority: object | None = None
        registration_parent_authority: object | None = None
        registration_embedding_digest = _training_embedding_digest(
            bundle, fit_sample_ids, fit_domain_ids
        )
        if bundle.source_state_sha256 == "test-only":
            if any(
                authority is not None
                for authority in (inner_train_authority, parent_outer_authority)
            ) or heldout_domain is not None:
                _raise_validation(
                    "test-only inner PCA cannot use production data authorities"
                )
            pca_model = _fit_test_only_pca(
                bundle.frozen_embedding[fit_idx],
                n_components=pca_dimension,
                fit_sample_ids=fit_sample_ids,
                fit_domain_ids=fit_domain_ids,
            )
        else:
            authority, parent, resolved_heldout, _inner_state = (
                _resolve_inner_pca_authorities(
                    bundle,
                    fit_idx,
                    fit_sample_ids,
                    fit_domain_ids,
                    inner_train_authority=inner_train_authority,
                    parent_outer_authority=parent_outer_authority,
                    heldout_domain=heldout_domain,
                    inner_query_domain=inner_query_domain,
                )
            )
            pca_model = _fit_production_pca(
                bundle,
                bundle.frozen_embedding[fit_idx],
                n_components=pca_dimension,
                fit_sample_ids=fit_sample_ids,
                fit_domain_ids=fit_domain_ids,
                training_authority=authority,
                heldout_domain=resolved_heldout,
                parent_outer_authority=parent,
                inner_query_domain=inner_query_domain,
            )
            registration_training_authority = authority
            registration_parent_authority = parent
        _register_pca_model(
            pca_model,
            fit_sample_ids,
            fit_domain_ids,
            bundle,
            training_authority=registration_training_authority,
            parent_authority=registration_parent_authority,
            training_embedding_digest=registration_embedding_digest,
        )
    x_fit = _candidate_matrix(bundle, candidate, fit_idx, pca_model=pca_model)
    x_query = _candidate_matrix(bundle, candidate, query_idx, pca_model=pca_model)
    model = fit_fold_ridge(
        x_fit,
        target[fit_idx],
        alpha=RIDGE_ALPHA,
        fit_sample_ids=tuple(bundle.sample_ids[index] for index in fit_idx),
        fit_domain_ids=tuple(bundle.domain_ids[index] for index in fit_idx),
    )
    prediction = model.predict(x_query)
    pca_hash = (
        ""
        if pca_model is None
        else _state_hash(
            pca_model.mean_,
            pca_model.components_,
            pca_model.explained_variance_,
            pca_model.singular_values_,
            pca_model.n_components_,
            pca_model.n_features_in_,
            pca_model.n_samples_in_,
            pca_model.fit_sample_ids,
            pca_model.fit_domain_ids,
        )
    )
    fit_state = _state_hash(candidate, pca_dimension, pca_hash, model.state_sha256)
    query_domains = tuple(bundle.domain_ids[index] for index in query_idx)
    targets = _readonly_array(target[query_idx])
    predictions = _readonly_array(prediction)
    per_domain = tuple(domain_mae(targets, predictions, query_domains).items())
    state = _state_hash(
        candidate,
        pca_dimension,
        fold_index,
        tuple(int(index) for index in fit_idx),
        tuple(int(index) for index in query_idx),
        tuple(bundle.sample_ids[index] for index in fit_idx),
        tuple(bundle.sample_ids[index] for index in query_idx),
        tuple(bundle.domain_ids[index] for index in fit_idx),
        query_domains,
        targets,
        predictions,
        per_domain,
        fit_state,
    )
    provenance = FoldProvenance(
        candidate=candidate,
        pca_dimension=pca_dimension,
        fold_index=fold_index,
        fit_indices=tuple(int(index) for index in fit_idx),
        query_indices=tuple(int(index) for index in query_idx),
        fit_sample_ids=tuple(bundle.sample_ids[index] for index in fit_idx),
        query_sample_ids=tuple(bundle.sample_ids[index] for index in query_idx),
        fit_domains=tuple(bundle.domain_ids[index] for index in fit_idx),
        query_domains=query_domains,
        targets=targets,
        predictions=predictions,
        domain_mae=per_domain,
        fit_state_sha256=fit_state,
        state_sha256=state,
    )
    return prediction, provenance


def _choose_earliest(
    scores: Mapping[int, float], order: Sequence[int]
) -> tuple[int, float]:
    if not scores:
        _raise_validation("no finite candidate scores were supplied")
    selected = None
    selected_score = None
    registered_order = tuple(
        dimension for dimension in PCA_DIMENSIONS if dimension in tuple(order)
    )
    for dimension in registered_order:
        if dimension not in scores:
            continue
        score = float(scores[dimension])
        if not np.isfinite(score):
            _raise_validation("candidate scores must be finite")
        if selected is None or score < float(selected_score) - PCA_TIE_TOLERANCE:
            selected = int(dimension)
            selected_score = score
    if selected is None or selected_score is None:
        _raise_validation("candidate scores do not contain a registered dimension")
    return selected, float(selected_score)


def select_pca_dimension(
    validation_scores: Mapping[int, float],
    *,
    candidate_dimensions: Iterable[int] = PCA_DIMENSIONS,
) -> int:
    """Select a registered PCA dimension with fixed-order near-tie handling."""

    requested = tuple(_validate_pca_dimension(value) for value in candidate_dimensions)
    if requested != PCA_DIMENSIONS:
        _raise_validation(
            "candidate_dimensions must be exactly (8, 16, 32) in registered order"
        )
    try:
        items = tuple(validation_scores.items())
    except (AttributeError, TypeError, ValueError) as error:
        raise FeatureValidationError(
            "PCA selection scores must be a mapping"
        ) from error
    if len(items) != len(PCA_DIMENSIONS):
        _raise_validation("PCA selection scores must contain exactly (8, 16, 32)")
    scores: dict[int, float] = {}
    for key, value in items:
        _validate_pca_dimension(key)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            _raise_validation("PCA selection scores must be finite numeric values")
        score = float(value)
        if not np.isfinite(score):
            _raise_validation("PCA selection scores must be finite")
        scores[int(key)] = score
    if set(scores) != set(PCA_DIMENSIONS):
        _raise_validation("PCA selection scores must contain exactly (8, 16, 32)")
    return _choose_earliest(scores, PCA_DIMENSIONS)[0]


def _evaluate_candidate(
    bundle: FeatureBundle,
    target: np.ndarray,
    candidate: str,
    train_idx: np.ndarray,
    *,
    pca_dimensions: tuple[int, ...],
    inner_train_authority: object | None = None,
    parent_outer_authority: object | None = None,
    heldout_domain: str | None = None,
    inner_query_domain: str | None = None,
) -> CandidateScore:
    uses_frozen = candidate in {
        "B_frozen",
        "B_combined",
        "I_frozen",
        "I_combined",
    }
    if uses_frozen and pca_dimensions != PCA_DIMENSIONS:
        _raise_validation(
            "frozen candidate selection requires all PCA dimensions (8, 16, 32)"
        )
    dimensions = pca_dimensions if uses_frozen else (None,)
    domains = tuple(sorted({bundle.domain_ids[index] for index in train_idx}))
    if len(domains) != 5:
        _raise_validation("inner selection requires exactly five training domains")
    dimension_records: list[DimensionScore] = []
    unavailable_dimensions: list[int] = []
    for dimension in dimensions:
        predictions: list[float] = []
        targets: list[float] = []
        prediction_domains: list[str] = []
        records: list[FoldProvenance] = []
        try:
            for fold_index, domain in enumerate(domains):
                query_idx = train_idx[
                    np.asarray(
                        [bundle.domain_ids[index] == domain for index in train_idx]
                    )
                ]
                fit_idx = train_idx[
                    np.asarray(
                        [bundle.domain_ids[index] != domain for index in train_idx]
                    )
                ]
                if fit_idx.size == 0 or query_idx.size == 0:
                    _raise_validation(
                        "inner fold must have non-empty fit and query rows"
                    )
                prediction, record = _fit_inner_fold(
                    bundle,
                    target,
                    candidate,
                    fit_idx,
                    query_idx,
                    pca_dimension=dimension,
                    fold_index=fold_index,
                    inner_train_authority=inner_train_authority,
                    parent_outer_authority=parent_outer_authority,
                    heldout_domain=heldout_domain,
                    inner_query_domain=domain,
                )
                predictions.extend(float(value) for value in prediction)
                targets.extend(float(value) for value in target[query_idx])
                prediction_domains.extend(
                    bundle.domain_ids[index] for index in query_idx
                )
                records.append(record)
        except _PcaRankUnavailable:
            if dimension is None:
                raise
            unavailable_dimensions.append(dimension)
            continue
        score = equal_domain_mae(
            np.asarray(targets), np.asarray(predictions), prediction_domains
        )
        per_domain = domain_mae(
            np.asarray(targets), np.asarray(predictions), prediction_domains
        )
        dimension_domain_mae = tuple(per_domain.items())
        dimension_state = _state_hash(
            "dimension",
            dimension,
            score,
            dimension_domain_mae,
            tuple(record.state_sha256 for record in records),
        )
        dimension_records.append(
            DimensionScore(
                pca_dimension=dimension,
                inner_equal_domain_mae=float(score),
                inner_domain_mae=dimension_domain_mae,
                fold_provenance=tuple(records),
                state_sha256=dimension_state,
            )
        )
    if uses_frozen and not dimension_records:
        _raise_validation("no registered PCA dimensions are representable")
    dimension_scores = {
        record.pca_dimension: record.inner_equal_domain_mae
        for record in dimension_records
    }
    selected_dimension, selected_score = (
        _choose_earliest(
            {
                int(key): value
                for key, value in dimension_scores.items()
                if key is not None
            },
            PCA_DIMENSIONS,
        )
        if uses_frozen
        else (None, dimension_scores[None])
    )
    selected_record = next(
        record
        for record in dimension_records
        if record.pca_dimension == selected_dimension
    )
    state = _state_hash(
        "candidate-score",
        candidate,
        selected_dimension,
        selected_score,
        selected_record.inner_domain_mae,
        tuple(
            (
                record.pca_dimension,
                record.inner_equal_domain_mae,
                record.state_sha256,
            )
            for record in dimension_records
        ),
        tuple(unavailable_dimensions),
    )
    return CandidateScore(
        candidate=candidate,
        pca_dimension=selected_dimension,
        inner_equal_domain_mae=float(selected_score),
        inner_domain_mae=selected_record.inner_domain_mae,
        fold_provenance=selected_record.fold_provenance,
        dimension_records=tuple(dimension_records),
        state_sha256=state,
        unavailable_pca_dimensions=tuple(unavailable_dimensions),
    )


def _validate_outer_contract(
    bundle: FeatureBundle,
    *,
    outer_train_indices: Sequence[int] | np.ndarray | None,
    outer_test_ids: Sequence[str] | None,
    outer_test_domains: Sequence[str] | None,
    train_indices: Sequence[int] | np.ndarray | None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if train_indices is not None:
        _raise_validation(
            "train_indices is not accepted; outer_train_indices is required"
        )
    if (
        outer_train_indices is None
        or outer_test_ids is None
        or outer_test_domains is None
    ):
        _raise_validation(
            "outer_train_indices, outer_test_ids, and outer_test_domains are required"
        )
    train_idx = np.sort(
        _indices(outer_train_indices, length=bundle.n_rows, name="outer_train_indices")
    )
    train_domains = {bundle.domain_ids[index] for index in train_idx}
    if len(train_domains) != 5:
        _raise_validation("outer training pool must contain exactly five domains")
    if isinstance(outer_test_ids, (str, bytes)) or isinstance(
        outer_test_domains, (str, bytes)
    ):
        _raise_validation("outer test IDs and domains must be sequences of strings")
    test_ids = tuple(outer_test_ids)
    test_domains = tuple(outer_test_domains)
    if any(not isinstance(value, str) or not value for value in test_ids):
        _raise_validation("outer test IDs must be non-empty strings")
    if any(not isinstance(value, str) or not value for value in test_domains):
        _raise_validation("outer test domains must be non-empty strings")
    if not test_ids or len(test_ids) != len(test_domains):
        _raise_validation("outer test IDs and domains must be non-empty and aligned")
    if len(set(test_ids)) != len(test_ids):
        _raise_validation("outer test IDs must be unique")
    if any(not value for value in test_ids + test_domains):
        _raise_validation("outer test IDs and domains cannot be empty")
    if len(set(test_domains)) != 1:
        _raise_validation("outer test pool must contain exactly one domain")
    if test_domains[0] in train_domains:
        _raise_validation("outer test domain must not occur in outer training domains")
    sample_ids = set(bundle.sample_ids)
    train_ids = {bundle.sample_ids[index] for index in train_idx}
    if not set(test_ids).issubset(sample_ids):
        _raise_validation("outer test IDs must be registered sample IDs")
    if set(test_ids) & train_ids:
        _raise_validation("outer train and test pools must be disjoint")
    if set(test_ids) != sample_ids - train_ids:
        _raise_validation("outer test IDs must cover the complement of outer training")
    sample_to_index = {
        sample_id: index for index, sample_id in enumerate(bundle.sample_ids)
    }
    outer_test_indices = tuple(sample_to_index[sample_id] for sample_id in test_ids)
    actual_test_domains = tuple(
        bundle.domain_ids[index] for index in outer_test_indices
    )
    actual_outer_domains = set(actual_test_domains)
    if len(actual_outer_domains) != 1:
        _raise_validation(
            "outer test pool must contain exactly one actual outer domain"
        )
    actual_outer_domain = actual_test_domains[0]
    if actual_outer_domain in train_domains:
        _raise_validation("actual outer domain occurs in outer training")
    if actual_test_domains != test_domains:
        _raise_validation(
            "actual outer test domains do not match the bundle outer domains"
        )
    canonical_test_indices = tuple(sorted(outer_test_indices))
    return (
        train_idx,
        tuple(bundle.sample_ids[index] for index in canonical_test_indices),
        tuple(bundle.domain_ids[index] for index in canonical_test_indices),
    )


def _validate_selector_pca_authority(
    bundle: FeatureBundle,
    train_idx: np.ndarray,
    *,
    outer_train_authority: object | None,
    heldout_domain: str | None,
    required: bool,
) -> None:
    if not required:
        return
    if outer_train_authority is None or heldout_domain is None:
        _raise_validation(
            "production PCA selection requires an issued outer-training view and heldout domain"
        )
    fit_sample_ids = tuple(bundle.sample_ids[index] for index in train_idx)
    fit_domain_ids = tuple(bundle.domain_ids[index] for index in train_idx)
    _validate_issued_pca_view(
        outer_train_authority,
        fit_sample_ids,
        fit_domain_ids,
        name="outer-training PCA authority",
        expected_domain_count=5,
    )
    _validate_pca_view_matches_bundle(
        bundle,
        outer_train_authority,
        name="outer-training PCA authority",
    )
    if heldout_domain in set(fit_domain_ids):
        _raise_validation("PCA heldout domain occurs in outer training")


def _select_registered_candidate(
    bundle: FeatureBundle,
    response: ResponseVector,
    *,
    registered_order: tuple[str, ...],
    selection_label: str,
    outer_train_indices: Sequence[int] | np.ndarray | None = None,
    outer_test_ids: Sequence[str] | None = None,
    outer_test_domains: Sequence[str] | None = None,
    train_indices: Sequence[int] | np.ndarray | None = None,
    candidate_order: Sequence[str] = PRIMARY_CANDIDATE_ORDER,
    pca_dimensions: Iterable[int] = PCA_DIMENSIONS,
    outer_train_authority: object | None = None,
    heldout_domain: str | None = None,
    _test_only: bool = False,
) -> CandidateSelection:
    if not isinstance(bundle, FeatureBundle):
        _raise_validation("bundle must be a FeatureBundle")
    if not isinstance(response, ResponseVector):
        _raise_validation("production selector requires a typed ResponseVector")
    if not _test_only:
        _require_production_bundle(bundle)
    validate_feature_bundle(bundle)
    train_idx, outer_test_ids_value, outer_test_domains_value = (
        _validate_outer_contract(
            bundle,
            outer_train_indices=outer_train_indices,
            outer_test_ids=outer_test_ids,
            outer_test_domains=outer_test_domains,
            train_indices=train_indices,
        )
    )
    validate_response_vector(response, bundle)
    target_value = response.values
    order = tuple(candidate_order)
    if not _test_only and order != registered_order:
        _raise_validation(
            f"production {selection_label} requires the complete registered candidate order"
        )
    if any(type(value) is not str or not value for value in order):
        _raise_validation("candidate order must contain exact registered strings")
    if not order or len(set(order)) != len(order):
        _raise_validation("candidate order must contain unique registered candidates")
    if any(value not in registered_order for value in order):
        _raise_validation(f"candidate is not registered for {selection_label}")
    requested_dimensions = tuple(
        _validate_pca_dimension(value) for value in pca_dimensions
    )
    if (
        any(
            value in {"B_frozen", "B_combined", "I_frozen", "I_combined"}
            for value in order
        )
        and requested_dimensions != PCA_DIMENSIONS
    ):
        _raise_validation(
            "PCA dimensions must be the complete registered order (8, 16, 32)"
        )
    if not requested_dimensions and any(
        value in {"B_frozen", "B_combined", "I_frozen", "I_combined"} for value in order
    ):
        _raise_validation("PCA dimensions must contain all registered dimensions")
    requested_dimensions = PCA_DIMENSIONS
    _validate_selector_pca_authority(
        bundle,
        train_idx,
        outer_train_authority=outer_train_authority,
        heldout_domain=heldout_domain,
        required=(
            not _test_only
            and any(
                value in {"B_frozen", "B_combined", "I_frozen", "I_combined"}
                for value in order
            )
        ),
    )
    scores: dict[str, CandidateScore] = {}
    for candidate in order:
        scores[candidate] = _evaluate_candidate(
            bundle,
            target_value,
            candidate,
            train_idx,
            pca_dimensions=requested_dimensions,
            parent_outer_authority=outer_train_authority,
            heldout_domain=heldout_domain,
        )
    selected_name = min(order, key=lambda candidate: registered_order.index(candidate))
    selected_score = scores[selected_name].inner_equal_domain_mae
    for candidate in order:
        if candidate == selected_name:
            continue
        score = scores[candidate].inner_equal_domain_mae
        if score < selected_score - PCA_TIE_TOLERANCE:
            selected_name = candidate
            selected_score = score
    selected = scores[selected_name]
    state = _state_hash(
        selected_name,
        selected.pca_dimension,
        selected_score,
        order,
        tuple(int(value) for value in train_idx),
        outer_test_ids_value,
        outer_test_domains_value,
        _selection_bundle_digest(bundle, train_idx),
        _response_training_digest(response, train_idx),
        {name: scores[name].to_dict() for name in order},
    )
    return CandidateSelection(
        selected_name=selected_name,
        selected_pca_dimension=selected.pca_dimension,
        inner_equal_domain_mae=float(selected_score),
        candidate_order=order,
        candidate_scores=scores,
        fold_provenance=selected.fold_provenance,
        state_sha256=state,
        outer_train_indices=tuple(int(value) for value in train_idx),
        outer_test_ids=outer_test_ids_value,
        outer_test_domains=outer_test_domains_value,
        bundle_state_sha256=_selection_bundle_digest(bundle, train_idx),
        response_state_sha256=_response_training_digest(response, train_idx),
    )


def select_candidate(
    bundle: FeatureBundle,
    response: ResponseVector,
    *,
    outer_train_indices: Sequence[int] | np.ndarray | None = None,
    outer_test_ids: Sequence[str] | None = None,
    outer_test_domains: Sequence[str] | None = None,
    train_indices: Sequence[int] | np.ndarray | None = None,
    candidate_order: Sequence[str] = PRIMARY_CANDIDATE_ORDER,
    pca_dimensions: Iterable[int] = PCA_DIMENSIONS,
    outer_train_authority: object | None = None,
    heldout_domain: str | None = None,
) -> CandidateSelection:
    """Inner-LODO selection for the registered primary field candidates."""
    return _select_registered_candidate(
        bundle,
        response,
        registered_order=_PRIMARY_OUTER_CANDIDATES_ORDER,
        selection_label="B_field_selected",
        outer_train_indices=outer_train_indices,
        outer_test_ids=outer_test_ids,
        outer_test_domains=outer_test_domains,
        train_indices=train_indices,
        candidate_order=candidate_order,
        pca_dimensions=pca_dimensions,
        outer_train_authority=outer_train_authority,
        heldout_domain=heldout_domain,
    )


def select_baseline_candidate(
    bundle: FeatureBundle,
    response: ResponseVector,
    *,
    outer_train_indices: Sequence[int] | np.ndarray | None = None,
    outer_test_ids: Sequence[str] | None = None,
    outer_test_domains: Sequence[str] | None = None,
    train_indices: Sequence[int] | np.ndarray | None = None,
    candidate_order: Sequence[str] = BASELINE_CANDIDATE_ORDER,
    pca_dimensions: Iterable[int] = PCA_DIMENSIONS,
    outer_train_authority: object | None = None,
    heldout_domain: str | None = None,
) -> CandidateSelection:
    """Select the registered surface/scalar baselines outside the B-field roster."""

    return _select_registered_candidate(
        bundle,
        response,
        registered_order=BASELINE_CANDIDATE_ORDER,
        selection_label="baseline_candidate",
        outer_train_indices=outer_train_indices,
        outer_test_ids=outer_test_ids,
        outer_test_domains=outer_test_domains,
        train_indices=train_indices,
        candidate_order=candidate_order,
        pca_dimensions=pca_dimensions,
        outer_train_authority=outer_train_authority,
        heldout_domain=heldout_domain,
    )


def select_candidate_test_only(
    bundle: FeatureBundle,
    response: ResponseVector,
    **kwargs: object,
) -> CandidateSelection:
    return _select_registered_candidate(
        bundle,
        response,
        registered_order=PRIMARY_CANDIDATE_ORDER,
        selection_label="B_field_selected",
        _test_only=True,
        **kwargs,
    )


INTERNAL_CANDIDATE_ORDER = ("I_morph", "I_frozen", "I_combined")


def select_internal_candidate(
    bundle: FeatureBundle,
    response: ResponseVector,
    *,
    outer_train_indices: Sequence[int] | np.ndarray | None = None,
    outer_test_ids: Sequence[str] | None = None,
    outer_test_domains: Sequence[str] | None = None,
    train_indices: Sequence[int] | np.ndarray | None = None,
    candidate_order: Sequence[str] = INTERNAL_CANDIDATE_ORDER,
    pca_dimensions: Iterable[int] = PCA_DIMENSIONS,
    outer_train_authority: object | None = None,
    heldout_domain: str | None = None,
) -> CandidateSelection:
    """Inner-LODO selection for the independent metadata-only candidates."""

    return _select_registered_candidate(
        bundle,
        response,
        registered_order=INTERNAL_CANDIDATE_ORDER,
        selection_label="I_field_selected",
        outer_train_indices=outer_train_indices,
        outer_test_ids=outer_test_ids,
        outer_test_domains=outer_test_domains,
        train_indices=train_indices,
        candidate_order=candidate_order,
        pca_dimensions=pca_dimensions,
        outer_train_authority=outer_train_authority,
        heldout_domain=heldout_domain,
    )


def select_internal_candidate_test_only(
    bundle: FeatureBundle,
    response: ResponseVector,
    **kwargs: object,
) -> CandidateSelection:
    return _select_registered_candidate(
        bundle,
        response,
        registered_order=INTERNAL_CANDIDATE_ORDER,
        selection_label="I_field_selected",
        _test_only=True,
        **kwargs,
    )


def _fold_provenance_state(record: FoldProvenance) -> str:
    return _state_hash(
        record.candidate,
        record.pca_dimension,
        record.fold_index,
        record.fit_indices,
        record.query_indices,
        record.fit_sample_ids,
        record.query_sample_ids,
        record.fit_domains,
        record.query_domains,
        record.targets,
        record.predictions,
        record.domain_mae,
        record.fit_state_sha256,
    )


def _dimension_score_state(record: DimensionScore) -> str:
    return _state_hash(
        "dimension",
        record.pca_dimension,
        record.inner_equal_domain_mae,
        record.inner_domain_mae,
        tuple(item.state_sha256 for item in record.fold_provenance),
    )


def _candidate_score_state(score: CandidateScore) -> str:
    selected_records = tuple(
        record
        for record in score.dimension_records
        if record.pca_dimension == score.pca_dimension
    )
    if len(selected_records) != 1:
        _raise_validation("candidate score has no unique selected dimension")
    selected = selected_records[0]
    return _state_hash(
        "candidate-score",
        score.candidate,
        score.pca_dimension,
        score.inner_equal_domain_mae,
        selected.inner_domain_mae,
        tuple(
            (
                record.pca_dimension,
                record.inner_equal_domain_mae,
                record.state_sha256,
            )
            for record in score.dimension_records
        ),
        score.unavailable_pca_dimensions,
    )


def _selection_state(selection: CandidateSelection) -> str:
    return _state_hash(
        selection.selected_name,
        selection.selected_pca_dimension,
        selection.inner_equal_domain_mae,
        selection.candidate_order,
        selection.outer_train_indices,
        selection.outer_test_ids,
        selection.outer_test_domains,
        selection.bundle_state_sha256,
        selection.response_state_sha256,
        {name: score.to_dict() for name, score in selection.candidate_scores.items()},
    )


def _selection_bundle_digest(
    bundle: FeatureBundle, train_indices: Sequence[int]
) -> str:
    indices = np.asarray(tuple(int(index) for index in train_indices), dtype=np.int64)
    return _state_hash(
        bundle.metadata[indices],
        bundle.surface_stats[indices],
        bundle.scalar_internal[indices],
        bundle.morphology[indices],
        bundle.frozen_embedding[indices],
        tuple(bundle.sample_ids[index] for index in indices),
        tuple(bundle.domain_ids[index] for index in indices),
    )


def _assert_same_domain_mae(
    expected: Sequence[tuple[str, float]], actual: Sequence[tuple[str, float]]
) -> None:
    if tuple(name for name, _value in expected) != tuple(
        name for name, _value in actual
    ) or len(expected) != len(actual):
        _raise_validation("selection state contains inconsistent domain MAE")
    if not all(
        np.isclose(float(expected_value), float(actual_value), rtol=0.0, atol=1e-12)
        for (_name, expected_value), (_other, actual_value) in zip(expected, actual)
    ):
        _raise_validation("selection state contains inconsistent domain MAE")


def _validate_fold_provenance(record: FoldProvenance) -> None:
    _assert_authority(record, _fold_provenance_state(record), name="fold provenance")
    if record.targets.flags.writeable or record.predictions.flags.writeable:
        _raise_validation("selection state arrays must be read-only")
    expected_domain_mae = tuple(
        domain_mae(record.targets, record.predictions, record.query_domains).items()
    )
    _assert_same_domain_mae(expected_domain_mae, record.domain_mae)
    if _fold_provenance_state(record) != record.state_sha256:
        _raise_validation("fold provenance state hash mismatch")


def _target_from_inner_provenance(
    bundle: FeatureBundle,
    score: CandidateScore,
    train_indices: np.ndarray,
) -> np.ndarray:
    if not score.dimension_records:
        _raise_validation("candidate score is missing inner provenance")
    targets = np.full(bundle.n_rows, np.nan, dtype=np.float64)
    seen: set[int] = set()
    train_set = {int(index) for index in train_indices}
    records = score.dimension_records[0].fold_provenance
    expected_domains = tuple(
        sorted({bundle.domain_ids[index] for index in train_indices})
    )
    if len(records) != len(expected_domains):
        _raise_validation("candidate provenance does not cover five inner folds")
    for fold_index, (record, expected_domain) in enumerate(
        zip(records, expected_domains)
    ):
        if record.fold_index != fold_index:
            _raise_validation("candidate provenance fold order is not canonical")
        query = tuple(int(index) for index in record.query_indices)
        fit = tuple(int(index) for index in record.fit_indices)
        if any(index not in train_set for index in query + fit):
            _raise_validation("candidate provenance escapes the outer training pool")
        expected_query = tuple(
            int(index)
            for index in train_indices
            if bundle.domain_ids[index] == expected_domain
        )
        expected_fit = tuple(
            int(index)
            for index in train_indices
            if bundle.domain_ids[index] != expected_domain
        )
        if query != expected_query or fit != expected_fit:
            _raise_validation("candidate provenance inner split does not match bundle")
        if (
            tuple(bundle.sample_ids[index] for index in query)
            != record.query_sample_ids
        ):
            _raise_validation(
                "candidate provenance query sample IDs do not match bundle"
            )
        if tuple(bundle.sample_ids[index] for index in fit) != record.fit_sample_ids:
            _raise_validation("candidate provenance fit sample IDs do not match bundle")
        if tuple(bundle.domain_ids[index] for index in query) != record.query_domains:
            _raise_validation("candidate provenance query domains do not match bundle")
        if tuple(bundle.domain_ids[index] for index in fit) != record.fit_domains:
            _raise_validation("candidate provenance fit domains do not match bundle")
        if len(query) != record.targets.size:
            _raise_validation(
                "candidate provenance targets do not align with query rows"
            )
        for index, target in zip(query, record.targets):
            if index in seen or np.isfinite(targets[index]):
                _raise_validation("candidate provenance query rows overlap")
            targets[index] = float(target)
            seen.add(index)
    if seen != train_set:
        _raise_validation("candidate provenance does not cover outer training rows")
    return targets


def validate_fold_ridge_state(model: FoldRidgeModel) -> bool:
    """Validate immutable Ridge parameters and their fold-local state hash."""

    if not isinstance(model, FoldRidgeModel):
        _raise_validation("ridge state has an invalid type")
    current_preprocessor = _preprocessor_state(model.preprocessor)
    _assert_authority(model.preprocessor, current_preprocessor, name="preprocessor")
    current_model = _state_hash(
        current_preprocessor,
        RIDGE_ALPHA,
        model.coef_,
        model.intercept_,
        model.fit_sample_ids,
        model.fit_domain_ids,
        model.preprocessor.n_features_in_,
    )
    _assert_authority(model, current_model, name="ridge model")
    arrays = (
        model.preprocessor.imputer_statistics_,
        model.preprocessor.mean_,
        model.preprocessor.scale_,
        model.coef_,
    )
    if any(array.flags.writeable for array in arrays):
        _raise_validation("ridge state arrays must be read-only")
    preprocessor_state = current_preprocessor
    if preprocessor_state != model.preprocessor.state_sha256:
        _raise_validation("ridge preprocessor state hash mismatch")
    if model.alpha != RIDGE_ALPHA:
        _raise_validation("ridge state alpha is not registered")
    expected_model_state = _state_hash(
        model.preprocessor.state_sha256,
        RIDGE_ALPHA,
        model.coef_,
        model.intercept_,
        model.fit_sample_ids,
        model.fit_domain_ids,
        model.preprocessor.n_features_in_,
    )
    if expected_model_state != model.state_sha256:
        _raise_validation("ridge model state hash mismatch")
    return True


def serialize_fold_ridge(model: FoldRidgeModel) -> bytes:
    validate_fold_ridge_state(model)
    return _json_bytes(
        {
            "version": 1,
            "alpha": model.alpha,
            "coef": _encode_array(model.coef_),
            "intercept": model.intercept_,
            "fit_sample_ids": list(model.fit_sample_ids),
            "fit_domain_ids": list(model.fit_domain_ids),
            "preprocessor": {
                "imputer_statistics": _encode_array(
                    model.preprocessor.imputer_statistics_
                ),
                "mean": _encode_array(model.preprocessor.mean_),
                "scale": _encode_array(model.preprocessor.scale_),
                "fit_sample_ids": list(model.preprocessor.fit_sample_ids),
                "fit_domain_ids": list(model.preprocessor.fit_domain_ids),
                "n_features_in": model.preprocessor.n_features_in_,
            },
            "state_sha256": model.state_sha256,
        }
    )


def deserialize_fold_ridge(
    value: bytes,
    *,
    authority: FoldRidgeModel | None = None,
    _budget: _ArrayDecodeBudget | None = None,
) -> FoldRidgeModel:
    try:
        if authority is None:
            _raise_validation(
                "serialized Ridge state requires an externally issued authority"
            )
        validate_fold_ridge_state(authority)
        payload = _load_json_bytes(value, name="Ridge state")
        _require_json_schema(
            payload,
            name="Ridge state",
            schema={
                "version": int,
                "alpha": float,
                "coef": str,
                "intercept": float,
                "fit_sample_ids": list,
                "fit_domain_ids": list,
                "preprocessor": Mapping,
                "state_sha256": str,
            },
        )
        if payload["version"] != 1:
            _raise_validation("serialized Ridge state version is unsupported")
        pre = payload["preprocessor"]
        if not isinstance(pre, Mapping):
            _raise_validation("serialized Ridge preprocessor is invalid")
        _require_json_schema(
            pre,
            name="Ridge preprocessor",
            schema={
                "imputer_statistics": str,
                "mean": str,
                "scale": str,
                "fit_sample_ids": list,
                "fit_domain_ids": list,
                "n_features_in": int,
            },
        )
        pre_sample_ids = _require_string_list(
            pre["fit_sample_ids"], name="Ridge preprocessor fit_sample_ids"
        )
        pre_domain_ids = _require_string_list(
            pre["fit_domain_ids"], name="Ridge preprocessor fit_domain_ids"
        )
        fit_sample_ids = _require_string_list(
            payload["fit_sample_ids"], name="Ridge fit_sample_ids"
        )
        fit_domain_ids = _require_string_list(
            payload["fit_domain_ids"], name="Ridge fit_domain_ids"
        )
        n_features = pre["n_features_in"]
        if n_features <= 0 or n_features > _MAX_SERIALIZED_ARRAY_ELEMENTS:
            _raise_validation("serialized Ridge feature count is invalid")
        if n_features != authority.preprocessor.n_features_in_:
            _raise_validation("serialized Ridge feature count does not match authority")
        expected_shape = (n_features,)
        budget = _budget if _budget is not None else _ArrayDecodeBudget()
        expected_pre = authority.preprocessor
        decoded_imputer = _decode_array(
            pre["imputer_statistics"],
            name="imputer statistics",
            budget=budget,
            expected_dtype=np.dtype(np.float64),
            expected_shape=expected_shape,
        )
        decoded_mean = _decode_array(
            pre["mean"],
            name="preprocessor mean",
            budget=budget,
            expected_dtype=np.dtype(np.float64),
            expected_shape=expected_shape,
        )
        decoded_scale = _decode_array(
            pre["scale"],
            name="preprocessor scale",
            budget=budget,
            expected_dtype=np.dtype(np.float64),
            expected_shape=expected_shape,
        )
        decoded_coef = _decode_array(
            payload["coef"],
            name="Ridge coefficients",
            budget=budget,
            expected_dtype=np.dtype(np.float64),
            expected_shape=expected_shape,
        )
        if not np.array_equal(decoded_imputer, expected_pre.imputer_statistics_):
            _raise_validation("serialized Ridge imputer does not match authority")
        if not np.array_equal(decoded_mean, expected_pre.mean_):
            _raise_validation("serialized Ridge mean does not match authority")
        if not np.array_equal(decoded_scale, expected_pre.scale_):
            _raise_validation("serialized Ridge scale does not match authority")
        if not np.array_equal(decoded_coef, authority.coef_):
            _raise_validation("serialized Ridge coefficients do not match authority")
        if tuple(pre_sample_ids) != expected_pre.fit_sample_ids:
            _raise_validation("serialized Ridge preprocessor IDs do not match authority")
        if tuple(pre_domain_ids) != expected_pre.fit_domain_ids:
            _raise_validation("serialized Ridge preprocessor domains do not match authority")
        if tuple(fit_sample_ids) != authority.fit_sample_ids:
            _raise_validation("serialized Ridge fit IDs do not match authority")
        if tuple(fit_domain_ids) != authority.fit_domain_ids:
            _raise_validation("serialized Ridge fit domains do not match authority")
        if payload["alpha"] != authority.alpha:
            _raise_validation("serialized Ridge alpha does not match authority")
        if payload["intercept"] != authority.intercept_:
            _raise_validation("serialized Ridge intercept does not match authority")
        if payload["state_sha256"] != authority.state_sha256:
            _raise_validation("serialized Ridge state does not match authority")
        return authority
    except FeatureValidationError:
        raise
    except Exception as error:
        raise FeatureValidationError("serialized Ridge state payload is invalid") from error


def validate_candidate_selection(
    selection: CandidateSelection,
    bundle: FeatureBundle,
    *,
    response: ResponseVector,
    outer_train_indices: Sequence[int] | np.ndarray,
    outer_test_ids: Sequence[str],
    outer_test_domains: Sequence[str],
    outer_train_authority: object | None = None,
    heldout_domain: str | None = None,
) -> bool:
    """Recompute all stored inner evidence and reject any state mutation."""

    if not isinstance(selection, CandidateSelection):
        _raise_validation("selection state has an invalid type")
    validate_feature_bundle(bundle)
    validate_response_vector(response, bundle)
    expected_train, expected_test_ids, expected_test_domains = _validate_outer_contract(
        bundle,
        outer_train_indices=outer_train_indices,
        outer_test_ids=outer_test_ids,
        outer_test_domains=outer_test_domains,
        train_indices=None,
    )
    if selection.outer_train_indices != tuple(int(value) for value in expected_train):
        _raise_validation("selection outer training identity does not match contract")
    if selection.outer_test_ids != expected_test_ids:
        _raise_validation("selection outer test identity does not match contract")
    if selection.outer_test_domains != expected_test_domains:
        _raise_validation("selection outer test domains do not match contract")
    bundle_digest = _selection_bundle_digest(bundle, expected_train)
    if selection.bundle_state_sha256 != bundle_digest:
        _raise_validation("selection is bound to a different feature bundle")
    if selection.response_state_sha256 != _response_training_digest(
        response, expected_train
    ):
        _raise_validation("selection is bound to a different response vector")
    _validate_selector_pca_authority(
        bundle,
        expected_train,
        outer_train_authority=outer_train_authority,
        heldout_domain=heldout_domain,
        required=(
            any(
                candidate in {"B_frozen", "B_combined", "I_frozen", "I_combined"}
                for candidate in selection.candidate_order
            )
            and _bundle_authority_record(bundle)[2]
        ),
    )
    _assert_authority(selection, _selection_state(selection), name="selection")
    if (
        not selection.outer_train_indices
        or not selection.outer_test_ids
        or len(selection.outer_test_ids) != len(selection.outer_test_domains)
        or len(set(selection.outer_test_domains)) != 1
    ):
        _raise_validation("selection state contains an invalid outer contract")
    if tuple(selection.candidate_order) != tuple(selection.candidate_scores):
        _raise_validation("selection state candidate order does not match scores")
    if any(
        not isinstance(score, CandidateScore)
        for score in selection.candidate_scores.values()
    ):
        _raise_validation("selection state contains an invalid candidate score")
    for key, score in selection.candidate_scores.items():
        if key != score.candidate:
            _raise_validation("candidate score mapping key does not match candidate")
        _assert_authority(score, _candidate_score_state(score), name="candidate score")
        if not score.dimension_records:
            _raise_validation("candidate score is missing PCA dimension records")
        if score.candidate in {"B_frozen", "B_combined", "I_frozen", "I_combined"}:
            expected_dimensions = tuple(
                dimension
                for dimension in PCA_DIMENSIONS
                if dimension not in score.unavailable_pca_dimensions
            )
            if not expected_dimensions:
                _raise_validation(
                    "candidate score has no available registered PCA dimension"
                )
        else:
            if score.unavailable_pca_dimensions:
                _raise_validation(
                    "non-frozen candidate cannot have unavailable PCA dimensions"
                )
            expected_dimensions = (None,)
        if tuple(record.pca_dimension for record in score.dimension_records) != tuple(
            expected_dimensions
        ):
            _raise_validation("candidate score is missing a registered PCA dimension")
        for dimension_record in score.dimension_records:
            _assert_authority(
                dimension_record,
                _dimension_score_state(dimension_record),
                name="dimension score",
            )
            if not dimension_record.fold_provenance:
                _raise_validation("PCA dimension record is missing OOF folds")
            for provenance in dimension_record.fold_provenance:
                _validate_fold_provenance(provenance)
            targets = np.concatenate(
                [item.targets for item in dimension_record.fold_provenance]
            )
            predictions = np.concatenate(
                [item.predictions for item in dimension_record.fold_provenance]
            )
            domains = tuple(
                domain
                for item in dimension_record.fold_provenance
                for domain in item.query_domains
            )
            score_value = equal_domain_mae(targets, predictions, domains)
            domain_values = tuple(domain_mae(targets, predictions, domains).items())
            if not np.isclose(
                score_value,
                dimension_record.inner_equal_domain_mae,
                rtol=0.0,
                atol=1e-12,
            ):
                _raise_validation("selection state contains an inconsistent OOF score")
            _assert_same_domain_mae(domain_values, dimension_record.inner_domain_mae)
            if (
                _dimension_score_state(dimension_record)
                != dimension_record.state_sha256
            ):
                _raise_validation("PCA dimension state hash mismatch")
        dimensions = {
            record.pca_dimension: record.inner_equal_domain_mae
            for record in score.dimension_records
        }
        if score.pca_dimension is None:
            selected_records = tuple(
                record
                for record in score.dimension_records
                if record.pca_dimension is None
            )
            if len(selected_records) != 1:
                _raise_validation("candidate score has no unique selected dimension")
            selected_dimension_record = selected_records[0]
        else:
            selected_dimension, _selected_value = _choose_earliest(
                {
                    int(key): value
                    for key, value in dimensions.items()
                    if key is not None
                },
                PCA_DIMENSIONS,
            )
            selected_records = tuple(
                record
                for record in score.dimension_records
                if record.pca_dimension == selected_dimension
            )
            if len(selected_records) != 1:
                _raise_validation("candidate score has no unique selected dimension")
            selected_dimension_record = selected_records[0]
        if score.pca_dimension != selected_dimension_record.pca_dimension:
            _raise_validation("candidate score selected dimension mismatch")
        if not np.isclose(
            score.inner_equal_domain_mae,
            selected_dimension_record.inner_equal_domain_mae,
            rtol=0.0,
            atol=1e-12,
        ):
            _raise_validation("candidate score selected MAE mismatch")
        _assert_same_domain_mae(
            score.inner_domain_mae, selected_dimension_record.inner_domain_mae
        )
        if tuple(item.state_sha256 for item in score.fold_provenance) != tuple(
            item.state_sha256 for item in selected_dimension_record.fold_provenance
        ):
            _raise_validation("candidate score selected folds mismatch")
        if _candidate_score_state(score) != score.state_sha256:
            _raise_validation("candidate score state hash mismatch")
        target_from_bundle = np.asarray(response.values, dtype=np.float64)
        for record in score.dimension_records[0].fold_provenance:
            if tuple(record.query_indices):
                observed = target_from_bundle[list(record.query_indices)]
                if not np.array_equal(observed, record.targets):
                    _raise_validation(
                        "candidate OOF targets do not match authoritative response"
                    )
        recomputed = _evaluate_candidate(
            bundle,
            target_from_bundle,
            score.candidate,
            expected_train,
            pca_dimensions=PCA_DIMENSIONS,
            parent_outer_authority=outer_train_authority,
            heldout_domain=heldout_domain,
        )
        if recomputed.state_sha256 != score.state_sha256:
            _raise_validation(
                "candidate OOF provenance does not match recomputed bundle evidence"
            )
    selection_alias = selection.selected_name in {
        "B_field_selected",
        "I_field_selected",
    }
    if selection.selected_name not in selection.candidate_scores and not selection_alias:
        _raise_validation("selection state selected candidate is missing")
    if set(selection.candidate_order).issubset(_PRIMARY_OUTER_CANDIDATES):
        registered_order = _PRIMARY_OUTER_CANDIDATES_ORDER
    elif set(selection.candidate_order).issubset(_BASELINE_OUTER_CANDIDATES):
        registered_order = BASELINE_CANDIDATE_ORDER
    elif set(selection.candidate_order).issubset(INTERNAL_CANDIDATE_ORDER):
        registered_order = INTERNAL_CANDIDATE_ORDER
    else:
        _raise_validation("selection state contains an unregistered candidate")
    selected_name = min(
        selection.candidate_order,
        key=lambda candidate: registered_order.index(candidate),
    )
    selected_score = selection.candidate_scores[selected_name].inner_equal_domain_mae
    for candidate in selection.candidate_order:
        score = selection.candidate_scores[candidate].inner_equal_domain_mae
        if score < selected_score - PCA_TIE_TOLERANCE:
            selected_name = candidate
            selected_score = score
    selected = selection.candidate_scores[selected_name]
    if not selection_alias and selection.selected_name != selected_name:
        _raise_validation("selection state selected candidate mismatch")
    if selection.selected_pca_dimension != selected.pca_dimension:
        _raise_validation("selection state selected PCA mismatch")
    if not np.isclose(
        selection.inner_equal_domain_mae,
        selected_score,
        rtol=0.0,
        atol=1e-12,
    ):
        _raise_validation("selection state selected MAE mismatch")
    if tuple(item.state_sha256 for item in selection.fold_provenance) != tuple(
        item.state_sha256 for item in selected.fold_provenance
    ):
        _raise_validation("selection state selected folds mismatch")
    if _selection_state(selection) != selection.state_sha256:
        _raise_validation("selection state hash mismatch")
    return True


def _outer_candidate_and_label(
    selection: CandidateSelection,
) -> tuple[str, str]:
    if not isinstance(selection, CandidateSelection):
        _raise_validation("outer fit requires a typed CandidateSelection")
    candidate_set = set(selection.candidate_order)
    if candidate_set.issubset(_PRIMARY_OUTER_CANDIDATES):
        label = "B_field_selected"
    elif candidate_set.issubset(_BASELINE_OUTER_CANDIDATES):
        selected = selection.selected_name
        if selected not in _BASELINE_OUTER_CANDIDATES:
            _raise_validation("outer baseline selection is not registered")
        if selected not in selection.candidate_scores:
            _raise_validation("outer baseline candidate is missing from selection scores")
        return selected, selected
    elif candidate_set.issubset(INTERNAL_CANDIDATE_ORDER):
        label = "I_field_selected"
    else:
        label = None
    if label is None:
        _raise_validation("outer selection candidate order is not registered")
    selected = selection.selected_name
    if selected == label:
        candidates = tuple(selection.candidate_scores)
        if not candidates:
            _raise_validation("outer selected alias has no candidate scores")
        selected = min(
            candidates,
            key=lambda name: (
                selection.candidate_scores[name].inner_equal_domain_mae,
                (
                    _PRIMARY_OUTER_CANDIDATES_ORDER
                    if label.startswith("B")
                    else INTERNAL_CANDIDATE_ORDER
                ).index(name),
            ),
        )
    if selected not in _OUTER_CANDIDATES:
        _raise_validation("outer selected candidate is not registered")
    if label in _BASELINE_OUTER_CANDIDATES and selected != label:
        _raise_validation("outer selected candidate does not match baseline")
    if label == "B_field_selected" and selected not in _PRIMARY_OUTER_CANDIDATES:
        _raise_validation("outer selected candidate does not match B alias")
    if label == "I_field_selected" and selected not in _INTERNAL_OUTER_CANDIDATES:
        _raise_validation("outer selected candidate does not match I alias")
    if selected not in selection.candidate_scores:
        _raise_validation("outer selected candidate is missing from selection scores")
    return selected, label


def _register_outer_fitted_authority(
    state: OuterFittedCandidate,
    bundle: FeatureBundle,
    response: ResponseVector,
    selection: CandidateSelection,
    outer_train_authority: object,
    outer_test_authority: object,
) -> None:
    identity = id(state)

    def remove(_dead: ReferenceType[object]) -> None:
        if _OUTER_AUTHORITY.get(identity, (None,))[0] is _dead:
            _OUTER_AUTHORITY.pop(identity, None)

    try:
        _OUTER_AUTHORITY[identity] = (
            ref(state, remove),
            ref(bundle, lambda _dead: _OUTER_AUTHORITY.pop(identity, None)),
            ref(response, lambda _dead: _OUTER_AUTHORITY.pop(identity, None)),
            ref(selection, lambda _dead: _OUTER_AUTHORITY.pop(identity, None)),
            ref(
                outer_train_authority,
                lambda _dead: _OUTER_AUTHORITY.pop(identity, None),
            ),
            ref(
                outer_test_authority,
                lambda _dead: _OUTER_AUTHORITY.pop(identity, None),
            ),
            state.state_sha256,
        )
    except TypeError as error:
        _OUTER_AUTHORITY.pop(identity, None)
        raise FeatureValidationError(
            "outer production authorities must support weak identity binding"
        ) from error


def _outer_authority_record(
    state: OuterFittedCandidate,
) -> tuple[FeatureBundle, ResponseVector, CandidateSelection, object, object]:
    record = _OUTER_AUTHORITY.get(id(state))
    if record is None or record[0]() is not state:
        _raise_validation("outer fitted state has no construction authority")
    bundle = record[1]()
    response = record[2]()
    selection = record[3]()
    train_authority = record[4]()
    test_authority = record[5]()
    if not isinstance(bundle, FeatureBundle) or not isinstance(response, ResponseVector):
        _raise_validation("outer fitted source authority is unavailable")
    if not isinstance(selection, CandidateSelection):
        _raise_validation("outer fitted selection authority is unavailable")
    if train_authority is None or test_authority is None:
        _raise_validation("outer fitted data authorities are unavailable")
    return bundle, response, selection, train_authority, test_authority


def validate_outer_fitted_candidate(
    state: OuterFittedCandidate,
    *,
    bundle: FeatureBundle,
    response: ResponseVector,
    selection: CandidateSelection,
    outer_train_authority: object,
    outer_test_authority: object,
) -> bool:
    """Validate an outer fitted state against trusted source authorities."""

    if not isinstance(state, OuterFittedCandidate):
        _raise_validation("outer fitted state has an invalid type")
    _require_production_bundle(bundle)
    validate_feature_bundle(bundle)
    validate_response_vector(response, bundle)
    if not isinstance(selection, CandidateSelection):
        _raise_validation("outer fit selection must be a typed CandidateSelection")
    train_indices, test_indices, heldout_domain, train_state, test_state, _source = (
        _validate_outer_split_authorities(
            bundle, selection, outer_train_authority, outer_test_authority
        )
    )
    validate_candidate_selection(
        selection,
        bundle,
        response=response,
        outer_train_indices=train_indices,
        outer_test_ids=selection.outer_test_ids,
        outer_test_domains=selection.outer_test_domains,
        outer_train_authority=outer_train_authority,
        heldout_domain=heldout_domain,
    )
    candidate, label = _outer_candidate_and_label(selection)
    expected_fit_ids = tuple(bundle.sample_ids[index] for index in train_indices)
    expected_fit_domains = tuple(bundle.domain_ids[index] for index in train_indices)
    expected_test_ids = tuple(bundle.sample_ids[index] for index in test_indices)
    expected_test_domains = tuple(bundle.domain_ids[index] for index in test_indices)
    if (state.candidate, state.selection_label) != (candidate, label):
        _raise_validation("outer fitted candidate selection is not frozen")
    if state.fit_sample_ids != expected_fit_ids or state.fit_domain_ids != expected_fit_domains:
        _raise_validation("outer fitted train identity does not match authority")
    if state.test_sample_ids != expected_test_ids or state.test_domain_ids != expected_test_domains:
        _raise_validation("outer fitted test identity does not match authority")
    expected_dimension = selection.selected_pca_dimension if candidate in _FROZEN_CANDIDATES else None
    if state.pca_dimension != expected_dimension:
        _raise_validation("outer fitted PCA dimension does not match selection")
    if state.alpha != RIDGE_ALPHA:
        _raise_validation("outer fitted alpha is not frozen to 10.0")
    if state.bundle_state_sha256 != selection.bundle_state_sha256:
        _raise_validation("outer fitted bundle state does not match selection")
    if state.response_training_state_sha256 != selection.response_state_sha256:
        _raise_validation("outer fitted response state does not match selection")
    if state.selection_state_sha256 != selection.state_sha256:
        _raise_validation("outer fitted selection state changed")
    if state.feature_bundle_state_sha256 != bundle.state_sha256:
        _raise_validation("outer fitted feature bundle state changed")
    source_state = _source_state(bundle._source_authority, name="feature bundle source")
    if state.source_authority_state_sha256 != source_state:
        _raise_validation("outer fitted source authority state changed")
    if state.response_state_sha256 != response.state_sha256:
        _raise_validation("outer fitted response authority state changed")
    if state.outer_train_authority_state_sha256 != train_state:
        _raise_validation("outer fitted train authority state changed")
    if state.outer_test_authority_state_sha256 != test_state:
        _raise_validation("outer fitted test authority state changed")
    record = _OUTER_AUTHORITY.get(id(state))
    if record is None or record[0]() is not state:
        _raise_validation("outer fitted state has no registered construction authority")
    if any(
        expected is not supplied
        for expected, supplied in (
            (record[1](), bundle),
            (record[2](), response),
            (record[3](), selection),
            (record[4](), outer_train_authority),
            (record[5](), outer_test_authority),
        )
    ):
        _raise_validation("outer fitted source authority identity changed")
    _assert_authority(state, _outer_fitted_state(state), name="outer fitted state")
    return True


def fit_outer_candidate_and_predict(
    bundle: FeatureBundle,
    response: ResponseVector,
    selection: CandidateSelection,
    outer_train_authority: object,
    outer_test_authority: object,
) -> OuterFittedCandidate:
    """Refit one frozen selection on five issued domains and predict one held-out domain."""

    _require_production_bundle(bundle)
    validate_feature_bundle(bundle)
    validate_response_vector(response, bundle)
    if not isinstance(selection, CandidateSelection):
        _raise_validation("outer fit requires a typed CandidateSelection")
    train_indices, test_indices, heldout_domain, train_state, test_state, _source = (
        _validate_outer_split_authorities(
            bundle, selection, outer_train_authority, outer_test_authority
        )
    )
    validate_candidate_selection(
        selection,
        bundle,
        response=response,
        outer_train_indices=train_indices,
        outer_test_ids=selection.outer_test_ids,
        outer_test_domains=selection.outer_test_domains,
        outer_train_authority=outer_train_authority,
        heldout_domain=heldout_domain,
    )
    candidate, label = _outer_candidate_and_label(selection)
    pca_model: FoldLocalPCA | None = None
    train_frozen: np.ndarray | None = None
    test_frozen: np.ndarray | None = None
    pca_dimension = selection.selected_pca_dimension if candidate in _FROZEN_CANDIDATES else None
    if candidate in _FROZEN_CANDIDATES:
        if pca_dimension is None:
            _raise_validation("outer frozen candidate selection has no PCA dimension")
        train_frozen, pca_model = _pca_transform_rows(
            bundle,
            train_indices,
            fit_indices=train_indices,
            pca_dimension=pca_dimension,
            pca_model=None,
            fit_sample_ids=tuple(bundle.sample_ids[index] for index in train_indices),
            fit_domain_ids=tuple(bundle.domain_ids[index] for index in train_indices),
            outer_train_authority=outer_train_authority,
            heldout_domain=heldout_domain,
        )
        test_frozen, replayed = _pca_transform_rows(
            bundle,
            test_indices,
            fit_indices=train_indices,
            pca_dimension=pca_dimension,
            pca_model=pca_model,
            fit_sample_ids=tuple(bundle.sample_ids[index] for index in train_indices),
            fit_domain_ids=tuple(bundle.domain_ids[index] for index in train_indices),
            outer_train_authority=outer_train_authority,
            heldout_domain=heldout_domain,
        )
        if replayed is not pca_model:
            _raise_validation("outer PCA replay returned a different model identity")
    train_values, _names, _units = _candidate_components(
        bundle, candidate, train_indices, train_frozen
    )
    test_values, _names, _units = _candidate_components(
        bundle, candidate, test_indices, test_frozen
    )
    fit_ids = tuple(bundle.sample_ids[index] for index in train_indices)
    fit_domains = tuple(bundle.domain_ids[index] for index in train_indices)
    test_ids = tuple(bundle.sample_ids[index] for index in test_indices)
    test_domains = tuple(bundle.domain_ids[index] for index in test_indices)
    ridge = fit_fold_ridge(
        train_values,
        response.values[train_indices],
        alpha=RIDGE_ALPHA,
        fit_sample_ids=fit_ids,
        fit_domain_ids=fit_domains,
    )
    predictions = ridge.predict(test_values)
    global _OUTER_CONSTRUCTION_CONTEXT
    context = _OUTER_CONSTRUCTION_CONTEXT
    _OUTER_CONSTRUCTION_CONTEXT = (
        bundle,
        response,
        selection,
        outer_train_authority,
        outer_test_authority,
    )
    try:
        state = OuterFittedCandidate(
            candidate=candidate,
            selection_label=label,
            pca_dimension=pca_dimension,
            alpha=RIDGE_ALPHA,
            ridge_model=ridge,
            pca_model=pca_model,
            fit_sample_ids=fit_ids,
            fit_domain_ids=fit_domains,
            test_sample_ids=test_ids,
            test_domain_ids=test_domains,
            predictions=predictions,
            bundle_state_sha256=selection.bundle_state_sha256,
            response_training_state_sha256=selection.response_state_sha256,
            selection_state_sha256=selection.state_sha256,
            feature_bundle_state_sha256=bundle.state_sha256,
            source_authority_state_sha256=_source_state(
                bundle._source_authority, name="feature bundle source"
            ),
            response_state_sha256=response.state_sha256,
            outer_train_authority_state_sha256=train_state,
            outer_test_authority_state_sha256=test_state,
            state_sha256="0" * 64,
        )
        _register_outer_fitted_authority(
            state,
            bundle,
            response,
            selection,
            outer_train_authority,
            outer_test_authority,
        )
    finally:
        _OUTER_CONSTRUCTION_CONTEXT = context
    validate_outer_fitted_candidate(
        state,
        bundle=bundle,
        response=response,
        selection=selection,
        outer_train_authority=outer_train_authority,
        outer_test_authority=outer_test_authority,
    )
    return state


def _serialize_pca_model(model: FoldLocalPCA) -> dict[str, object]:
    _validate_pca_model(model)
    return {
        "mean": _encode_array(model.mean_),
        "components": _encode_array(model.components_),
        "explained_variance": _encode_array(model.explained_variance_),
        "singular_values": _encode_array(model.singular_values_),
        "n_components": model.n_components_,
        "n_features_in": model.n_features_in_,
        "n_samples_in": model.n_samples_in_,
        "authority_mode": model.authority_mode,
        "fit_sample_ids": list(model.fit_sample_ids),
        "fit_domain_ids": list(model.fit_domain_ids),
        "heldout_domain": model.heldout_domain,
        "inner_query_domain": model.inner_query_domain,
        "fit_authority_state_sha256": model.fit_authority_state_sha256,
        "outer_train_state_sha256": model.outer_train_state_sha256,
        "fit_embeddings_sha256": model.fit_embeddings_sha256,
        "state_sha256": model.state_sha256,
    }


def serialize_outer_fitted_candidate(state: OuterFittedCandidate) -> bytes:
    """Serialize complete outer state; source authorities remain external."""

    bundle, response, selection, train_authority, test_authority = _outer_authority_record(state)
    validate_outer_fitted_candidate(
        state,
        bundle=bundle,
        response=response,
        selection=selection,
        outer_train_authority=train_authority,
        outer_test_authority=test_authority,
    )
    ridge_payload = base64.b64encode(serialize_fold_ridge(state.ridge_model)).decode("ascii")
    return _json_bytes(
        {
            "version": 1,
            "candidate": state.candidate,
            "selection_label": state.selection_label,
            "pca_dimension": state.pca_dimension,
            "alpha": state.alpha,
            "ridge": ridge_payload,
            "pca": None if state.pca_model is None else _serialize_pca_model(state.pca_model),
            "fit_sample_ids": list(state.fit_sample_ids),
            "fit_domain_ids": list(state.fit_domain_ids),
            "test_sample_ids": list(state.test_sample_ids),
            "test_domain_ids": list(state.test_domain_ids),
            "predictions": _encode_array(state.predictions),
            "bundle_state_sha256": state.bundle_state_sha256,
            "response_training_state_sha256": state.response_training_state_sha256,
            "selection_state_sha256": state.selection_state_sha256,
            "feature_bundle_state_sha256": state.feature_bundle_state_sha256,
            "source_authority_state_sha256": state.source_authority_state_sha256,
            "response_state_sha256": state.response_state_sha256,
            "outer_train_authority_state_sha256": state.outer_train_authority_state_sha256,
            "outer_test_authority_state_sha256": state.outer_test_authority_state_sha256,
            "state_sha256": state.state_sha256,
        }
    )


def deserialize_outer_fitted_candidate(
    value: bytes,
    *,
    bundle: FeatureBundle,
    response: ResponseVector,
    selection: CandidateSelection,
    outer_train_authority: object,
    outer_test_authority: object,
) -> OuterFittedCandidate:
    """Replay serialized state only against the supplied trusted artifacts."""

    try:
        payload = _load_json_bytes(value, name="outer fitted candidate state")
        _require_json_schema(
            payload,
            name="outer fitted candidate state",
            schema={
                "version": int,
                "candidate": str,
                "selection_label": str,
                "pca_dimension": (int, type(None)),
                "alpha": float,
                "ridge": str,
                "pca": object,
                "fit_sample_ids": list,
                "fit_domain_ids": list,
                "test_sample_ids": list,
                "test_domain_ids": list,
                "predictions": str,
                "bundle_state_sha256": str,
                "response_training_state_sha256": str,
                "selection_state_sha256": str,
                "feature_bundle_state_sha256": str,
                "source_authority_state_sha256": str,
                "response_state_sha256": str,
                "outer_train_authority_state_sha256": str,
                "outer_test_authority_state_sha256": str,
                "state_sha256": str,
            },
        )
        for key in (
            "fit_sample_ids",
            "fit_domain_ids",
            "test_sample_ids",
            "test_domain_ids",
        ):
            _require_string_list(payload[key], name=f"outer {key}")
        pca_payload = payload["pca"]
        budget = _ArrayDecodeBudget()
        if pca_payload is not None:
            if not isinstance(pca_payload, Mapping):
                _raise_validation("serialized outer PCA state is invalid")
            _require_json_schema(
                pca_payload,
                name="outer PCA state",
                schema={
                    "mean": str,
                    "components": str,
                    "explained_variance": str,
                    "singular_values": str,
                    "n_components": int,
                    "n_features_in": int,
                    "n_samples_in": int,
                    "authority_mode": str,
                    "fit_sample_ids": list,
                    "fit_domain_ids": list,
                    "heldout_domain": str,
                    "inner_query_domain": str,
                    "fit_authority_state_sha256": str,
                    "outer_train_state_sha256": str,
                    "fit_embeddings_sha256": str,
                    "state_sha256": str,
                },
            )
            _require_string_list(
                pca_payload["fit_sample_ids"], name="outer PCA fit_sample_ids"
            )
            _require_string_list(
                pca_payload["fit_domain_ids"], name="outer PCA fit_domain_ids"
            )
            n_components = pca_payload["n_components"]
            n_features = pca_payload["n_features_in"]
            n_samples = pca_payload["n_samples_in"]
            if not (0 < n_components <= _MAX_SERIALIZED_ARRAY_ELEMENTS):
                _raise_validation("serialized outer PCA component count is invalid")
            if not (0 < n_features <= _MAX_SERIALIZED_ARRAY_ELEMENTS):
                _raise_validation("serialized outer PCA feature count is invalid")
            if not (0 < n_samples <= _MAX_SERIALIZED_ARRAY_ELEMENTS):
                _raise_validation("serialized outer PCA sample count is invalid")
            _decode_array(
                pca_payload["mean"],
                name="outer PCA mean",
                budget=budget,
                expected_dtype=np.dtype(np.float64),
                expected_shape=(n_features,),
            )
            _decode_array(
                pca_payload["components"],
                name="outer PCA components",
                budget=budget,
                expected_dtype=np.dtype(np.float64),
                expected_shape=(n_components, n_features),
            )
            _decode_array(
                pca_payload["explained_variance"],
                name="outer PCA explained variance",
                budget=budget,
                expected_dtype=np.dtype(np.float64),
                expected_shape=(n_components,),
            )
            _decode_array(
                pca_payload["singular_values"],
                name="outer PCA singular values",
                budget=budget,
                expected_dtype=np.dtype(np.float64),
                expected_shape=(n_components,),
            )
        if payload["version"] != 1:
            _raise_validation("serialized outer fitted candidate version is unsupported")
        expected = fit_outer_candidate_and_predict(
            bundle,
            response,
            selection,
            outer_train_authority,
            outer_test_authority,
        )
        checks = {
            "candidate": expected.candidate,
            "selection_label": expected.selection_label,
            "pca_dimension": expected.pca_dimension,
            "alpha": expected.alpha,
            "fit_sample_ids": list(expected.fit_sample_ids),
            "fit_domain_ids": list(expected.fit_domain_ids),
            "test_sample_ids": list(expected.test_sample_ids),
            "test_domain_ids": list(expected.test_domain_ids),
            "bundle_state_sha256": expected.bundle_state_sha256,
            "response_training_state_sha256": expected.response_training_state_sha256,
            "selection_state_sha256": expected.selection_state_sha256,
            "feature_bundle_state_sha256": expected.feature_bundle_state_sha256,
            "source_authority_state_sha256": expected.source_authority_state_sha256,
            "response_state_sha256": expected.response_state_sha256,
            "outer_train_authority_state_sha256": expected.outer_train_authority_state_sha256,
            "outer_test_authority_state_sha256": expected.outer_test_authority_state_sha256,
            "state_sha256": expected.state_sha256,
        }
        if any(payload.get(key) != item for key, item in checks.items()):
            _raise_validation("serialized outer fitted candidate identity changed")
        predictions = _decode_array(
            payload["predictions"],
            name="outer predictions",
            budget=budget,
            expected_dtype=np.dtype(np.float64),
            expected_shape=expected.predictions.shape,
        )
        if not np.array_equal(predictions, expected.predictions):
            _raise_validation("serialized outer predictions do not match trusted replay")
        try:
            ridge = deserialize_fold_ridge(
                _decode_base64_bytes(payload["ridge"], name="outer Ridge state"),
                authority=expected.ridge_model,
                _budget=budget,
            )
        except FeatureValidationError:
            raise
        except (ValueError, TypeError, MemoryError) as error:
            raise FeatureValidationError("serialized outer Ridge state is invalid") from error
        if ridge.state_sha256 != expected.ridge_model.state_sha256:
            _raise_validation("serialized outer Ridge state does not match trusted replay")
        if expected.pca_model is None:
            if pca_payload is not None:
                _raise_validation("serialized non-PCA outer state carries PCA payload")
        elif not isinstance(pca_payload, Mapping):
            _raise_validation("serialized outer PCA state is invalid")
        else:
            expected_pca = _serialize_pca_model(expected.pca_model)
            for key, expected_value in expected_pca.items():
                if pca_payload.get(key) != expected_value:
                    _raise_validation(
                        "serialized outer PCA state does not match trusted replay"
                    )
        return expected
    except FeatureValidationError:
        raise
    except Exception as error:
        raise FeatureValidationError(
            "serialized outer fitted candidate payload is invalid"
        ) from error


__all__ = [
    "A_FEATURE_COUNT",
    "BASELINE_CANDIDATE_ORDER",
    "B_SCALAR_FEATURE_COUNT",
    "FROZEN_EMBEDDING_FEATURE_COUNT",
    "INTERNAL_CANDIDATE_ORDER",
    "METADATA_FEATURE_COUNT",
    "METADATA_FEATURE_NAMES",
    "MORPHOLOGY_FEATURE_COUNT",
    "PCA_DIMENSIONS",
    "PCA_TIE_TOLERANCE",
    "PRIMARY_CANDIDATE_ORDER",
    "RIDGE_ALPHA",
    "SCALAR_FEATURE_NAMES",
    "SCALAR_FEATURE_UNITS",
    "SURFACE_FEATURE_COUNT",
    "SURFACE_FEATURE_NAMES",
    "CandidateScore",
    "CandidateSelection",
    "DimensionScore",
    "FeatureBundle",
    "FeatureMatrix",
    "FeatureValidationError",
    "FoldPreprocessor",
    "FoldProvenance",
    "FoldRidgeModel",
    "OuterFittedCandidate",
    "ResponseVector",
    "assemble_feature_matrices",
    "deserialize_feature_matrix",
    "deserialize_fold_ridge",
    "deserialize_outer_fitted_candidate",
    "domain_mae",
    "equal_domain_mae",
    "fit_fold_preprocessor",
    "fit_fold_ridge",
    "fit_fold_scaler",
    "fit_outer_candidate_and_predict",
    "fit_ridge",
    "make_feature_bundle_from_v3_data",
    "make_production_feature_bundle",
    "make_response_vector",
    "make_response_vector_from_v3_data",
    "make_test_feature_bundle",
    "make_test_response_vector",
    "select_baseline_candidate",
    "select_candidate",
    "select_internal_candidate",
    "select_pca_dimension",
    "serialize_feature_matrix",
    "serialize_fold_ridge",
    "serialize_outer_fitted_candidate",
    "validate_candidate_selection",
    "validate_feature_bundle",
    "validate_feature_matrix",
    "validate_fold_preprocessor_state",
    "validate_fold_ridge_state",
    "validate_outer_fitted_candidate",
    "validate_response_vector",
]
