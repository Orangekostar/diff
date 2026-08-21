"""Process-local fold authorities for D8 search isolation."""

from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import InitVar, dataclass, field

import numpy as np

from cmc_bbdm.cpb_v3.data import (
    V3Data,
    V3DataView,
    validate_issued_data_authority,
)

from .config import DOMAIN_ORDER, D8Config


class D8AuthorityError(ValueError):
    """Raised when a D8 fold view loses its loader-issued authority."""


_ISSUER = object()
_SEARCH_REGISTRY: weakref.WeakKeyDictionary[D8SearchView, str] = (
    weakref.WeakKeyDictionary()
)
_INNER_REGISTRY: weakref.WeakKeyDictionary[D8InnerFold, str] = (
    weakref.WeakKeyDictionary()
)


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strict_domain(value: object, *, label: str) -> str:
    if type(value) is not str or value not in DOMAIN_ORDER:
        raise D8AuthorityError(f"{label} is not a registered domain")
    return value


def _strict_config_sha256(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise D8AuthorityError("config SHA-256 is invalid")
    return value


def _readonly_indices(value: object, *, length: int, label: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise D8AuthorityError(f"{label} must be integer indices") from error
    if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.integer):
        raise D8AuthorityError(f"{label} must be integer indices")
    indices = np.asarray(raw, dtype=np.int64)
    if (
        indices.ndim != 1
        or len(indices) == 0
        or len(np.unique(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= length)
    ):
        raise D8AuthorityError(f"{label} are invalid")
    indices = np.array(indices, dtype=np.int64, copy=True)
    indices.setflags(write=False)
    return indices


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class D8SearchView:
    """Five-domain view issued before any prospective outer evaluation."""

    _token: InitVar[object]
    outer_domain: str
    data_view: V3DataView
    config_sha256: str
    state_sha256: str = field(init=False)

    def __post_init__(self, _token: object) -> None:
        if _token is not _ISSUER:
            raise TypeError("D8 search views require the authority issuer")
        outer = _strict_domain(self.outer_domain, label="outer domain")
        if type(self.data_view) is not V3DataView:
            raise D8AuthorityError("search view requires an exact V3DataView")
        try:
            source_state = validate_issued_data_authority(self.data_view)
        except (TypeError, ValueError) as error:
            raise D8AuthorityError(
                "search view requires loader-issued V3 data"
            ) from error
        domains = tuple(str(item) for item in self.data_view.dataset_ids.tolist())
        expected = tuple(domain for domain in DOMAIN_ORDER if domain != outer)
        if outer in domains or tuple(dict.fromkeys(domains)) != expected:
            raise D8AuthorityError("search view domain roster changed")
        config_sha = _strict_config_sha256(self.config_sha256)
        state = _canonical_digest(
            {
                "kind": "D8SearchView",
                "outer_domain": outer,
                "config_sha256": config_sha,
                "source_state_sha256": source_state,
                "specimen_ids": self.specimen_ids,
                "dataset_ids": domains,
            }
        )
        object.__setattr__(self, "outer_domain", outer)
        object.__setattr__(self, "config_sha256", config_sha)
        object.__setattr__(self, "state_sha256", state)
        _SEARCH_REGISTRY[self] = state

    @property
    def specimen_count(self) -> int:
        return self.data_view.n_samples

    @property
    def specimen_ids(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.data_view.sample_ids.tolist())

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.data_view.dataset_ids.tolist())

    def __copy__(self):
        raise TypeError("D8 search views cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("D8 search views cannot be copied")

    def __reduce__(self):
        raise TypeError("D8 search views cannot be pickled")


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class D8InnerFold:
    """Four-domain fit and one-domain query split inside a D8 search view."""

    _token: InitVar[object]
    search_view: D8SearchView
    query_domain: str
    fit_indices: np.ndarray
    query_indices: np.ndarray
    state_sha256: str = field(init=False)

    def __post_init__(self, _token: object) -> None:
        if _token is not _ISSUER:
            raise TypeError("D8 inner folds require the authority issuer")
        parent_state = validate_search_view(self.search_view)
        query = _strict_domain(self.query_domain, label="query domain")
        if query == self.search_view.outer_domain:
            raise D8AuthorityError("inner query cannot equal the outer domain")
        fit = _readonly_indices(
            self.fit_indices,
            length=self.search_view.specimen_count,
            label="inner fit indices",
        )
        query_indices = _readonly_indices(
            self.query_indices,
            length=self.search_view.specimen_count,
            label="inner query indices",
        )
        if set(fit.tolist()) & set(query_indices.tolist()):
            raise D8AuthorityError("inner fit and query identities overlap")
        if set(fit.tolist()) | set(query_indices.tolist()) != set(
            range(self.search_view.specimen_count)
        ):
            raise D8AuthorityError("inner fold does not cover the search view")
        domains = np.asarray(self.search_view.data_view.dataset_ids)
        if set(domains[query_indices].tolist()) != {query}:
            raise D8AuthorityError("inner query domain roster changed")
        expected_fit = set(DOMAIN_ORDER) - {self.search_view.outer_domain, query}
        if set(domains[fit].tolist()) != expected_fit:
            raise D8AuthorityError("inner fit domain roster changed")
        state = _canonical_digest(
            {
                "kind": "D8InnerFold",
                "search_state_sha256": parent_state,
                "query_domain": query,
                "fit_indices": fit.tolist(),
                "query_indices": query_indices.tolist(),
            }
        )
        object.__setattr__(self, "query_domain", query)
        object.__setattr__(self, "fit_indices", fit)
        object.__setattr__(self, "query_indices", query_indices)
        object.__setattr__(self, "state_sha256", state)
        _INNER_REGISTRY[self] = state

    @property
    def outer_domain(self) -> str:
        return self.search_view.outer_domain

    @property
    def fit_specimen_ids(self) -> tuple[str, ...]:
        values = self.search_view.data_view.sample_ids[self.fit_indices]
        return tuple(str(item) for item in values.tolist())

    @property
    def query_specimen_ids(self) -> tuple[str, ...]:
        values = self.search_view.data_view.sample_ids[self.query_indices]
        return tuple(str(item) for item in values.tolist())

    @property
    def fit_dataset_ids(self) -> tuple[str, ...]:
        values = self.search_view.data_view.dataset_ids[self.fit_indices]
        return tuple(str(item) for item in values.tolist())

    @property
    def query_dataset_ids(self) -> tuple[str, ...]:
        values = self.search_view.data_view.dataset_ids[self.query_indices]
        return tuple(str(item) for item in values.tolist())

    def __copy__(self):
        raise TypeError("D8 inner folds cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("D8 inner folds cannot be copied")

    def __reduce__(self):
        raise TypeError("D8 inner folds cannot be pickled")


def validate_search_view(value: object) -> str:
    if type(value) is not D8SearchView:
        raise D8AuthorityError("exact D8SearchView is required")
    registered = _SEARCH_REGISTRY.get(value)
    if registered is None:
        raise D8AuthorityError("search view has no process-local authority")
    try:
        source_state = validate_issued_data_authority(value.data_view)
    except (TypeError, ValueError) as error:
        raise D8AuthorityError("search view data authority is invalid") from error
    current = _canonical_digest(
        {
            "kind": "D8SearchView",
            "outer_domain": value.outer_domain,
            "config_sha256": value.config_sha256,
            "source_state_sha256": source_state,
            "specimen_ids": value.specimen_ids,
            "dataset_ids": value.dataset_ids,
        }
    )
    if current != registered or value.state_sha256 != registered:
        raise D8AuthorityError("search view state changed")
    return current


def validate_inner_fold(value: object) -> str:
    if type(value) is not D8InnerFold:
        raise D8AuthorityError("exact D8InnerFold is required")
    registered = _INNER_REGISTRY.get(value)
    if registered is None:
        raise D8AuthorityError("inner fold has no process-local authority")
    parent_state = validate_search_view(value.search_view)
    current = _canonical_digest(
        {
            "kind": "D8InnerFold",
            "search_state_sha256": parent_state,
            "query_domain": value.query_domain,
            "fit_indices": value.fit_indices.tolist(),
            "query_indices": value.query_indices.tolist(),
        }
    )
    if (
        value.fit_indices.flags.writeable
        or value.query_indices.flags.writeable
        or current != registered
        or value.state_sha256 != registered
    ):
        raise D8AuthorityError("inner fold state changed")
    return current


def issue_search_view(
    data: object, *, outer_domain: str, config: D8Config
) -> D8SearchView:
    if type(data) is not V3Data:
        raise D8AuthorityError("search issuance requires exact V3Data")
    try:
        validate_issued_data_authority(data)
    except (TypeError, ValueError) as error:
        raise D8AuthorityError("search issuance requires loader-issued data") from error
    if type(config) is not D8Config:
        raise D8AuthorityError("search issuance requires exact D8Config")
    outer = _strict_domain(outer_domain, label="outer domain")
    if tuple(config.outer_domains) != DOMAIN_ORDER:
        raise D8AuthorityError("D8 config domain roster changed")
    indices = np.flatnonzero(np.asarray(data.dataset_ids) != outer)
    return D8SearchView(
        _ISSUER,
        outer_domain=outer,
        data_view=data.subset(indices),
        config_sha256=config.config_sha256,
    )


def issue_inner_fold(search_view: D8SearchView, *, query_domain: str) -> D8InnerFold:
    validate_search_view(search_view)
    query = _strict_domain(query_domain, label="query domain")
    domains = np.asarray(search_view.data_view.dataset_ids)
    query_indices = np.flatnonzero(domains == query)
    fit_indices = np.flatnonzero(domains != query)
    return D8InnerFold(
        _ISSUER,
        search_view=search_view,
        query_domain=query,
        fit_indices=fit_indices,
        query_indices=query_indices,
    )


def issue_evaluation_view(
    data: object,
    *,
    selection: object,
    outer_domain: str,
    config: D8Config,
) -> None:
    del data, outer_domain, config
    if selection is None:
        raise D8AuthorityError(
            "outer evaluation requires a hash-frozen selection authority"
        )
    raise D8AuthorityError("outer evaluation is prohibited during the D8 pilot")


__all__ = [
    "D8AuthorityError",
    "D8InnerFold",
    "D8SearchView",
    "issue_evaluation_view",
    "issue_inner_fold",
    "issue_search_view",
    "validate_inner_fold",
    "validate_search_view",
]
