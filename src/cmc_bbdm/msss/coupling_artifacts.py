"""Protocol, parent evidence, and immutable artifacts for NO-GO coupling."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import yaml

from .artifacts import S1ArtifactError, validate_s1_package
from .authority import load_authority
from .coupling import CouplingDiagnostic, diagnose_coupling
from .protocol import load_protocol
from .scale_features import ScaleCondition


class CouplingArtifactError(ValueError):
    """Raised when the coupling protocol or artifact package is invalid."""


@dataclass(frozen=True, slots=True)
class CouplingSource:
    name: str
    path: Path
    relative_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class CouplingProtocol:
    config_path: Path
    config_sha256: str
    sources: tuple[CouplingSource, ...]
    msss_config: Path
    parent_path: Path
    required_gate_status: str
    required_test_only: bool
    parent_scientific_digest: str
    parent_output_tree_sha256: str
    axes: tuple[str, ...]
    wavelet_primary_family: str
    wavelet_primary_mode: str
    margin: float
    groupings: tuple[str, ...]
    damage_metrics: tuple[str, ...]
    output_formal: Path
    output_replay: Path


@dataclass(frozen=True, slots=True)
class ParentCandidateEvidence:
    conditions: tuple[ScaleCondition, ...]
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    absolute_errors: Mapping[str, np.ndarray]
    scientific_digest: str
    output_tree_sha256: str

    @property
    def specimen_count(self) -> int:
        return len(self.specimen_ids)


@dataclass(frozen=True, slots=True)
class CouplingPackageValidation:
    coupling_status: str
    validation_status: str
    s2_status: str
    diagnostic_state_sha256: str
    scientific_digest: str
    output_tree_sha256: str


REQUIRED_COUPLING_FILES = frozenset(
    {
        "group_scale_curves.csv",
        "group_scale_selection.csv",
        "damage_size_bins.csv",
        "factor_trends.csv",
        "summary.json",
        "REPORT.md",
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    }
)

_SOURCE_NAMES = (
    "msss_config",
    "coupling_protocol",
    "coupling_design",
    "coupling_plan",
)
_AXES = ("sampling", "gaussian", "wavelet")
_GROUPINGS = (
    "domain",
    "ply_count",
    "layup_family",
    "damage_area",
    "damage_height",
    "damage_width",
)
_DAMAGE = ("damage_area", "damage_height", "damage_width")
_CURVE_FIELDS = (
    "axis",
    "grouping",
    "group_value",
    "condition_id",
    "coarse_rank",
    "specimen_count",
    "domain_count",
    "equal_domain_mae",
    "full_equal_domain_mae",
    "relative_gap",
    "noninferior_05",
)
_SELECTION_FIELDS = (
    "axis",
    "grouping",
    "group_value",
    "selected_condition_id",
    "selected_coarse_rank",
    "full_condition_id",
    "over_coarse_condition_id",
    "boundary_confirmed",
    "sufficient_condition_ids_json",
    "specimen_count",
    "domain_count",
)
_DAMAGE_FIELDS = ("specimen_id", "dataset_id", "metric", "value", "tertile")
_TREND_FIELDS = (
    "axis",
    "factor",
    "group_order_json",
    "selected_condition_ids_json",
    "coarse_ranks_json",
    "direction",
    "alignment_status",
    "aligned_direction",
    "aligned_axis_count",
)
_SCIENTIFIC_FILES = (
    "group_scale_curves.csv",
    "group_scale_selection.csv",
    "damage_size_bins.csv",
    "factor_trends.csv",
    "summary.json",
)
_PARENT_CURVE_FIELDS = (
    "condition_id",
    "value",
    "coarse_rank",
    "primary_eligible",
    "wavelet",
    "level",
    "mode",
    "normalized_retention_index",
    "equal_domain_mae",
    "ci_low",
    "ci_high",
    "full_equal_domain_mae",
    "relative_gap",
    "noninferior_025",
    "noninferior_05",
    "noninferior_075",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CouplingArtifactError(f"{label} must be a lowercase SHA-256")
    return value


def _mapping(value: object, keys: Sequence[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or tuple(value) != tuple(keys):
        raise CouplingArtifactError(f"{label} keys are not exact")
    return value


def _path(root: Path, value: object, label: str, *, must_exist: bool) -> Path:
    if type(value) is not str or not value:
        raise CouplingArtifactError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise CouplingArtifactError(f"{label} path escapes the project root")
    result = (root / relative).resolve()
    if not result.is_relative_to(root) or (must_exist and not result.exists()):
        raise CouplingArtifactError(f"{label} path is unavailable")
    return result


def load_coupling_protocol(
    config_path: str | Path, *, project_root: str | Path
) -> CouplingProtocol:
    """Load the exact frozen post-NO-GO diagnostic protocol."""

    root = Path(project_root).resolve(strict=True)
    config = Path(config_path).resolve(strict=True)
    if not config.is_relative_to(root) or not config.is_file():
        raise CouplingArtifactError("coupling config is outside the project root")
    try:
        payload = config.read_bytes()
        raw = yaml.safe_load(payload)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CouplingArtifactError("coupling config is unreadable") from error
    top = _mapping(
        raw,
        ("schema_version", "scope", "sources", "parent", "analysis", "outputs"),
        "coupling config",
    )
    if top["schema_version"] != 1 or top["scope"] != "msss_s1_no_go_coupling_diagnostic":
        raise CouplingArtifactError("coupling config identity changed")
    source_raw = _mapping(top["sources"], _SOURCE_NAMES, "coupling sources")
    sources: list[CouplingSource] = []
    for name in _SOURCE_NAMES:
        item = _mapping(source_raw[name], ("path", "sha256"), f"source {name}")
        source_path = _path(root, item["path"], f"source {name}", must_exist=True)
        expected = _hash_text(item["sha256"], f"source {name}")
        if not source_path.is_file() or _sha256(source_path) != expected:
            raise CouplingArtifactError(f"source hash mismatch: {name}")
        sources.append(
            CouplingSource(name, source_path, str(item["path"]), expected)
        )
    parent = _mapping(
        top["parent"],
        (
            "path",
            "required_gate_status",
            "required_test_only",
            "scientific_digest",
            "output_tree_sha256",
        ),
        "coupling parent",
    )
    parent_path = _path(root, parent["path"], "coupling parent", must_exist=True)
    if (
        not parent_path.is_dir()
        or parent["required_gate_status"] != "NO_GO"
        or type(parent["required_test_only"]) is not bool
        or parent["required_test_only"]
    ):
        raise CouplingArtifactError("coupling parent gate is invalid")
    analysis = _mapping(
        top["analysis"],
        (
            "axes",
            "wavelet_primary_family",
            "wavelet_primary_mode",
            "relative_margin",
            "groupings",
            "damage_metrics",
            "damage_binning",
            "damage_bin_labels",
            "cross_axis_alignment_minimum",
            "validation_status",
            "s2_status",
        ),
        "coupling analysis",
    )
    if (
        tuple(analysis["axes"]) != _AXES
        or analysis["wavelet_primary_family"] != "db2"
        or analysis["wavelet_primary_mode"] != "low_only"
        or float(analysis["relative_margin"]) != 0.05
        or tuple(analysis["groupings"]) != _GROUPINGS
        or tuple(analysis["damage_metrics"]) != _DAMAGE
        or analysis["damage_binning"]
        != "stable_rank_balanced_tertiles_value_then_specimen_id"
        or tuple(analysis["damage_bin_labels"]) != ("low", "middle", "high")
        or analysis["cross_axis_alignment_minimum"] != 2
        or analysis["validation_status"] != "NOT_VALIDATED_POST_HOC"
        or analysis["s2_status"] != "NOT_RUN_NOT_AUTHORIZED"
    ):
        raise CouplingArtifactError("coupling analysis registry changed")
    outputs = _mapping(top["outputs"], ("formal", "replay"), "coupling outputs")
    output_formal = _path(root, outputs["formal"], "formal output", must_exist=False)
    output_replay = _path(root, outputs["replay"], "replay output", must_exist=False)
    source_index = {item.name: item for item in sources}
    return CouplingProtocol(
        config_path=config,
        config_sha256=hashlib.sha256(payload).hexdigest(),
        sources=tuple(sources),
        msss_config=source_index["msss_config"].path,
        parent_path=parent_path,
        required_gate_status=str(parent["required_gate_status"]),
        required_test_only=bool(parent["required_test_only"]),
        parent_scientific_digest=_hash_text(
            parent["scientific_digest"], "parent scientific digest"
        ),
        parent_output_tree_sha256=_hash_text(
            parent["output_tree_sha256"], "parent output-tree digest"
        ),
        axes=_AXES,
        wavelet_primary_family="db2",
        wavelet_primary_mode="low_only",
        margin=0.05,
        groupings=_GROUPINGS,
        damage_metrics=_DAMAGE,
        output_formal=output_formal,
        output_replay=output_replay,
    )


def _json_file(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CouplingArtifactError(f"invalid JSON: {path.name}") from error
    if type(value) is not dict:
        raise CouplingArtifactError(f"invalid JSON: {path.name}")
    return value


def _parent_conditions(parent: Path) -> tuple[ScaleCondition, ...]:
    conditions: list[ScaleCondition] = []
    try:
        for axis, expected_count in zip(_AXES, (9, 9, 4), strict=True):
            axis_conditions: list[ScaleCondition] = []
            with (parent / f"{axis}_curve.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                reader = csv.DictReader(handle, strict=True)
                if tuple(reader.fieldnames or ()) != _PARENT_CURVE_FIELDS:
                    raise CouplingArtifactError("parent curve schema changed")
                for row in reader:
                    eligible = row["primary_eligible"].lower()
                    if eligible not in {"true", "false"}:
                        raise CouplingArtifactError("parent eligibility is invalid")
                    if eligible == "false":
                        continue
                    rank = int(row["coarse_rank"])
                    level = None if not row["level"] else int(row["level"])
                    condition = ScaleCondition(
                        condition_id=row["condition_id"],
                        axis=axis,
                        value=float(row["value"]),
                        coarse_rank=rank,
                        primary_eligible=True,
                        is_full_identity=rank == 0,
                        wavelet=None if not row["wavelet"] else row["wavelet"],
                        level=level,
                        mode=None if not row["mode"] else row["mode"],
                    )
                    axis_conditions.append(condition)
            if (
                len(axis_conditions) != expected_count
                or tuple(item.coarse_rank for item in axis_conditions)
                != tuple(range(expected_count))
                or sum(item.is_full_identity for item in axis_conditions) != 1
                or any(item.axis != axis for item in axis_conditions)
                or (
                    axis == "wavelet"
                    and any(
                        item.wavelet != "db2" or item.mode != "low_only"
                        for item in axis_conditions
                    )
                )
            ):
                raise CouplingArtifactError("primary parent condition registry changed")
            conditions.extend(axis_conditions)
    except CouplingArtifactError:
        raise
    except (OSError, UnicodeError, csv.Error, TypeError, ValueError) as error:
        raise CouplingArtifactError("parent scale curves are unreadable") from error
    if len({item.condition_id for item in conditions}) != len(conditions):
        raise CouplingArtifactError("parent condition IDs are duplicated")
    return tuple(conditions)


def load_parent_candidate_errors(
    protocol: CouplingProtocol, *, project_root: str | Path
) -> ParentCandidateEvidence:
    """Validate and load the complete primary S1 cross-fitted error roster."""

    if type(protocol) is not CouplingProtocol:
        raise CouplingArtifactError("issued coupling protocol is required")
    root = Path(project_root).resolve(strict=True)
    try:
        validation = validate_s1_package(
            protocol.parent_path,
            project_root=root,
            config_path=protocol.msss_config,
        )
    except (S1ArtifactError, OSError, ValueError) as error:
        raise CouplingArtifactError("parent S1 package validation failed") from error
    if (
        validation.gate_status != protocol.required_gate_status
        or validation.test_only != protocol.required_test_only
    ):
        raise CouplingArtifactError("parent S1 gate does not authorize this diagnostic")
    if (
        validation.scientific_digest != protocol.parent_scientific_digest
        or validation.output_tree_sha256 != protocol.parent_output_tree_sha256
    ):
        raise CouplingArtifactError("parent S1 digest changed")
    conditions = _parent_conditions(protocol.parent_path)
    if tuple(sum(item.axis == axis for item in conditions) for axis in protocol.axes) != (
        9,
        9,
        4,
    ):
        raise CouplingArtifactError("primary parent condition registry changed")
    index = _json_file(protocol.parent_path / "feature_index.json")
    specimen_raw, dataset_raw = index.get("specimen_ids"), index.get("dataset_ids")
    if not isinstance(specimen_raw, list) or not isinstance(dataset_raw, list):
        raise CouplingArtifactError("parent feature identity is missing")
    specimens = tuple(str(value) for value in specimen_raw)
    datasets = tuple(str(value) for value in dataset_raw)
    if (
        len(specimens) != 276
        or len(set(specimens)) != 276
        or len(datasets) != 276
        or any(not value for value in specimens + datasets)
    ):
        raise CouplingArtifactError("parent feature roster changed")
    specimen_index = {value: index for index, value in enumerate(specimens)}
    condition_index = {item.condition_id: item for item in conditions}
    arrays = {
        condition_id: np.full(276, np.nan, dtype=np.float64)
        for condition_id in condition_index
    }
    seen: set[tuple[str, str]] = set()
    try:
        with (protocol.parent_path / "candidate_predictions.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle, strict=True)
            required = (
                "axis",
                "condition_id",
                "specimen_id",
                "dataset_id",
                "outer_group",
                "target",
                "prediction",
                "absolute_error",
                "selected_pca_dimension",
                "fit_state_sha256",
            )
            if tuple(reader.fieldnames or ()) != required:
                raise CouplingArtifactError("parent candidate schema changed")
            for row in reader:
                condition_id = row["condition_id"]
                specimen = row["specimen_id"]
                if condition_id not in condition_index:
                    continue
                key = (condition_id, specimen)
                if key in seen or specimen not in specimen_index:
                    raise CouplingArtifactError("parent candidate roster is duplicated")
                index_value = specimen_index[specimen]
                if (
                    row["axis"] != condition_index[condition_id].axis
                    or row["dataset_id"] != datasets[index_value]
                    or row["outer_group"] != datasets[index_value]
                ):
                    raise CouplingArtifactError("parent candidate identity changed")
                target = float(row["target"])
                prediction = float(row["prediction"])
                error = float(row["absolute_error"])
                if (
                    not all(math.isfinite(value) for value in (target, prediction, error))
                    or error < 0.0
                    or not math.isclose(
                        error, abs(target - prediction), rel_tol=1.0e-12, abs_tol=1.0e-14
                    )
                ):
                    raise CouplingArtifactError("parent candidate error is invalid")
                arrays[condition_id][index_value] = error
                seen.add(key)
    except CouplingArtifactError:
        raise
    except (OSError, UnicodeError, csv.Error, TypeError, ValueError) as error:
        raise CouplingArtifactError("parent candidate predictions are unreadable") from error
    expected = len(conditions) * len(specimens)
    if len(seen) != expected or any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise CouplingArtifactError("parent candidate roster is incomplete")
    frozen: dict[str, np.ndarray] = {}
    for condition_id, array in arrays.items():
        value = np.frombuffer(array.astype("<f8", copy=False).tobytes(), dtype="<f8")
        value.setflags(write=False)
        frozen[condition_id] = value
    return ParentCandidateEvidence(
        conditions=conditions,
        specimen_ids=specimens,
        dataset_ids=datasets,
        absolute_errors=MappingProxyType(frozen),
        scientific_digest=validation.scientific_digest,
        output_tree_sha256=validation.output_tree_sha256,
    )


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _scientific_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for name in _SCIENTIFIC_FILES:
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name not in {"artifact_manifest.json", "CHECKSUMS.sha256"}:
            digest.update(path.name.encode("ascii"))
            digest.update(b"\0")
            digest.update(_sha256(path).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def _write_package(
    root: Path, *, protocol: CouplingProtocol, diagnostic: CouplingDiagnostic
) -> None:
    _write_csv(root / "group_scale_curves.csv", _CURVE_FIELDS, [asdict(item) for item in diagnostic.curves])
    selection_rows = []
    for item in diagnostic.selections:
        row = asdict(item)
        row["sufficient_condition_ids_json"] = _json(row.pop("sufficient_condition_ids"))
        selection_rows.append(row)
    _write_csv(root / "group_scale_selection.csv", _SELECTION_FIELDS, selection_rows)
    _write_csv(root / "damage_size_bins.csv", _DAMAGE_FIELDS, [asdict(item) for item in diagnostic.damage_bins])
    alignment_index = {item.factor: item for item in diagnostic.alignments}
    trend_rows = []
    for item in diagnostic.trends:
        alignment = alignment_index[item.factor]
        trend_rows.append(
            {
                "axis": item.axis,
                "factor": item.factor,
                "group_order_json": _json(item.group_order),
                "selected_condition_ids_json": _json(item.selected_condition_ids),
                "coarse_ranks_json": _json(item.coarse_ranks),
                "direction": item.direction,
                "alignment_status": alignment.status,
                "aligned_direction": "" if alignment.direction is None else alignment.direction,
                "aligned_axis_count": alignment.axis_count,
            }
        )
    _write_csv(root / "factor_trends.csv", _TREND_FIELDS, trend_rows)
    summary = {
        "schema_version": 1,
        "stage": "S1_NO_GO_COUPLING_DIAGNOSTIC",
        "coupling_status": diagnostic.coupling_status,
        "validation_status": diagnostic.validation_status,
        "s2_status": diagnostic.s2_status,
        "diagnostic_state_sha256": diagnostic.state_sha256,
        "parent": {
            "gate_status": protocol.required_gate_status,
            "scientific_digest": protocol.parent_scientific_digest,
            "output_tree_sha256": protocol.parent_output_tree_sha256,
        },
        "margin": protocol.margin,
        "aligned_factors": [asdict(item) for item in diagnostic.alignments],
        "group_curve_rows": len(diagnostic.curves),
        "group_selections": len(diagnostic.selections),
    }
    (root / "summary.json").write_text(_json(summary) + "\n", encoding="utf-8")
    lines = [
        "# MSSS S1 NO-GO Coupling Diagnostic",
        "",
        f"Exploratory status: **{diagnostic.coupling_status}**.",
        "",
        "| Factor | Cross-axis status | Direction | Axes |",
        "|---|---|---|---:|",
    ]
    lines.extend(
        f"| {item.factor} | {item.status} | {item.direction or 'none'} | {item.axis_count}/3 |"
        for item in diagnostic.alignments
    )
    lines.extend(
        [
            "",
            "This is a post-hoc diagnostic derived from cross-fitted formal S1 predictions.",
            "It does not validate Scale-Laminate Coupling, change S1 NO_GO, or authorize S2.",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    scientific = _scientific_digest(root)
    tree = _tree_digest(root)
    manifest = {
        "schema_version": 1,
        "config_sha256": protocol.config_sha256,
        "sources": {
            item.name: {"path": item.relative_path, "sha256": item.sha256}
            for item in protocol.sources
        },
        "parent_scientific_digest": protocol.parent_scientific_digest,
        "parent_output_tree_sha256": protocol.parent_output_tree_sha256,
        "diagnostic_state_sha256": diagnostic.state_sha256,
        "scientific_digest": scientific,
        "output_tree_sha256": tree,
        "files": {
            path.name: _sha256(path)
            for path in sorted(root.iterdir())
            if path.is_file()
        },
    }
    (root / "artifact_manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    checksum_paths = tuple(path for path in sorted(root.iterdir()) if path.is_file())
    (root / "CHECKSUMS.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="ascii",
    )


def validate_coupling_package(
    output: str | Path,
    *,
    protocol: CouplingProtocol,
    config_path: str | Path,
) -> CouplingPackageValidation:
    """Validate checksums, parent binding, and diagnostic package digests."""

    if type(protocol) is not CouplingProtocol:
        raise CouplingArtifactError("issued coupling protocol is required")
    root = Path(output).resolve(strict=True)
    config = Path(config_path).resolve(strict=True)
    if not root.is_dir() or _sha256(config) != protocol.config_sha256:
        raise CouplingArtifactError("coupling package config changed")
    names = {path.name for path in root.iterdir() if path.is_file()}
    if names != REQUIRED_COUPLING_FILES:
        raise CouplingArtifactError("coupling package file roster is invalid")
    checksums: dict[str, str] = {}
    try:
        for line in (root / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines():
            digest, name = line.split("  ", 1)
            if name in checksums or "/" in name or "\\" in name:
                raise ValueError
            checksums[name] = _hash_text(digest, "artifact checksum")
    except (OSError, UnicodeError, ValueError) as error:
        raise CouplingArtifactError("checksum registry is invalid") from error
    if set(checksums) != names - {"CHECKSUMS.sha256"}:
        raise CouplingArtifactError("checksum registry does not cover the package")
    for name, digest in checksums.items():
        if _sha256(root / name) != digest:
            raise CouplingArtifactError(f"checksum mismatch: {name}")
    manifest = _json_file(root / "artifact_manifest.json")
    summary = _json_file(root / "summary.json")
    if (
        manifest.get("config_sha256") != protocol.config_sha256
        or manifest.get("parent_scientific_digest")
        != protocol.parent_scientific_digest
        or manifest.get("parent_output_tree_sha256")
        != protocol.parent_output_tree_sha256
        or manifest.get("diagnostic_state_sha256")
        != summary.get("diagnostic_state_sha256")
        or manifest.get("scientific_digest") != _scientific_digest(root)
        or manifest.get("output_tree_sha256") != _tree_digest(root)
    ):
        raise CouplingArtifactError("coupling package digest validation failed")
    if (
        summary.get("coupling_status")
        not in {"EXPLORATORY_SIGNAL", "NO_CONSISTENT_SIGNAL"}
        or summary.get("validation_status") != "NOT_VALIDATED_POST_HOC"
        or summary.get("s2_status") != "NOT_RUN_NOT_AUTHORIZED"
    ):
        raise CouplingArtifactError("coupling package status is invalid")
    return CouplingPackageValidation(
        coupling_status=str(summary["coupling_status"]),
        validation_status=str(summary["validation_status"]),
        s2_status=str(summary["s2_status"]),
        diagnostic_state_sha256=str(summary["diagnostic_state_sha256"]),
        scientific_digest=str(manifest["scientific_digest"]),
        output_tree_sha256=str(manifest["output_tree_sha256"]),
    )


def publish_coupling_package(
    output: str | Path,
    *,
    protocol: CouplingProtocol,
    diagnostic: CouplingDiagnostic,
    config_path: str | Path,
) -> CouplingPackageValidation:
    """Publish one coupling diagnostic by atomic non-overwriting rename."""

    if type(protocol) is not CouplingProtocol or type(diagnostic) is not CouplingDiagnostic:
        raise CouplingArtifactError("issued protocol and diagnostic are required")
    destination = Path(output).resolve()
    if destination.exists():
        raise CouplingArtifactError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        _write_package(staging, protocol=protocol, diagnostic=diagnostic)
        validation = validate_coupling_package(
            staging, protocol=protocol, config_path=config_path
        )
        os.replace(staging, destination)
        return validation
    except CouplingArtifactError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except (OSError, TypeError, ValueError) as error:
        shutil.rmtree(staging, ignore_errors=True)
        raise CouplingArtifactError("coupling package publication failed") from error


def build_coupling_diagnostic(
    protocol: CouplingProtocol, *, project_root: str | Path
) -> CouplingDiagnostic:
    """Bind the formal S1 parent to structural authorities and diagnose groups."""

    if type(protocol) is not CouplingProtocol:
        raise CouplingArtifactError("issued coupling protocol is required")
    root = Path(project_root).resolve(strict=True)
    parent = load_parent_candidate_errors(protocol, project_root=root)
    msss_protocol = load_protocol(protocol.msss_config, project_root=root)
    authority = load_authority(msss_protocol, project_root=root)
    if (
        parent.specimen_ids != authority.specimen_ids
        or parent.dataset_ids != authority.dataset_ids
        or authority.data.scalar_internal3.shape != (276, 3)
    ):
        raise CouplingArtifactError("parent predictions and structural authority differ")
    damage = np.asarray(authority.data.scalar_internal3, dtype=np.float64)
    try:
        return diagnose_coupling(
            conditions=parent.conditions,
            specimen_ids=parent.specimen_ids,
            dataset_ids=parent.dataset_ids,
            ply_count=authority.ply_count,
            layup_family=authority.layup_family,
            damage_sizes={
                "damage_area": damage[:, 0],
                "damage_height": damage[:, 1],
                "damage_width": damage[:, 2],
            },
            absolute_errors=parent.absolute_errors,
            margin=protocol.margin,
        )
    except ValueError as error:
        raise CouplingArtifactError("coupling diagnostic computation failed") from error


def replay_coupling_package(
    source: str | Path,
    destination: str | Path,
    *,
    protocol: CouplingProtocol,
    project_root: str | Path,
    config_path: str | Path,
) -> CouplingPackageValidation:
    """Recompute the diagnostic from its parent before atomic replay publication."""

    source_root = Path(source).resolve(strict=True)
    output = Path(destination).resolve()
    if output.exists():
        raise CouplingArtifactError(f"output already exists: {output}")
    source_validation = validate_coupling_package(
        source_root, protocol=protocol, config_path=config_path
    )
    diagnostic = build_coupling_diagnostic(protocol, project_root=project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    container = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.replay-", dir=output.parent)
    )
    staged = container / "package"
    try:
        replay_validation = publish_coupling_package(
            staged,
            protocol=protocol,
            diagnostic=diagnostic,
            config_path=config_path,
        )
        if replay_validation != source_validation:
            raise CouplingArtifactError("coupling replay digest changed")
        os.replace(staged, output)
        container.rmdir()
        return replay_validation
    except CouplingArtifactError:
        shutil.rmtree(container, ignore_errors=True)
        raise
    except OSError as error:
        shutil.rmtree(container, ignore_errors=True)
        raise CouplingArtifactError("coupling replay publication failed") from error


__all__ = [
    "REQUIRED_COUPLING_FILES",
    "CouplingArtifactError",
    "CouplingPackageValidation",
    "CouplingProtocol",
    "ParentCandidateEvidence",
    "build_coupling_diagnostic",
    "load_coupling_protocol",
    "load_parent_candidate_errors",
    "publish_coupling_package",
    "replay_coupling_package",
    "validate_coupling_package",
]
