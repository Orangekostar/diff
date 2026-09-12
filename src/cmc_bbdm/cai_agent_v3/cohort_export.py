"""Build the complete capture-group-isolated INTERNAL_REUSED_COHORT_V3."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

from .cohort import assign_capture_group_splits, build_capture_groups
from .files import read_csv, write_csv, write_json


def _load_author_labels(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for row in rows:
        if (
            row["measurement_semantics"] != "CAI_STRENGTH"
            or row["unit_normalized"] != "MPa"
            or row["source_kind"] != "AUTHOR_WORKBOOK"
        ):
            continue
        key = row["specimen_key"]
        value = float(row["value_numeric"])
        if key in labels or not math.isfinite(value) or value <= 0.0:
            raise ValueError("author CAI MPa source is duplicated or invalid")
        labels[key] = row
    return labels


def _load_oof_labels(rows: list[dict[str, str]]) -> dict[str, float]:
    labels: dict[str, float] = {}
    for row in rows:
        key = f"{row['domain_id']}:{row['specimen_id']}"
        if key in labels:
            raise ValueError("v2 label cross-check contains duplicate identity")
        labels[key] = float(row["cai_strength_mpa"])
    return labels


def export_cohort(*, project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    output = root / "results/cai_agent_v3/new_protocol"
    p0r = read_csv(
        root
        / "results/agentic_task_driven_nde/p0r_author_registration/surface_manifest.csv"
    )
    author_labels = _load_author_labels(
        read_csv(root / "results/hasebe_reference_evidence/v1/author_measurements.csv")
    )
    old_labels = _load_oof_labels(
        read_csv(root / "results/multiview/e1_audit/oof_predictions.csv")
    )
    old_split_keys = {
        row["specimen_key"]
        for row in read_csv(root / "results/cai_active_image/v2/split_manifest.csv")
    }
    if len(p0r) != 276:
        raise ValueError(f"P0R authorized candidate count changed: {len(p0r)}")
    capture_groups = build_capture_groups(p0r)
    candidate_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []
    for row in sorted(p0r, key=lambda item: (item["dataset_id"], item["specimen_id"])):
        key = f"{row['dataset_id']}:{row['specimen_id']}"
        reasons: list[str] = []
        if row["p0r_roster_status"] != "AUTHORIZED":
            reasons.append("P0R_NOT_AUTHORIZED")
        if row["identity_status"] != "PASS_EXACT_SPECIMEN_ID_AND_HASH":
            reasons.append("P0R_IDENTITY_NOT_EXACT")
        label = author_labels.get(key)
        if label is None:
            reasons.append("AUTHOR_CAI_MPA_MISSING")
        if reasons:
            exclusion_rows.append(
                {
                    "specimen_key": key,
                    "dataset_id": row["dataset_id"],
                    "specimen_id": row["specimen_id"],
                    "exclusion_reasons": ";".join(reasons),
                }
            )
            continue
        candidate_rows.append(
            {
                "specimen_key": key,
                "dataset_id": row["dataset_id"],
                "dataset_version": row["dataset_version"],
                "specimen_id": row["specimen_id"],
                "capture_group_id": capture_groups[key],
                "author_cai_mpa": float(label["value_numeric"]),
                "label_source_path": label["source_path_or_url"],
                "label_source_sheet": label["source_sheet"],
                "label_source_cell": label["source_cell"],
                "label_source_sha256": label["source_sha256"],
                "label_source_dataset_version": label["source_dataset_version"],
                "cscan_source_path": row["cscan_source_path"],
                "cscan_source_sha256": row["cscan_source_sha256"],
                "cscan_panel_index": row["cscan_panel_index"],
                "registered_cscan_crop_path": row["registered_cscan_crop_path"],
                "registered_cscan_crop_sha256": row["registered_cscan_crop_sha256"],
                "registered_cscan_height_px": row["registered_cscan_height_px"],
                "registered_cscan_width_px": row["registered_cscan_width_px"],
                "impacted_surface_path": row["impacted_surface_path"],
                "surface_sha256": row["surface_sha256"],
                "identity_status": row["identity_status"],
                "p0r_roster_status": row["p0r_roster_status"],
                "v2_member": key in old_split_keys,
            }
        )
    split_by_key = assign_capture_group_splits(candidate_rows)
    for row in candidate_rows:
        row["split"] = split_by_key[str(row["specimen_key"])]

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidate_rows:
        groups[str(row["capture_group_id"])].append(row)
    group_rows: list[dict[str, object]] = []
    for group_id, members in sorted(groups.items()):
        domains = {str(row["dataset_id"]) for row in members}
        splits = {str(row["split"]) for row in members}
        if len(domains) != 1 or len(splits) != 1:
            raise ValueError("capture group was split or crosses domains")
        group_rows.append(
            {
                "capture_group_id": group_id,
                "dataset_id": next(iter(domains)),
                "split": next(iter(splits)),
                "physical_specimen_count": len(members),
                "specimen_keys": ";".join(
                    sorted(str(row["specimen_key"]) for row in members)
                ),
                "cscan_source_sha256": ";".join(
                    sorted({str(row["cscan_source_sha256"]) for row in members})
                ),
                "surface_sha256": ";".join(
                    sorted({str(row["surface_sha256"]) for row in members})
                ),
            }
        )
    split_rows = [
        {
            "specimen_key": row["specimen_key"],
            "dataset_id": row["dataset_id"],
            "specimen_id": row["specimen_id"],
            "capture_group_id": row["capture_group_id"],
            "split": row["split"],
            "v2_member": row["v2_member"],
        }
        for row in candidate_rows
    ]
    crosscheck_rows = []
    for row in candidate_rows:
        key = str(row["specimen_key"])
        previous = old_labels.get(key)
        author = float(row["author_cai_mpa"])
        crosscheck_rows.append(
            {
                "specimen_key": key,
                "author_cai_mpa": author,
                "v2_oof_label_mpa": "" if previous is None else previous,
                "difference_mpa": "" if previous is None else previous - author,
                "status": (
                    "NOT_IN_V2_LABEL_TABLE"
                    if previous is None
                    else "MATCH"
                    if math.isclose(previous, author, rel_tol=0.0, abs_tol=1e-9)
                    else "MISMATCH"
                ),
            }
        )
    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (str(row["dataset_id"]), str(row["specimen_id"])),
    )
    write_csv(output / "candidate_queue.csv", candidate_rows)
    write_csv(
        output / "exclusions.csv",
        exclusion_rows,
        fieldnames=(
            "specimen_key",
            "dataset_id",
            "specimen_id",
            "exclusion_reasons",
        ),
    )
    write_csv(output / "capture_groups.csv", group_rows)
    write_csv(output / "split_manifest.csv", split_rows)
    write_csv(output / "label_crosscheck.csv", crosscheck_rows)
    split_counts = Counter(str(row["split"]) for row in candidate_rows)
    group_split_counts = Counter(str(row["split"]) for row in group_rows)
    domain_split_counts = {
        domain: dict(
            sorted(
                Counter(
                    str(row["split"])
                    for row in candidate_rows
                    if row["dataset_id"] == domain
                ).items()
            )
        )
        for domain in sorted({str(row["dataset_id"]) for row in candidate_rows})
    }
    summary = {
        "protocol": "INTERNAL_REUSED_COHORT_V3",
        "status": "COHORT_READY",
        "authorized_candidate_count": len(p0r),
        "included_specimen_count": len(candidate_rows),
        "excluded_specimen_count": len(exclusion_rows),
        "capture_group_count": len(group_rows),
        "multi_specimen_capture_group_count": sum(
            int(row["physical_specimen_count"]) > 1 for row in group_rows
        ),
        "split_specimen_counts": dict(sorted(split_counts.items())),
        "split_capture_group_counts": dict(sorted(group_split_counts.items())),
        "domain_split_specimen_counts": domain_split_counts,
        "v2_member_count": sum(bool(row["v2_member"]) for row in candidate_rows),
        "label_source": "AUTHOR_WORKBOOK CAI_STRENGTH MPa",
        "v2_oof_label_crosscheck_counts": dict(
            sorted(Counter(row["status"] for row in crosscheck_rows).items())
        ),
        "test_role": "INTERNAL_REUSED_COHORT_V3_NOT_UNTOUCHED_CONFIRMATION",
    }
    write_json(output / "cohort_summary.json", summary)
    return summary


__all__ = ["export_cohort"]
