"""Download-cache-only external pairing and raw PA feasibility manifests."""

from __future__ import annotations

import csv
import hashlib
import json
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image


@cache
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("external manifest cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _rss(cache: Path, output: Path) -> list[dict[str, object]]:
    root = cache / "rss"
    prefix = root / "Data Repository Data"
    mappings = (
        ("AP-3", "APply_1.csv", "ap-3_35j_1_Spec_1_Ch_1.csv", "AP-ply"),
        ("B24-4", "Baseline.csv", "b24-4_35j_1_Spec_1_Ch_1.csv", "Baseline"),
        (
            "HBI-1",
            "Hybrid RSS inner.csv",
            "hbi-1_35j_1_Spec_1_Ch_1.csv",
            "Hybrid RSS inner",
        ),
        (
            "HBO-1",
            "Hybrid RSS outer.csv",
            "hbo-1_35j_1_Spec_1_Ch_1.csv",
            "Hybrid RSS outer",
        ),
        (
            "SP2-3",
            "RSS min amplitude x2.csv",
            "sp2-3_35j_1_Spec_1_Ch_1.csv",
            "RSS minimum amplitude",
        ),
        (
            "SP4-3",
            "RSS int amplitude x4.csv",
            "sp4-3_35j_1_Spec_1_Ch_1.csv",
            "RSS intermediate amplitude",
        ),
        (
            "SP8-3",
            "RSS max amplitude x8.csv",
            "sp8-3_35j_1_Spec_1_Ch_1.csv",
            "RSS maximum amplitude",
        ),
    )
    containers = tuple(sorted((prefix / "LVI/cscans").glob("*.txt")))
    if len(containers) != 2:
        raise ValueError("RSS C-scan container roster changed")
    rows: list[dict[str, object]] = []
    for specimen, cai_name, impact_name, group in mappings:
        cai = prefix / "CAI" / cai_name
        impact = prefix / "LVI" / impact_name
        if not cai.is_file() or not impact.is_file():
            raise ValueError("RSS specimen roster changed")
        rows.append(
            {
                "specimen_id": specimen,
                "cscan_path": "|".join(_relative(path, root) for path in containers),
                "cscan_sha256": "|".join(_sha256(path) for path in containers),
                "cai_raw_path": _relative(cai, root),
                "cai_raw_sha256": _sha256(cai),
                "impact_raw_path": _relative(impact, root),
                "impact_raw_sha256": _sha256(impact),
                "material": "not documented in downloaded archive",
                "layup_or_group": group,
                "impact_condition": "35 J LVI (from specimen filenames)",
                "specimen_geometry": "not documented in downloaded archive",
                "possible_cai_target": "peak force from Force(kN) column",
                "pairing_status": "UNRESOLVED_CSCAN_SPECIMEN_ROI",
                "paired_cscan_cai": False,
                "license": "CC BY 4.0",
                "role": "SEALED_POTENTIAL_PAIR_PENDING_ROI_MAP",
            }
        )
    _write_csv(output / "imperial_rss_manifest.csv", rows)
    return rows


def _interlock(cache: Path, output: Path) -> list[dict[str, object]]:
    root = cache / "interlock"
    base = root / "Compression after impact"
    rows: list[dict[str, object]] = []
    for group, prefix in (("Baseline", "BL"), ("Reinforced interlock", "RE")):
        for serial in range(7, 12):
            specimen = f"{prefix}-{serial:03d}"
            cscans = tuple((base / "C-Scans").glob(f"{specimen} Post* C-Scan.csv"))
            compression = base / "Compression Tests" / f"{specimen} Compression.csv"
            if len(cscans) != 1 or not compression.is_file():
                raise ValueError("Interlock pair roster changed")
            with compression.open("r", encoding="utf-8-sig", newline="") as handle:
                records = list(csv.reader(handle))
            pmax = next(float(row[1]) for row in records if row and row[0] == "Pmax")
            rows.append(
                {
                    "specimen_id": specimen,
                    "cscan_path": _relative(cscans[0], root),
                    "cscan_sha256": _sha256(cscans[0]),
                    "cai_raw_path": _relative(compression, root),
                    "cai_raw_sha256": _sha256(compression),
                    "material": "Skyflex USN 020A T700/K51 thin ply + MTM28-1/T800",
                    "layup_or_group": group,
                    "impact_condition": "ASTM D7136; 6.7 J/mm nominal thickness",
                    "specimen_geometry": "approximately 150 x 100 x 4.5 mm; exact dimensions in archive",
                    "possible_cai_target": "Pmax_kN",
                    "pmax_kN": pmax,
                    "pairing_status": "EXACT_SPECIMEN_ID",
                    "paired_cscan_cai": True,
                    "license": "CC BY 4.0",
                    "role": "SEALED_SMALL_EXTERNAL_PILOT",
                }
            )
    _write_csv(output / "imperial_interlock_manifest.csv", rows)
    return rows


def _tudelft(cache: Path, output: Path) -> list[dict[str, object]]:
    root = cache / "tudelft"
    rows: list[dict[str, object]] = []
    for index in range(1, 4):
        cscans = tuple(sorted(root.glob(f"CAI-spec{index}-Cscan*.jpg")))
        force = root / f"CAIspec{index}_mts.csv"
        if len(cscans) != 2 or not force.is_file():
            raise ValueError("TU Delft pair roster changed")
        rows.append(
            {
                "specimen_id": f"CAI_spec{index}",
                "cscan_path": "|".join(path.name for path in cscans),
                "cscan_sha256": "|".join(_sha256(path) for path in cscans),
                "cai_raw_path": force.name,
                "cai_raw_sha256": _sha256(force),
                "material": "Toray M30SC / Deltapreg DT120-200-36 UD",
                "layup_or_group": "[-45,0,45,90]4s",
                "impact_condition": "ASTM D7136; target 34 J",
                "specimen_geometry": "150 x 100 x 5.15 mm",
                "possible_cai_target": "peak absolute force from force(kN) column",
                "pairing_status": "EXACT_SPECIMEN_ID_TWO_CSCAN_IMAGES",
                "paired_cscan_cai": True,
                "license": "CC0",
                "role": "MICRO_CASE_VALIDATION_ONLY",
            }
        )
    _write_csv(output / "tudelft_manifest.csv", rows)
    return rows


def _raw_header(path: Path) -> tuple[str, int, int, int]:
    data_type = ""
    declared_frames = 0
    observed_frames = 0
    buckets = 0
    focal_laws = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if line.startswith("#Data Type,") and not data_type:
                data_type = line.rstrip().split(",", 1)[1]
            elif line.startswith("#Frame,"):
                fields = line.rstrip().split(",")
                observed_frames += 1
                if not declared_frames:
                    declared_frames = int(fields[3])
            elif line.startswith("#Spec,") and not buckets:
                fields = line.rstrip().split(",")
                buckets = int(fields[2])
                focal_laws = int(fields[4])
    if (
        not data_type
        or declared_frames != observed_frames
        or buckets not in {504, 836}
        or focal_laws != 57
    ):
        raise ValueError("Cranfield raw PA header changed")
    return data_type, observed_frames, buckets, focal_laws


def _cranfield(
    cache: Path, output: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    root = cache / "cranfield" / "Dataset_WP2_cranfield"
    raw_files = tuple(sorted(root.rglob("*.csv")))
    raw_rows: list[dict[str, object]] = []
    for path in raw_files:
        data_type, frames, buckets, focal_laws = _raw_header(path)
        frequency = "5MHz" if "Wheel_Probe" in path.as_posix() else "10MHz"
        category = (
            next(
                value
                for value in ("Coupons", "Laminates", "Repaired_Patch")
                if value.lower() in path.as_posix().lower()
            )
            if frequency == "10MHz"
            else "Coupons"
        )
        raw_rows.append(
            {
                "raw_id": path.stem,
                "frequency": frequency,
                "category": category,
                "raw_path": _relative(path, root),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "data_type": data_type,
                "frame_count": frames,
                "waveform_buckets": buckets,
                "focal_law_count": focal_laws,
                "tensor_shape": f"{frames}x{buckets}x{focal_laws}",
                "indexed_spatial_locations": frames * focal_laws,
                "physical_spacing_status": (
                    "150 mm scan stated; exact coordinate vector absent"
                    if frequency == "5MHz"
                    else "coordinate vector and spacing absent"
                ),
            }
        )
    if len(raw_rows) != 29:
        raise ValueError("Cranfield raw file count changed")
    _write_csv(output / "cranfield_wp2/raw_file_manifest.csv", raw_rows)

    processed = tuple(sorted((*root.rglob("*.png"), *root.rglob("*.tif"))))
    pair_rows: list[dict[str, object]] = []
    for image_path in processed:
        stem = image_path.stem.replace("_2D_colored", "")
        if "_ABC" in stem:
            raw = tuple(
                sorted(
                    root.joinpath(
                        "Raw_CSV_C_Scan_Impact_Damages_On_Diffrent_Specimens",
                        "Repaired_Patch",
                    ).glob(stem.replace("_ABC", "_?") + ".csv")
                )
            )
        elif "_AB" in stem:
            raw = tuple(
                sorted(
                    root.joinpath(
                        "Raw_CSV_C_Scan_Impact_Damages_On_Diffrent_Specimens",
                        "Laminates",
                    ).glob(stem.replace("_AB", "_?") + ".csv")
                )
            )
        else:
            raw = tuple(path for path in raw_files if path.stem == stem)
        if not raw:
            raise ValueError(f"Cranfield processed/raw pair missing: {image_path.name}")
        with Image.open(image_path) as image:
            width, height = image.size
        pair_rows.append(
            {
                "scan_id": image_path.stem,
                "processed_path": _relative(image_path, root),
                "processed_sha256": _sha256(image_path),
                "processed_width": width,
                "processed_height": height,
                "raw_paths": "|".join(_relative(path, root) for path in raw),
                "raw_file_count": len(raw),
                "pairing_status": (
                    "EXACT_STEM"
                    if len(raw) == 1
                    else "EXPLICIT_A_B_COMPONENTS"
                    if len(raw) == 2
                    else "EXPLICIT_A_B_C_COMPONENTS"
                ),
            }
        )
    if len(pair_rows) != 26 or sum(row["raw_file_count"] for row in pair_rows) != 29:
        raise ValueError("Cranfield processed pair count changed")
    _write_csv(output / "cranfield_wp2/scan_pair_manifest.csv", pair_rows)
    return raw_rows, pair_rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _checksums(directory: Path) -> None:
    files = tuple(
        sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.name != "CHECKSUMS.sha256"
        )
    )
    text = "".join(
        f"{_sha256(path)}  {path.relative_to(directory).as_posix()}\n" for path in files
    )
    (directory / "CHECKSUMS.sha256").write_text(text, encoding="ascii")


def generate_external_audits(cache_root: str | Path, output_root: str | Path) -> Path:
    """Generate evidence-only manifests; no model or acquisition method is run."""

    cache = Path(cache_root).resolve(strict=True)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rss = _rss(cache, output)
    interlock = _interlock(cache, output)
    tudelft = _tudelft(cache, output)
    raw_rows, pair_rows = _cranfield(cache, output)
    archive_hashes = {
        name: _sha256(cache / filename)
        for name, filename in (
            ("imperial_rss", "rss.zip"),
            ("imperial_interlock", "interlock.zip"),
            ("cranfield_wp2", "cranfield_wp2.zip"),
        )
    }
    _write_json(
        output / "EXTERNAL_DATA_MANIFEST.json",
        {
            "schema_version": 1,
            "discipline": "download_unpack_hash_inspect_pair_count_document_only",
            "method_performance_present": False,
            "datasets": {
                "imperial_rss": {
                    "doi": "10.17632/wg4dmwddjy.2",
                    "archive_sha256": archive_hashes["imperial_rss"],
                    "license": "CC BY 4.0",
                    "exact_paired_cscan_cai_n": sum(
                        bool(row["paired_cscan_cai"]) for row in rss
                    ),
                    "potential_filename_linked_n": len(rss),
                    "role": "SEALED_POTENTIAL_PAIR_PENDING_ROI_MAP",
                },
                "imperial_interlock": {
                    "doi": "10.5281/zenodo.1476887",
                    "archive_sha256": archive_hashes["imperial_interlock"],
                    "license": "CC BY 4.0",
                    "exact_paired_cscan_cai_n": len(interlock),
                    "groups": {"baseline": 5, "reinforced": 5},
                    "role": "SEALED_SMALL_EXTERNAL_PILOT",
                },
                "tudelft": {
                    "doi": "10.4121/21621381",
                    "license": "CC0",
                    "exact_paired_cscan_cai_n": len(tudelft),
                    "role": "MICRO_CASE_VALIDATION_ONLY",
                },
                "cranfield_wp2": {
                    "doi": "10.5281/zenodo.4405277",
                    "archive_sha256": archive_hashes["cranfield_wp2"],
                    "raw_file_n": len(raw_rows),
                    "processed_scan_pair_n": len(pair_rows),
                    "method_performance_present": False,
                    "role": "RAW_ACQUISITION_REALIZABILITY_AUDIT_ONLY",
                },
            },
        },
    )
    _write_json(
        output / "cranfield_wp2/grid_schema.json",
        {
            "schema_version": 1,
            "csv_block": {
                "frame_header": "#Frame,<index>,of,<count>",
                "spec_header": "#Spec,<Buckets|Amplitudes>,<504|836>,FL,57",
                "tensor_axes": ["frame_index", "waveform_bucket", "focal_law_index"],
                "frame_counts_observed": sorted(
                    {row["frame_count"] for row in raw_rows}
                ),
                "waveform_buckets_observed": sorted(
                    {row["waveform_buckets"] for row in raw_rows}
                ),
                "focal_laws": 57,
            },
            "physical_acquisition_record": "one frame/focal-law spatial index with the file-declared 504 or 836 amplitude samples",
            "spatial_grid_recoverable": True,
            "physical_coordinate_spacing_recoverable": False,
            "normalized_8x8_mapping": "partition frame and focal-law index axes with endpoint-preserving rounded linspace boundaries",
            "sparse_mask_definition": "subset of unique (frame_index, focal_law_index) locations retaining every waveform bucket",
            "exact_measurement_fraction": "selected unique frame/focal-law pairs divided by frame_count*57",
            "scanner_time_claim_authorized": False,
        },
    )
    example_raw = next(
        row
        for row in raw_rows
        if row["frequency"] == "5MHz"
        and row["frame_count"] == 150
        and row["waveform_buckets"] == 504
    )
    frame_count, focal_laws = 150, 57
    frame_boundaries = np.rint(np.linspace(0, frame_count - 1, 9)).astype(int)
    law_boundaries = np.rint(np.linspace(0, focal_laws - 1, 9)).astype(int)
    row, column = 3, 4
    _write_json(
        output / "cranfield_wp2/example_mapping.json",
        {
            "raw_example": example_raw["raw_path"],
            "frame_count": frame_count,
            "focal_law_count": focal_laws,
            "normalized_cell_index": row * 8 + column,
            "cell_row": row,
            "cell_column": column,
            "frame_index_inclusive": [
                int(frame_boundaries[row]),
                int(frame_boundaries[row + 1]),
            ],
            "focal_law_index_inclusive": [
                int(law_boundaries[column]),
                int(law_boundaries[column + 1]),
            ],
            "waveform_bucket_indices": [
                0,
                int(example_raw["waveform_buckets"]) - 1,
            ],
            "physical_mm_mapping": None,
        },
    )
    _checksums(output / "cranfield_wp2")
    _checksums(output)
    return output


__all__ = ["generate_external_audits"]
