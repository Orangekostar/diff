from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from cmc_bbdm.agentic_nde.author_authority import (
    build_author_registration_authority,
)
from cmc_bbdm.agentic_nde.contracts import PRIMARY_COUNTS, P0RGateFacts, decide_p0r
from cmc_bbdm.agentic_nde.p0r_artifacts import write_p0r_package
from cmc_bbdm.agentic_nde.p0r_qc import render_p0r_qc

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(tmp_path: Path) -> tuple[Path, Path]:
    external = tmp_path / "external"
    data = external / "data"
    data.mkdir(parents=True)
    surface = data / "surface.png"
    crop = data / "crop.png"
    surface_image = Image.new("RGB", (120, 80), "white")
    surface_draw = ImageDraw.Draw(surface_image)
    surface_draw.rectangle((8, 6, 54, 35), fill=(180, 30, 50))
    surface_draw.ellipse((72, 38, 111, 74), fill=(25, 105, 175))
    surface_image.save(surface)
    crop_image = surface_image.transpose(Image.Transpose.ROTATE_270).resize((80, 120))
    crop_image.save(crop)
    surface_hash = _sha256(surface)
    crop_hash = _sha256(crop)

    surface_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    registration_rows: list[dict[str, object]] = []
    registration_qc_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    for domain in PRIMARY_COUNTS:
        for index in range(3):
            specimen = f"qc-{index}"
            transform_hash = hashlib.sha256(
                f"{domain}/{specimen}".encode("ascii")
            ).hexdigest()
            identity = {"dataset_id": domain, "specimen_id": specimen}
            surface_rows.append(
                {
                    **identity,
                    "p0r_roster_status": "AUTHORIZED",
                    "impacted_surface_path": "data/surface.png",
                    "surface_sha256": surface_hash,
                }
            )
            provenance_rows.append(
                {
                    **identity,
                    "registered_cscan_crop_path": "data/crop.png",
                    "registered_crop_file_sha256": crop_hash,
                    "decoded_pixel_equal": True,
                }
            )
            registration_rows.append(
                {
                    **identity,
                    "status": "AUTHORIZED",
                    "orientation": "ROT90",
                    "source_width_px": 120,
                    "source_height_px": 80,
                    "destination_width_px": 80,
                    "destination_height_px": 120,
                    "transform_sha256": transform_hash,
                }
            )
            registration_qc_rows.append({**identity, "status": "PASS"})
            for cell_id in range(64):
                row, column = divmod(cell_id, 8)
                cscan_x0 = column * 79 / 8
                cscan_x1 = (column + 1) * 79 / 8
                cscan_y0 = row * 119 / 8
                cscan_y1 = (row + 1) * 119 / 8
                surface_x0 = row * 119 / 8
                surface_x1 = (row + 1) * 119 / 8
                surface_y0 = (7 - column) * 79 / 8
                surface_y1 = (8 - column) * 79 / 8
                grid_rows.append(
                    {
                        **identity,
                        "cell_id": cell_id,
                        "row": row,
                        "column": column,
                        "cscan_x0": cscan_x0,
                        "cscan_y0": cscan_y0,
                        "cscan_x1": cscan_x1,
                        "cscan_y1": cscan_y1,
                        "surface_x0": surface_x0,
                        "surface_y0": surface_y0,
                        "surface_x1": surface_x1,
                        "surface_y1": surface_y1,
                        "round_trip_status": "PASS",
                        "transform_sha256": transform_hash,
                    }
                )

    facts = P0RGateFacts(
        authorized_by_domain=dict(PRIMARY_COUNTS),
        exact_identity_hashes=True,
        author_statement_bound=True,
        global_orientation_rot90=True,
        all_panels_resolved=True,
        processing_provenance_deterministic=True,
        no_unsupported_rotation_reflection=True,
        composed_transform_replayable=True,
        no_result_driven_orientation=True,
        author_evidence_conflict=False,
        processing_provenance_unresolved=False,
    )
    decision = decide_p0r(facts)
    config_text = (
        ROOT / "paper_v3/configs/agentic_nde_p0r_author_registration.yaml"
    ).read_text(encoding="utf-8")
    package = write_p0r_package(
        tmp_path / "p0r",
        config_text=config_text,
        author_authority=build_author_registration_authority().as_dict(),
        surface_manifest=surface_rows,
        scan_processing_provenance=provenance_rows,
        registration=registration_rows,
        registration_qc=registration_qc_rows,
        grid_mapping_qc=grid_rows,
        summary={
            **decision.as_dict(),
            "gate_facts": facts.as_dict(),
            "authorized_registration_count": 276,
        },
        report="# Synthetic P0R\n",
    )
    return package, external


def _manifest_rows(output: Path) -> list[dict[str, str]]:
    with (output / "overlay_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        return list(csv.DictReader(stream))


def test_qc_renders_two_deterministic_four_panel_overlays_per_domain(
    tmp_path: Path,
) -> None:
    package, external = _package(tmp_path)
    before = {path.name: path.read_bytes() for path in package.iterdir()}

    output = render_p0r_qc(
        package=package,
        surface_root=external,
        output=tmp_path / "qc",
    )
    rows = _manifest_rows(output)

    assert len(rows) == 12
    assert {domain: sum(row["dataset_id"] == domain for row in rows) for domain in PRIMARY_COUNTS} == {
        domain: 2 for domain in PRIMARY_COUNTS
    }
    assert all(row["panel_count"] == "4" for row in rows)
    for row in rows:
        with Image.open(output / row["filename"]) as image:
            assert image.size == (960, 960)
            assert image.mode == "RGB"
    assert before == {path.name: path.read_bytes() for path in package.iterdir()}


def test_qc_output_is_byte_deterministic(tmp_path: Path) -> None:
    package, external = _package(tmp_path)

    left = render_p0r_qc(
        package=package,
        surface_root=external,
        output=tmp_path / "left",
    )
    right = render_p0r_qc(
        package=package,
        surface_root=external,
        output=tmp_path / "right",
    )

    assert {path.name: path.read_bytes() for path in left.iterdir()} == {
        path.name: path.read_bytes() for path in right.iterdir()
    }


def test_qc_selection_matches_seeded_sha256_order(tmp_path: Path) -> None:
    package, external = _package(tmp_path)
    output = render_p0r_qc(
        package=package,
        surface_root=external,
        output=tmp_path / "qc",
    )
    seed = yaml.safe_load((package / "config.yaml").read_text(encoding="utf-8"))[
        "qc"
    ]["selection_seed"]

    for domain in PRIMARY_COUNTS:
        expected = sorted(
            (hashlib.sha256(f"{seed}\0{domain}\0qc-{index}".encode()).hexdigest(), f"qc-{index}")
            for index in range(3)
        )[:2]
        actual = sorted(
            (row["selection_sha256"], row["specimen_id"])
            for row in _manifest_rows(output)
            if row["dataset_id"] == domain
        )
        assert actual == expected

