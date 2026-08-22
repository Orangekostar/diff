#!/usr/bin/env python3
"""Run or replay the registered MGMR M0 component gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from cmc_bbdm.cpb_v3.embeddings import encode_resnet18
from cmc_bbdm.mgmr.artifacts import publish_m0_package
from cmc_bbdm.mgmr.authority import load_authority
from cmc_bbdm.mgmr.feature_bank import (
    load_feature_bank,
    publish_feature_bank,
)
from cmc_bbdm.mgmr.formal_outer import evaluate_formal_outer
from cmc_bbdm.mgmr.m0_components import (
    evaluate_components,
    load_registered_b0,
)
from cmc_bbdm.mgmr.m0_residual_audit import audit_residual_arrays
from cmc_bbdm.mgmr.protocol import load_protocol
from cmc_bbdm.mgmr.replay import replay_m0_package
from cmc_bbdm.mgmr.spatial_encoder import extract_m0_feature_bank
from cmc_bbdm.mgmr.specificity_bank import (
    extract_specificity_bank,
    load_specificity_bank,
    publish_specificity_bank,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTERED_CONFIG = ROOT / "paper_v3/configs/mgmr_m0.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--replay", action="store_true")
    return parser


def _status(message: str) -> None:
    print(f"[MGMR M0] {message}", file=sys.stderr, flush=True)


def _manifest_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _combined_feature_sha(primary: str, specificity: str) -> str:
    return hashlib.sha256(f"{primary}\0{specificity}".encode("ascii")).hexdigest()


def run_registered_m0(*, config_path: Path, device: str, replay: bool) -> dict[str, object]:
    if config_path.resolve(strict=True) != REGISTERED_CONFIG.resolve(strict=True):
        raise ValueError("only paper_v3/configs/mgmr_m0.yaml is registered")
    protocol = load_protocol(config_path, project_root=ROOT)
    if device != protocol.device:
        raise ValueError("runtime device differs from the registered protocol")
    formal_path = ROOT / protocol.output_paths["formal"]
    replay_path = ROOT / protocol.output_paths["replay"]
    if replay:
        _status("validating and replaying the formal package")
        validation = replay_m0_package(
            formal_path,
            replay_path,
            project_root=ROOT,
            config_path=config_path,
        )
        return {
            "mode": "replay",
            "status": validation.status,
            "scientific_digest": validation.scientific_digest,
            "output_tree_sha256": validation.output_tree_sha256,
            "formal_path": str(protocol.output_paths["formal"]),
            "replay_path": str(protocol.output_paths["replay"]),
        }

    _status("loading checksum-bound cohort authority")
    authority = load_authority(protocol, project_root=ROOT)
    feature_path = ROOT / protocol.output_paths["feature_bank"]
    specificity_path = feature_path.parent / f"{feature_path.name}_p3"
    if feature_path.exists():
        _status("validating cached FULL/coarse feature bank")
        feature_manifest_sha = _manifest_sha(feature_path / "manifest.json")
        feature_bank = load_feature_bank(
            feature_path,
            expected_manifest_sha256=feature_manifest_sha,
            expected_specimen_ids=authority.specimen_ids,
            expected_dataset_ids=authority.dataset_ids,
            expected_config_sha256=protocol.config_sha256,
        )
    else:
        _status("extracting FULL/coarse feature bank")
        extraction = extract_m0_feature_bank(
            protocol,
            authority,
            project_root=ROOT,
            device=device,
            status_hook=_status,
        )
        publication = publish_feature_bank(feature_path, extraction.bank)
        feature_manifest_sha = publication.manifest_sha256
        feature_bank = extraction.bank

    if specificity_path.exists():
        _status("validating cached P3 directional feature bank")
        specificity_manifest_sha = _manifest_sha(
            specificity_path / "manifest.json"
        )
        specificity_bank = load_specificity_bank(
            specificity_path,
            expected_manifest_sha256=specificity_manifest_sha,
            expected_specimen_ids=authority.specimen_ids,
            expected_dataset_ids=authority.dataset_ids,
            expected_config_sha256=protocol.config_sha256,
        )
    else:
        _status("extracting P3 directional feature bank")
        encoder = encode_resnet18(
            weight_path=protocol.sources["resnet_weights"].path,
            project_root=ROOT,
            device=device,
            batch_size=protocol.batch_size,
        )
        specificity_bank = extract_specificity_bank(
            protocol,
            authority,
            encoder,
            feature_bank_state_sha256=feature_bank.state_sha256,
            status_hook=_status,
        )
        publication = publish_specificity_bank(specificity_path, specificity_bank)
        specificity_manifest_sha = publication.manifest_sha256

    _status("validating exact registered B0")
    baseline = load_registered_b0(protocol, authority, project_root=ROOT)
    _status("evaluating direct B0-B4 component roster")
    components = evaluate_components(protocol, authority, feature_bank, baseline)
    _status("running strict source-domain OOF residual and P3 audits")
    residual = audit_residual_arrays(
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        domain_order=protocol.domain_order,
        targets=authority.targets,
        metadata=authority.metadata13,
        full=feature_bank.full_global,
        coarse=feature_bank.coarse_gap,
        boundary=feature_bank.full_directional,
        coarse_outer_predictions=components.predictions["B1"],
        full_outer_predictions=baseline.predictions,
        shuffled_boundary=specificity_bank.directional,
        pca_dimensions=protocol.pca_dimensions,
        ridge_alpha=protocol.ridge_alpha,
        tie_tolerance=protocol.pca_tie_tolerance,
    )
    _status("deriving metrics, bootstrap intervals, and frozen gate")
    formal = evaluate_formal_outer(protocol, components, residual)
    combined_manifest = _combined_feature_sha(
        feature_manifest_sha, specificity_manifest_sha
    )
    _status("publishing and independently validating formal artifacts")
    validation = publish_m0_package(
        formal_path,
        protocol=protocol,
        formal=formal,
        feature_manifest_sha256=combined_manifest,
    )
    return {
        "mode": "formal",
        "status": validation.status,
        "gates": dict(formal.decision.gates),
        "scientific_digest": validation.scientific_digest,
        "output_tree_sha256": validation.output_tree_sha256,
        "feature_manifest_sha256": feature_manifest_sha,
        "specificity_manifest_sha256": specificity_manifest_sha,
        "formal_path": str(protocol.output_paths["formal"]),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_registered_m0(
            config_path=args.config, device=args.device, replay=args.replay
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"MGMR M0 failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
