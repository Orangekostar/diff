from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cmc_bbdm.agentic_nde.contracts import (
    EvidenceClass,
    EvidenceRole,
    FrameGeometry,
    Orientation,
)
from cmc_bbdm.agentic_nde.grid import render_surface_grid
from cmc_bbdm.agentic_nde.p1 import load_p1_config
from cmc_bbdm.agentic_nde.registration import create_transform
from cmc_bbdm.agentic_nde.surface_cells import (
    SurfaceCellAuthority,
    SurfaceCellRecord,
)
from cmc_bbdm.agentic_nde.surface_encoder import (
    build_surface_resnet18,
    load_surface_feature_bank,
    materialize_surface_feature_bank,
    preprocess_surface_rgb,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/agentic_nde_p1_visual_observability.yaml"
TRANSFORM_SHA = "2b275ebbc220e6a0376d305d0996f4ffe80509fc8b27223fd919331a100acbe5"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_surface_rgb_preprocessing_is_exact_and_not_grayscale() -> None:
    image = Image.fromarray(
        np.full((5, 7, 3), (255, 0, 0), dtype=np.uint8)
    )
    output = preprocess_surface_rgb(image)
    assert output.shape == (3, 224, 224)
    assert output.dtype == np.dtype("<f4")
    assert np.all(np.isfinite(output))
    expected = np.asarray(
        [
            (1.0 - 0.485) / 0.229,
            (0.0 - 0.456) / 0.224,
            (0.0 - 0.406) / 0.225,
        ],
        dtype=np.float32,
    )
    assert np.allclose(output[:, 0, 0], expected, rtol=0.0, atol=1.0e-6)
    assert len({float(value) for value in output[:, 0, 0]}) == 3


def test_formal_surface_resnet_is_frozen_and_deterministic() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = load_p1_config(CONFIG, project_root=ROOT)
    encoder = build_surface_resnet18(config)
    first = Image.fromarray(np.arange(19 * 23 * 3, dtype=np.uint8).reshape(19, 23, 3))
    second = Image.fromarray(np.flip(np.asarray(first), axis=1).copy())
    left = encoder.encode((first, second))
    right = encoder.encode((first, second))
    assert left.shape == (2, 512)
    assert left.dtype == np.dtype("<f4")
    assert np.array_equal(left, right)
    assert np.all(np.isfinite(left))
    assert encoder.weights_sha256 == config.encoder_weight_sha256
    assert encoder.transform_sha256 == config.surface_transform_sha256
    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())
    assert not encoder.model.training


class _FakeEncoder:
    weights_sha256 = "f" * 64
    transform_sha256 = TRANSFORM_SHA
    device = "cpu"
    batch_size = 8

    def encode(self, images: tuple[Image.Image, ...] | list[Image.Image]) -> np.ndarray:
        rows = []
        for image in images:
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
            channel = array.mean(axis=(0, 1), dtype=np.float64) / 255.0
            coordinate = np.asarray(
                [array[0, 0, 0], array[-1, -1, 1]], dtype=np.float64
            ) / 255.0
            base = np.concatenate((channel, coordinate)).astype(np.float32)
            rows.append(np.resize(base, 512))
        return np.asarray(rows, dtype="<f4")

    def provenance(self) -> dict[str, object]:
        return {
            "encoder": "fake_surface_encoder",
            "weights_sha256": self.weights_sha256,
            "transform_sha256": self.transform_sha256,
            "device": self.device,
            "batch_size": self.batch_size,
            "output_dimension": 512,
        }


def _record(
    root: Path, specimen: str, domain: str, image: np.ndarray
) -> SurfaceCellRecord:
    relative = Path("data") / domain / f"{specimen}.png"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    source = FrameGeometry(image.shape[1], image.shape[0], 1.0, 1.0)
    destination = FrameGeometry(9, 9, 1.0, 1.0)
    transform = create_transform(
        source=source,
        destination=destination,
        orientation=Orientation.ROT90,
        evidence_class=EvidenceClass.A_DIRECT_METADATA,
        evidence_roles=(EvidenceRole.AUTHOR_CORRESPONDENCE,),
        evidence_hashes=("a" * 64,),
    )
    boxes = np.asarray(
        [row["surface_box"] for row in render_surface_grid(transform)], dtype="<f8"
    )
    boxes.setflags(write=False)
    return SurfaceCellRecord(
        specimen_id=specimen,
        dataset_id=domain,
        surface_path=relative,
        surface_sha256=_sha256(path),
        source=source,
        destination=destination,
        evidence_class=EvidenceClass.A_DIRECT_METADATA,
        evidence_roles=(EvidenceRole.AUTHOR_CORRESPONDENCE,),
        evidence_hashes=("a" * 64,),
        scale_x=8.0,
        scale_y=8.0,
        offset_x=0.0,
        offset_y=0.0,
        transform_sha256=transform.sha256,
        cell_boxes=boxes,
    )


def _authority(root: Path) -> SurfaceCellAuthority:
    y, x = np.mgrid[0:17, 0:21]
    first = np.stack((x * 7, y * 11, (x + y) * 5), axis=-1).astype(np.uint8)
    second = np.stack((y * 13, x * 3, (2 * x + y) * 4), axis=-1).astype(np.uint8)
    records = (
        _record(root, "a", "domain-x", first),
        _record(root, "b", "domain-x", second),
    )
    boxes = np.asarray([record.cell_boxes for record in records], dtype="<f8")
    boxes.setflags(write=False)
    return SurfaceCellAuthority(
        records=records,
        specimen_ids=("a", "b"),
        dataset_ids=("domain-x", "domain-x"),
        surface_paths=tuple(record.surface_path for record in records),
        surface_sha256=tuple(record.surface_sha256 for record in records),
        cell_boxes=boxes,
        state_sha256="b" * 64,
    )


def test_surface_feature_bank_is_label_free_hash_bound_and_replayable(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    authority = _authority(external)
    first_path = tmp_path / "features-a"
    second_path = tmp_path / "features-b"
    first = materialize_surface_feature_bank(
        authority,
        external_root=external,
        output=first_path,
        encoder=_FakeEncoder(),
        wrong_orientation_seed="agentic-nde-p1-wrong-orientation-v1",
    )
    second = materialize_surface_feature_bank(
        authority,
        external_root=external,
        output=second_path,
        encoder=_FakeEncoder(),
        wrong_orientation_seed="agentic-nde-p1-wrong-orientation-v1",
    )
    assert first.state_sha256 == second.state_sha256
    assert first.specimen_ids == second.specimen_ids == ("a", "b")
    assert first.global_embeddings.shape == (2, 512)
    assert first.local_correct_embeddings.shape == (2, 64, 512)
    assert first.local_wrong_orientation_embeddings.shape == (2, 64, 512)
    assert not np.array_equal(
        first.local_correct_embeddings, first.local_wrong_orientation_embeddings
    )
    assert (first_path / "manifest.json").read_bytes() == (
        second_path / "manifest.json"
    ).read_bytes()
    for name in (
        "global_surface_embeddings.npy",
        "local_correct_embeddings.npy",
        "local_wrong_orientation_embeddings.npy",
    ):
        assert _sha256(first_path / name) == _sha256(second_path / name)
    loaded = load_surface_feature_bank(
        first_path,
        authority=authority,
        expected_transform_sha256=TRANSFORM_SHA,
    )
    assert loaded.state_sha256 == first.state_sha256
    assert np.array_equal(loaded.local_correct_embeddings, first.local_correct_embeddings)
