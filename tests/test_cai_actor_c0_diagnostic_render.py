from pathlib import Path

import numpy as np
from PIL import Image

from scripts.cai_actor_c0_diagnostic.render import (
    _candidate_masks,
    _html_document,
    audit_local_html_links,
    centered_legal_logits,
    nearest_rgba_layer,
)


def test_candidate_masks_distinguish_vlm_candidates_from_effective_c0():
    indicator = np.asarray([0.0, 0.25, 1.0, 0.0])
    proposal = np.asarray([True, False, True, True])

    vlm, effective = _candidate_masks(
        indicator,
        proposal,
        c0_reason="HIGHEST_RELIABLE_CONFIDENCE_C0",
    )
    assert np.array_equal(vlm, np.asarray([False, True, True, False]))
    assert np.array_equal(effective, proposal)

    vlm, effective = _candidate_masks(
        indicator,
        proposal,
        c0_reason="VLM_UNAVAILABLE",
    )
    assert np.array_equal(vlm, np.asarray([False, True, True, False]))
    assert not np.any(effective)


def test_nearest_rgba_layer_preserves_cell_boundaries():
    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    layer = nearest_rgba_layer(values, width=80, height=80, vmin=0, vmax=63)
    assert isinstance(layer, Image.Image)
    array = np.asarray(layer)
    assert array.shape == (80, 80, 4)
    assert np.array_equal(array[0, 0], array[9, 9])
    assert not np.array_equal(array[9, 9], array[10, 10])


def test_html_audit_rejects_remote_or_missing_assets(tmp_path: Path):
    (tmp_path / "ok.png").write_bytes(b"png")
    good = tmp_path / "good.html"
    good.write_text('<img src="ok.png"><a href="good.html">self</a>', encoding="utf-8")
    assert audit_local_html_links(good) == ["good.html", "ok.png"]
    remote = tmp_path / "remote.html"
    remote.write_text('<script src="https://example.com/x.js"></script>', encoding="utf-8")
    try:
        audit_local_html_links(remote)
    except ValueError as error:
        assert "remote" in str(error)
    else:
        raise AssertionError("remote asset should fail")
    missing = tmp_path / "missing.html"
    missing.write_text('<img src="absent.png">', encoding="utf-8")
    try:
        audit_local_html_links(missing)
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("missing asset should fail")


def test_centered_legal_logits_masks_illegal_cells_and_centers_legal_values():
    logits = np.asarray([1.0, 4.0, 9.0, -3.0])
    legal = np.asarray([True, True, False, True])
    centered = centered_legal_logits(logits, legal)
    assert np.isnan(centered[2])
    assert np.isclose(centered[legal].mean(), 0.0)
    assert np.allclose(centered[legal], np.asarray([1.0, 4.0, -3.0]) - 2.0 / 3.0)


def test_html_has_required_question_navigation_and_raw_state_links(tmp_path: Path):
    selected = [{"specimen_key": "domain:q1", "target_mpa": "250.0"}]
    common = {
        "specimen_key": "domain:q1",
        "physical_state_sha256": "a" * 64,
        "prefix_cells_in_order": "[]",
        "t": "0",
        "action_count": "0",
        "exact_cost": "0.0",
        "terminal_view_only": "False",
        "policy_input_sha256": '{"C":"' + "b" * 64 + '","N":"' + "c" * 64 + '"}',
        "c_prior_sha256": "d" * 64,
    }
    states = [
        common
        | {
            "state_id": "s00_shared",
            "sources": "['C_NATIVE_PREFIX', 'N_NATIVE_PREFIX']",
            "source_models": "['C', 'N']",
            "prefix_lengths": "[0, 0]",
            "state_origins": '[{"model":"C","t":0},{"model":"N","t":0}]',
        },
        common
        | {
            "state_id": "s01_c",
            "sources": "['C_NATIVE_PREFIX']",
            "source_models": "['C']",
            "prefix_lengths": "[1]",
            "state_origins": '[{"model":"C","t":1}]',
            "prefix_cells_in_order": "[7]",
            "t": "1",
        },
        common
        | {
            "state_id": "s01_n",
            "sources": "['N_NATIVE_PREFIX']",
            "source_models": "['N']",
            "prefix_lengths": "[1]",
            "state_origins": '[{"model":"N","t":1}]',
            "prefix_cells_in_order": "[3]",
            "t": "1",
        },
    ]
    figure_paths = {
        name: f"panels/domain_q1/{name}.png"
        for name in (
            "clean",
            "heatmap_layer",
            "candidate_layer",
            "selection_layer",
            "number_layer",
            "A",
            "B",
            "D",
            "E",
            "F",
            "F_csv",
        )
    }

    document = _html_document(
        tmp_path,
        selected,
        states,
        {"domain:q1": figure_paths},
    )

    for label in (
        "原生轨迹对照",
        "相同状态对照",
        "C来源状态",
        "N来源状态",
        "概率热图",
        "VLM全部候选",
        "原生选中",
        "格子编号",
        "动作顺序及差值表",
    ):
        assert label in document
    for relative in (
        "metadata.json",
        "physical_state.npz",
        "C/scores.csv",
        "N/scores.csv",
        "C/attention.npz",
        "N/attention.npz",
        "C/surface_attribution.npz",
        "N/surface_attribution.npz",
        "C/zero_prior_scores.npz",
        "C/zero_prior_sensitivity.json",
        "C/query_checks.json",
        "N/query_checks.json",
        "panels/domain_q1/group_C_s01_c.png",
        "panels/domain_q1/group_C_s01_n.png",
    ):
        assert relative in document
