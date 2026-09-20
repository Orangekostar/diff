from __future__ import annotations


def test_effect_sentence_preserves_adverse_direction():
    from scripts.cai_c_retrain.paper import _effect_sentence

    sentence = _effect_sentence(
        "control",
        {
            "gain_mpa": "-1.25",
            "ci_low": "-2.0",
            "ci_high": "-0.5",
            "metric": "MAE",
        },
    )

    assert "higher error by 1.250 MPa" in sentence
    assert "[-2.000, -0.500] MPa" in sentence
    assert "did not include zero" in sentence


def test_term_pages_locates_each_required_visual_item():
    from scripts.cai_c_retrain.paper import _term_pages

    extracted = "Abstract\fAlgorithm 1 and Figure 2\fTable 3\f"

    assert _term_pages(
        extracted, ("Abstract", "Algorithm 1", "Figure 2", "Table 3")
    ) == {
        "Abstract": 1,
        "Algorithm 1": 2,
        "Figure 2": 2,
        "Table 3": 3,
    }


def test_actor_supplement_rows_use_selected_score():
    from scripts.cai_c_retrain.paper import (
        _actor_selection_table,
        _actor_supplement_rows,
    )

    rows = _actor_supplement_rows(
        [
            {
                "method": "VLM_SPATIAL_FEEDBACK",
                "training_seed": 2026091301,
                "selected_update": 250,
                "selected_score": 45.846742151578745,
                "logical_updates": 1250,
                "actual_optimizer_updates": 1250,
            }
        ]
    )

    assert rows == [
        {
            "method": "C spatial feedback (main)",
            "seed": 2026091301,
            "selected": 250,
            "area": 45.846742151578745,
            "logical": 1250,
            "actual": 1250,
        }
    ]
    assert "Table S1." in _actor_selection_table(rows)


def test_portable_build_script_uses_pandoc_compatible_highlight_flag():
    from scripts.cai_c_retrain.paper import _portable_build_script

    source = (
        "pandoc = os.environ.get('PANDOC') or shutil.which('pandoc')\n"
        "subprocess.run([pandoc, '--syntax-highlighting=none'], "
        "capture_output=True, check=True)\n"
    )
    rendered = _portable_build_script(source)

    assert "pandoc_prefix = [pandoc]" in rendered
    assert "PANDOC_DATA_DIR" in rendered
    assert "subprocess.run(pandoc_prefix + [ '--no-highlight']" in rendered
    assert "capture_output=True" not in rendered


def test_vlm_terminal_counts_include_schema_invalid_rows():
    from scripts.cai_c_retrain.paper import _vlm_terminal_counts

    repairs, invalid = _vlm_terminal_counts(
        [
            {"status": "VALID_FIRST_PASS"},
            {"status": "VALID_AFTER_REPAIR"},
            {"status": "SCHEMA_INVALID_AFTER_ONE_REPAIR"},
            {"status": "SCHEMA_INVALID_AFTER_ONE_REPAIR"},
        ]
    )

    assert repairs == 1
    assert invalid == 2


def test_statistical_tail_uses_current_event_grid_count():
    from scripts.cai_c_retrain.paper import _update_event_grid_count

    old = "A supplementary event grid contains the 599 shared breakpoints obtained from saved episodes."

    assert "578 shared breakpoints" in _update_event_grid_count(old, 578)


def test_markdown_table_preserves_integer_fields():
    from scripts.cai_c_retrain.paper import _markdown_table

    table = _markdown_table(
        [{"method": "actor", "seed": 2026091301, "area": 45.8467}],
        (("method", "Method"), ("seed", "Seed"), ("area", "A")),
    )

    assert "2026091301.000" not in table
    assert "2026091301" in table
    assert "45.847" in table


def test_clear_visual_previews_removes_only_generated_pngs(tmp_path):
    from scripts.cai_c_retrain.paper import _clear_visual_previews

    (tmp_path / "old_page.png").write_bytes(b"old")
    (tmp_path / "keep.json").write_text("{}", encoding="utf-8")

    _clear_visual_previews(tmp_path)

    assert not (tmp_path / "old_page.png").exists()
    assert (tmp_path / "keep.json").is_file()
