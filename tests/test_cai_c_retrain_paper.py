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
