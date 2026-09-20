from pathlib import Path

import numpy as np
from PIL import Image

from scripts.cai_actor_c0_diagnostic.render import (
    audit_local_html_links,
    nearest_rgba_layer,
)


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
