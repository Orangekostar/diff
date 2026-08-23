from __future__ import annotations

import numpy as np

from cmc_bbdm.mva.oracle_execution import _uniform_archive_valid


def test_uniform_archive_missing_checkpoint_is_invalid(tmp_path) -> None:
    path = tmp_path / "bank.npz"
    np.savez_compressed(path, embedding_0p0625=np.zeros((276, 512)))

    with np.load(path, allow_pickle=False) as archive:
        assert _uniform_archive_valid(archive, specimen_count=276) is False
