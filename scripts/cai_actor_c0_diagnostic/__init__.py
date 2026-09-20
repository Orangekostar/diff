"""Frozen Actor C0 mechanism diagnostics."""

from pathlib import Path

import cmc_bbdm

_LOCAL_PACKAGE = str(Path(__file__).resolve().parents[2] / "src/cmc_bbdm")
if _LOCAL_PACKAGE not in cmc_bbdm.__path__:
    cmc_bbdm.__path__.append(_LOCAL_PACKAGE)
