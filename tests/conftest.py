from __future__ import annotations

from pathlib import Path

import cmc_bbdm

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE = str(ROOT / "src/cmc_bbdm")

if LOCAL_PACKAGE not in cmc_bbdm.__path__:
    cmc_bbdm.__path__.append(LOCAL_PACKAGE)
