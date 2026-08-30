from __future__ import annotations

import importlib
from pathlib import Path

import cmc_bbdm

LOCAL_PACKAGE = str(Path(__file__).resolve().parents[1] / "src/cmc_bbdm")
if LOCAL_PACKAGE not in cmc_bbdm.__path__:
    cmc_bbdm.__path__.insert(0, LOCAL_PACKAGE)

main = importlib.import_module("cmc_bbdm.damage_response.pipeline").main

if __name__ == "__main__":
    raise SystemExit(main())
