#!/usr/bin/env python3
"""Run registered staged MAVIS workers and package validation."""

from __future__ import annotations

import importlib
from pathlib import Path

import cmc_bbdm

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE = str(ROOT / "src/cmc_bbdm")
if LOCAL_PACKAGE not in cmc_bbdm.__path__:
    cmc_bbdm.__path__.append(LOCAL_PACKAGE)

main = importlib.import_module("cmc_bbdm.mavis.cli").main


if __name__ == "__main__":
    raise SystemExit(main())
