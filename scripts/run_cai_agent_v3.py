#!/usr/bin/env python3
"""Thin executable wrapper for :mod:`cmc_bbdm.cai_agent_v3.cli`."""

from pathlib import Path

import cmc_bbdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE = str(PROJECT_ROOT / "src/cmc_bbdm")
if LOCAL_PACKAGE not in cmc_bbdm.__path__:
    cmc_bbdm.__path__.append(LOCAL_PACKAGE)

from cmc_bbdm.cai_agent_v3.cli import main

if __name__ == "__main__":
    main()
