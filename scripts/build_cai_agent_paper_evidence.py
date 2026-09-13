"""Frozen-result analysis CLI; never launches research inference or training."""

from pathlib import Path

import cmc_bbdm

LOCAL_PACKAGE = str(Path(__file__).resolve().parents[1] / "src/cmc_bbdm")
if LOCAL_PACKAGE not in cmc_bbdm.__path__:
    cmc_bbdm.__path__.append(LOCAL_PACKAGE)
from cmc_bbdm.cai_agent_v3.paper_evidence import main

if __name__ == "__main__":
    main()
