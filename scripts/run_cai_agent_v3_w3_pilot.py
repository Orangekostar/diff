"""Run the explicitly authorized W3-only pilot from this checkout."""

from pathlib import Path

import cmc_bbdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
cmc_bbdm.__path__.append(str(PROJECT_ROOT / "src/cmc_bbdm"))
from cmc_bbdm.cai_agent_v3.w3_pilot import main

if __name__ == "__main__":
    main()
