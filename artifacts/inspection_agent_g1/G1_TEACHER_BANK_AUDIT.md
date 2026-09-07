# G1 Teacher-Bank Audit

Status: `COMPLETE_SOURCE_ONLY_NO_TARGET_LEAKAGE`

Base: `7a10cd425de582fa158bf6639285731ccd8ff7a7`

Frozen config: `paper_v3/configs/inspection_agent_g1.yaml`

Formal manifest: `results/inspection_agent/g1/teacher_bank_manifest.csv`
(`06e3a07715da100796885c3f72a167b70ec24c4f802c7d1fc2f593c6a7106c54`)

## Crossfit And Storage

Thirty teacher banks cover every ordered outer-target/source-domain pair. For
each pair, the prior and StateCAIAssessor were fit on exactly the other four
domains. The outer target and teacher-labeled source domain were excluded from
fitting, normalization, PCA, and labels. The exact 30 rosters are frozen in
`G1_PRIVILEGE_AND_SPLIT_CONTRACT.md`.

The raw Zstandard Parquet banks are reproducible work products under
`results/inspection_agent/g1_work/teacher_banks/` and are intentionally excluded
from Git. The committed formal manifest binds every bank by row count, Parquet
SHA-256, logical-record SHA-256, and bank-manifest SHA-256. Across the 30 banks,
the sorted-unique newline-delimited digest is:

| Identity set | Count | SHA-256 |
|---|---:|---|
| Parquet hashes | 30 | `7ae6f3bae8e4ab94aea4dcd261663f47b5a93c4bc0d4af58240bb7d9544c48dc` |
| Logical-record hashes | 30 | `75f6f92d2f94e1dfdd3bb0643bb0bf7ffbda5e6c041125c299a9ff997ff023f6` |
| Bank-manifest hashes | 30 | `c437028db6550ed8409ba2d001ac754bcb15599523765f4de182331cbc8cc518` |
| Prior hashes | 30 | `27408c73e875b4e8a9fc77db7681a14ab5726ee13f0ee01d130220eda7882b36` |
| Assessor hashes | 30 | `843ae2ea63a5aa2d781f14f257846c3ec0c94b4f68c807e6208ac429ccc37a87` |
| Source-state hashes | 46,920 | `d041b19dcd840d25ec98231bd96be76963df29563e90bf41a0f68f1b473c3710` |
| Observable policy-state hashes | 46,920 | `e78d0283c91abe370afebd877b2de9fc88137a7edd12afdbe57a03d8b845d3b1` |
| Privileged teacher-state hashes | 46,920 | `7fcb546a41d370e5f2d6775bbff37d3de697e458508a1777c4a482bf57eb57be` |

The exact 276 decoded/source C-scan hashes remain in the frozen G0 roster
`results/inspection_agent/g0/authorized_roster.csv` (file SHA-256
`36ae9de700bf3f2306abdac16b729eb56ca8b859ba7a05a7ad78c527e209dd6b`).
Its sorted decoded-C-scan hash-set digest is
`bc34b8ebc6669a0e14b1850a7084a1f6956f6391d973400296a15cfdaf6a3c79`.

## Counts

- Banks: 30; rows: 46,920; physical specimens: 276.
- Each physical specimen contributes 170 records over its five source roles;
  each outer/source specimen contributes 34 records, 17 per task.
- Tasks: FIELD 23,460; CAI 23,460.
- Source-domain rows: `74t7kcdgkr` 7,650; `cgtnjyggtm` 8,330;
  `w68dtmpfyf` 7,310; `xcmzfsbd9t` 10,030; `yfxyg8jm46` 7,140;
  `ykhs7s2dck` 6,460.
- Outer-fold rows: `74t7kcdgkr` 7,854; `cgtnjyggtm` 7,718;
  `w68dtmpfyf` 7,922; `xcmzfsbd9t` 7,378; `yfxyg8jm46` 7,956;
  `ykhs7s2dck` 8,092.
- State sources: WARM_START 2,760; each of UNIFORM_CONTINUE,
  SURFACE_FOCUS_CONTINUE, RANDOM_CONTINUE, and ALTERNATE_BROADEN_REFINE 8,280;
  ORACLE_CHECKPOINT 11,040.
- Candidate count per state: min 1, median 62, max 64, mean
  46.35880221653879.
- Candidate decision occurrences: FOCUS 82,334; BROADEN 668,635; REFINE
  1,424,186.

Selected teacher decisions by task and state origin:

| Origin | Task | FOCUS | BROADEN | REFINE |
|---|---|---:|---:|---:|
| Base teacher bank | FIELD | 1,885 | 9,155 | 12,420 |
| Base teacher bank | CAI | 1,942 | 10,387 | 11,131 |
| DAgger actor-visited | FIELD | 6,069 | 14,482 | 16,121 |
| DAgger actor-visited | CAI | 4,264 | 15,461 | 16,947 |

Source-only DAgger added 60 banks with 88,320 actor-visited records: 44,160
FIELD and 44,160 CAI records over the same 276 physical specimens. The STOP
teacher banks contain 46,920 records: 20,249 sufficient and 26,671 insufficient.
By task, CAI has 13,929 sufficient and 9,531 insufficient records; FIELD has
6,320 sufficient and 17,140 insufficient records.

## Reproduction

Bank command template:

```bash
python scripts/run_inspection_agent_g1.py build-bank \
  --config paper_v3/configs/inspection_agent_g1.yaml \
  --source-project-root /home/ww/paper3 \
  --project-root . \
  --device cuda:2 \
  --outer-target <outer-domain> \
  --source-domain <source-domain> \
  --work-root results/inspection_agent/g1_work
```

The run used Python 3.13.13, NumPy 2.5.1, Polars 1.42.0, scikit-learn
1.9.0, PyTorch 2.12.1+cu130, CUDA 13.0, cuDNN 9.2.0, and an NVIDIA A40.
The config fixes seeds `2026090101` through `2026090104`; training enters the
deterministic-algorithm context, disables cuDNN benchmarking and TF32, seeds
Torch CPU/CUDA, uses explicit NumPy PCG64 generators, and uses AdamW. The Python
standard-library RNG is not called in the G1 implementation. No target outcome
was opened while constructing or selecting any teacher or DAgger bank.
