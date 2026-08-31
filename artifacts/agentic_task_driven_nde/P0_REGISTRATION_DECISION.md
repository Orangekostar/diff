# P0 Registration Decision

Date: 2026-08-31 UTC

## Decision

```text
P0_SPATIAL_REGISTRATION_NO_GO
P1-P4 = NOT_RUN_NOT_AUTHORIZED
```

Exact surface/C-scan/CAI identity binding succeeded, but no deployable,
source-supported surface-to-C-scan transform exists. This is a completed
negative P0 result, not a runtime failure and not a request for post hoc model
rescue.

## Gate matrix

| P0 requirement | Evidence | Result |
| --- | --- | --- |
| Exact identity and hash binding | 276/276 rows; six-domain counts `45/49/43/59/42/38`; every row `PASS_EXACT_SPECIMEN_ID_AND_HASH` | PASS |
| Deterministic transform without CAI/hidden damage content | no shared frame, orientation, scale/offset, or export transform exists | FAIL |
| Orientation resolved by authoritative/geometry-only evidence | all 276 rows are `UNRESOLVED_8_WAY_AMBIGUITY` | FAIL |
| Operational coverage | authorized counts are `0/45`, `0/49`, `0/43`, `0/59`, `0/42`, `0/38`; total `0/276` | FAIL |
| Transform replay | no real transform exists to replay; package-level negative-decision replay is deterministic | FAIL for GO gate |

The identity-specific NO-GO is not applicable because identity authority
passed. `P0_HUMAN_REVIEW_REQUIRED` is not applicable because the issue is not
marginal coverage: all real transforms are unsupported.

## Registration-evidence audit

### A. Direct metadata / export transform

Not found. The source does not publish a common surface/C-scan frame,
instrument-coordinate correspondence, crop offsets, axis directions, or an
export transform.

### B. Deterministic geometry-only transform

Not authorized. Published `80 x 80 mm` specimen and `75 x 75 mm` scan extents
do not establish that image edges are common physical boundaries and do not
resolve axis swap/reflection, scale, crop, or offset. Normalized image
coordinates are not physical correspondence.

### C. Source-only learned registration

Not run. There is no preregistered non-mechanical correspondence target or
source-only landmark authority that could validate this route without hidden
C-scan content. A learned procedure was not introduced merely to rescue P0.

## Integrity controls

- CAI accessed during registration: `false`
- hidden C-scan content used during registration: `false`
- damage centroid or mask used: `false`
- target-domain label or oracle value used: `false`
- manual target alignment used: `false`
- normalized coordinates accepted as correspondence: `false`
- new training: `false`
- authorized real transforms: `0`

The typed transform and 8x8 grid code remain validated by synthetic
known-transform tests. Those tests establish implementation correctness; they
do not manufacture missing real-world authority.

## Machine authority

- `summary.json` SHA-256:
  `9d0032828614c79c2ed9faaf0fcfdbbb2e7c39382fa7e4cb1fc1d28ec4a9a826`
- `registration.csv` SHA-256:
  `fc7a3b9ceef28e371a6a340444db992f1bf8aa87596cb1155c8ab48957c61f96`
- `registration_qc.csv` SHA-256:
  `36a43f4b3f57a279a6a9ce429da3d25a81929073eabb8c8b56ab9d737d8d7630`
- `source_hashes.csv` SHA-256:
  `f31fbdd9494fb23124094102181305342498b1e32e6c33c3407de5e5b7436f2a`
- `artifact_manifest.json` SHA-256:
  `2ad59787b8c7c8726ce3d1dc0475540e7db56bfecb7dc0604f0b982557338523`

The formal package contains exactly ten files. Its checksum ledger is verified
from within `results/agentic_task_driven_nde/p0_registration/` because the
ledger intentionally stores package-local filenames for destination-independent
determinism.

## Stop decision

No P1 config, P1 protocol, visual scorer, feature cache, embedding, prompt set,
or model output is authorized. P2, P3, P4, and manuscript creation are also
prohibited. The only remaining work is P0 verification, frozen-path audit,
handoff, and Git/GitHub synchronization.
