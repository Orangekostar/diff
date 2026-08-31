# Author Surface/C-scan Registration Authority

Date recorded: 2026-08-31 UTC

## Source identity

```text
source_type: USER_ATTESTED_PERSONAL_COMMUNICATION_WITH_DATASET_AUTHOR
evidence_status: USER_ATTESTED
controlling_prompt_sha256: 37265bb06eef238dca2325b590d1353a8159514ac693e4b4d070a637fb3b8eb8
original_author_communication_artifact: NOT_PROVIDED
manuscript_archive_status: ORIGINAL_AUTHOR_COMMUNICATION_ARCHIVE_RECOMMENDED
```

No author email, account, message timestamp, signature, screenshot, or private
contact detail was supplied or inferred.

## Verbatim statement

```text
hasebe所有表面rgb的png图像顺时针旋转90度就是扫描图像jpg，
比如Q24-7astm.png顺时针旋转90度得到Q24-7astm.jpg，
其外边框的大小未做裁切，只是比例不同
```

The SHA-256 of the exact UTF-8 statement above, with LF separators and no
terminal newline, is:

```text
3560662d4509ea3e059d597cedca15950cce02f706a992330b161381acfba6ba
```

## Technical claims

1. Surface PNG to corresponding scan-image orientation is clockwise 90 degrees.
2. The corresponding specimen outer frame is not additionally cropped.
3. Pixel dimensions and proportions may differ between modalities.
4. The supplied example is `Q24-7astm.png -> Q24-7astm.jpg`.

## Authorized interpretation

```text
orientation: ROT90
outer_frame_crop: NONE_AT_SPECIMEN_FRAME
mapping_basis: AUTHOR_FULL_FRAME_PIXEL_CORRESPONDENCE
physical_mm_used_for_cross_modal_mapping: false
example_surface: Q24-7astm.png
example_scan: Q24-7astm.jpg
```

This evidence authorizes normalized full-frame pixel correspondence after the
historical screenshot is reduced to its known specimen panel. It does not
authorize physical scanner-coordinate calibration, equal mm/pixel resolution,
per-specimen orientation selection, mirror selection, manual alignment, or any
registration selected from C-scan damage, CAI, oracle value, or target outcome.

## Historical relation

The frozen P0 NO-GO remains scientifically correct for the evidence available
at that time. This later user-attested author communication supplies the missing
orientation and specimen-frame semantics and is evaluated in the separate P0R
stage.
