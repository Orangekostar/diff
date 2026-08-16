# CPB V3 Extended Gate Status

## Current decision

The registered sequence was extended beyond the historical P2 stop point after
explicit authorization. Cohorts, endpoints, thresholds, seed panels, and
held-out-dataset inferential units were frozen before each new result was read.

```text
G1 scalar observability: FAIL
G2 scalar incremental utility: FAIL
P1 measured full-field utility: PASS
P2 privileged surface transfer: FAIL
P3 spatial specificity: PASS
P4 dense learned representation: NO_GO
P5 sparse-scan retention: PASS
P6 diffusion reconstruction: NO_MECHANICAL_GAIN
P7 surface+sparse fusion: NO_COMPLEMENTARITY
```

## P3: spatial specificity

The measured full-field method had equal-domain MAE 0.1284893565. Destroying
8x8-patch spatial organization increased it to 0.1839398265. The registered
effect was 0.0554504700, a 30.1460% relative improvement for the original field;
6/6 held-out domains improved and the simultaneous lower bound was 0.0090475767.
P3 therefore passed. Pixel shuffle and canonical compact controls also degraded
prediction. This establishes predictive spatial specificity, not a unique
causal damage map.

## P4: dense representations

The frozen ResNet global baseline remained best at MAE 0.1284893565. DINOv2
global/spatial reached 0.1516672851/0.1541306371, DDPM spatial reached
0.1979309161, and residual-diffusion fusion reached 0.1548007175. The primary
fusion-minus-baseline improvement was negative (-0.0263113609), only 1/6
domains improved, and its simultaneous interval was entirely non-positive.
P4 therefore returned `NO_GO`. Spatial destruction affected some learned
representations, but that sensitivity did not make them mechanically superior.

## P5: sparse retention

At 25% registered sampling density, bilinear sparse reconstruction achieved MAE
0.1345137120 versus 0.1284893565 for the full field and 0.1881207811 for the
surface baseline. It retained 0.8989734770 of the full-field gain, above the
registered 0.80 threshold; 25% and 50% densities passed and 25% was selected as
the knee. P5 therefore passed and authorized P6/P7.

## P6: diffusion reconstruction

P6 used identical 25% coordinates for all methods, exact native measured-point
overwrite, six outer-domain folds, 8 diffusion draws, and 100,000 common
bootstrap resamples. Both P1 surface and full-field predictions were reproduced
with zero difference at the 1e-12 tolerance. Every one of 1,656 reconstruction
rows had zero measured-point error.

Classical interpolation retained almost all mechanical information:

| Method | Equal-domain MAE | Retention | RGB L1 | SSIM | PSNR dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| bilinear | 0.1298021627 | 0.9779846579 | 0.0060214798 | 0.9716298967 | 32.2857 |
| PCHIP | 0.1293203561 | 0.9860644010 | 0.0055800915 | 0.9750828268 | 32.3511 |
| deterministic U-Net | 0.1430596629 | 0.7556606018 | 0.0163601700 | 0.8855648178 | 27.7603 |
| diffusion | 0.1422009259 | 0.7700613476 | 0.0471427492 | 0.6428599360 | 23.9987 |

Diffusion was worse than bilinear by 0.0123987632 MAE (relative -9.5520%,
1/6 domains; simultaneous interval [-0.0381994874, 0.0143074862]) and worse
than PCHIP by 0.0128805698 (relative -9.9602%, 1/6; simultaneous interval
[-0.0410903637, 0.0160192752]). Its 0.0008587370 point advantage over the
deterministic U-Net was only 0.6003%, occurred in 3/6 domains, and had a
simultaneous interval crossing zero [-0.0030657042, 0.0052017409]. Retention,
image-L1 non-inferiority, all effect directions, domain count, and simultaneous
interval gates failed. P6 therefore returned `NO_MECHANICAL_GAIN`.

The registered evidence shows that learned unmeasured-region reconstructions
were less faithful and less mechanically useful than simple interpolation. It
does not isolate whether the cause was cohort size, cross-domain shift, the
U-Net objective, the DDIM schedule, or the frozen downstream embedding.

An explicitly post-hoc diagnostic found eight-draw range coverage of 10.14%
and an overall correlation of -0.269 between CAI predictive standard deviation
and absolute error. The two highest-error domains had zero draw-range coverage
and the smallest mean predictive spreads. These values suggest under-dispersed,
domain-misaligned uncertainty but are not registered calibration evidence.

## P7: surface complementarity

At 25% density, sparse-only MAE was 0.0901159208 while surface+sparse MAE was
0.1345137120. The registered sparse-minus-fusion effect was -0.0443977912
(relative -49.2674%), only 1/6 domains improved, and the simultaneous interval
was [-0.1175444300, -0.0035476021]. At 50%, the effect was -0.0409716649
(relative -45.4853%), 2/6 domains improved, and the simultaneous interval was
[-0.1101490849, -0.0008755527]. P7 therefore returned
`NO_COMPLEMENTARITY`. This is estimator-specific evidence against adding the
registered surface representation; it is not proof that every possible surface
measurement is irrelevant.

## Reproducibility

Production and replay directories are byte-identical for every P3-P7 stage.

| Stage | Scientific digest | Output-tree digest | Manifest SHA-256 |
| --- | --- | --- | --- |
| P3 | `a31903f424523162fba4ccbfbc8b9f2dfbe4988d4156b1a8b24711b030107d4d` | `4c5c88b80ef7e8c88b88b74ce61db2c07465f844d04efb34958891b39a274373` | `bb4d764bdb4939cb8f25c9939382b2620adc6be0a5dcd5544bb28fcb5b0fbbc3` |
| P4 | `f62f4f27122323aa57e95d1223fd10a3882ab4eb1d69d0056870c9d427a12f23` | `85392ce550d3bf2ebedebc0b9ddbbfac6f4ebb8580657756ea3b2b7127b2f12b` | `4f02f243904819ad45b8efe9040b99b024d2b3ac0453aededfdf9273aba0f406` |
| P5 | `87b1da699b0ee59cf1723339a2150d673998b20004b6e990bb1a1a87d48f2257` | `a2c54b7fdc53252767c7788365d254bbcdbe36571d2e553ce56f406dc039c80d` | `b3f6554c15b41d18264c4a8e12bcafabd6824c81db608e3e24099b225e45041e` |
| P6 | `fc8431597eadc9f1dff9b956d16810b6821888eef0b24c3db4df8a0dff50d505` | `433699f015914c166696d2c19feaf8fe452482ff61200d663afb76fae6f493e9` | `01bb169f5648d0809466e26a8482c7025bc3471de306a0cc2bf8c2f4b0e08e6e` |
| P7 | `2c5c7fc6ea953367af4b82146030a79821af29ea35cdcfff0ee40977a73deab6` | `384e96ac1f04630ee8eb367865b25811a50dd750f7f64541db54cfe789057769` | `bb88319469398aec581a4814b5deb09043eaa8b3e99fb66c7330eaed1661dd94` |

`V3_FINAL_GATE_STATUS.md` remains the historical P2-time snapshot and is
superseded for current stage status by this document.
