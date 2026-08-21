# D8 Pilot Decision

Date: 2026-08-18

## Frozen decision

```text
TRAIN_RESIDUAL_DIFFUSION
```

This decision is the direct output of the preregistered D8 escalation rule. It
does not use any formal outer-domain prediction or target.

## Validated pilot package

- Package: `results/d8_search`
- Registered validation command: `scripts/run_d8_exploration.sh validate`
- Initial trials: 432 (`72` for each of six prospective outer folds)
- Complete trials: 331
- Pruned trials: 101
- Formal outer evaluations: 0
- Config SHA-256: `a040eb25a95b166ca674a1744b53088b7c1b5ec14c11a88fbada953846ff119b`
- P6 residual-bank SHA-256: `a8b4723cc343a7bb4480b24fd9495f08b856f798a30e9faf3a587cc0890be18b`
- Escalation evidence SHA-256: `c55e017303297ac2d0f101ce8e517b7092c8d542109d3d65c6f812592a456ba4`
- Scientific digest: `3478d97858236c1873c88d8fc3e910dbe659e05d2c4e472eac15825e999474ca`
- Output-tree SHA-256: `685798b852590b37c2d3857b95e7de212ca32cc087cd306b99fa748d7946eb2d`
- Artifact-manifest SHA-256: `3e9a5af42bddb78a9c573265313dd6db6a1b0a6b3bc39d57713ee3b6130070d4`

The wrapper is part of the registered runtime contract. In particular, it
sets the BLAS thread count to one before recomputing the simplex ensemble.

## Trigger evidence

| Prospective outer fold | Inner domains improved by at least 1% | Low-band residual energy | Selected diffusion weight | Freeze condition |
| --- | ---: | ---: | ---: | --- |
| `74t7kcdgkr` | 3/5 | 0.914821 | 1.0 | met |
| `cgtnjyggtm` | 3/5 | 0.888717 | 0.0 | not met |
| `w68dtmpfyf` | 2/5 | 0.928425 | 1.0 | met |
| `xcmzfsbd9t` | 2/5 | 0.927297 | 1.0 | met |
| `yfxyg8jm46` | 3/5 | 0.941707 | 1.0 | met |
| `ykhs7s2dck` | 3/5 | 0.939994 | 1.0 | met |

The trend trigger requires at least three improved inner domains in at least
three prospective outer studies. Four studies met it. The mismatch trigger
requires at least three studies with low-band energy at least 0.50 or low
acceptance at `alpha=0.1`; all six studies met the energy condition. The pilot
freeze condition also held in five studies, but the registered priority is:

```text
TRAIN_RESIDUAL_DIFFUSION
FREEZE_PILOT_FOR_OUTER_EVALUATION
CLOSE_DIFFUSION_SPECIFIC_ROUTE
```

Therefore the training branch takes precedence.

## Scientific interpretation

The pilot provides positive inner-domain evidence for nuisance
marginalization, but the P6 reconstruction residual is dominated by energy
below the selected nuisance cutoff. It is therefore not an adequate formal
residual prior. This result authorizes a new fold-local conditional residual
model `p(R|S)`; it does not show that D8 beats the frozen outer baseline.

## Next gate

Before any outer view is issued:

1. freeze the residual definition, model, training splits, search budget, and
   promotion rule;
2. train and select the residual model using outer-fit and inner-query domains
   only, with no CAI value supplied to the generator;
3. publish and independently validate the pre-outer training package;
4. freeze one pipeline per prospective outer fold.

Only after these steps may the one-way formal outer evaluation begin.
