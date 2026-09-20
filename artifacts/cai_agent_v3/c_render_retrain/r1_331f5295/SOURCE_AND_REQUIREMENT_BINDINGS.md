# Source And Requirement Bindings

- Authority: `docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json`
- Frozen source: `331f52952b93bd9442ad2e44243cc71c7d966b4d`
- Prior C: exact P0 prompt plus exact pilot R1 renderer.
- Cohort: 161 TRAIN plus 50 VALID; TEST image, labels and scoring are forbidden.
- Predictors: frozen W2 MEAN_SC P_all and original three-fold OOF routing.
- Actors: only the three registered C-dependent methods and registered seeds.
- Evidence and manuscript: rebuilt in isolated C release roots.
