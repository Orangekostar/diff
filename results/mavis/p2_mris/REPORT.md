# MAVIS P2 MRIS Informativeness

Status: `COMPLETE`. This stage does not assign the final MAVIS claim tier.

The domain-balanced real-state CAI AUEBC is `0.1250432019`. Control differences are positive when the control has higher error:

- `positions_only` minus `real`: `-0.0178288354` AUEBC
- `reconstruction` minus `real`: `-0.0373613302` AUEBC
- `shuffled` minus `real`: `0.0040816280` AUEBC
- `static` minus `real`: `0.0122325582` AUEBC

All predictions are nested leave-one-domain-out. Model selection and early stopping use source domains only. Metrics first aggregate state rows to physical specimens and then weight the six held-out domains equally. Shuffled content retains recipient positions and exact cost while using a recorded different donor specimen. Reconstruction values reuse strict-OOF P1 predictions and introduce no new reconstruction network.
