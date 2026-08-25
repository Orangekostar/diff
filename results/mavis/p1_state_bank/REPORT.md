# MAVIS P1 Strict-OOF Sequential State Bank

Status: `COMPLETE`.

The package contains `8280` causal states and `2157215` strict-OOF state-action labels for `276` physical specimens across six leave-one-domain-out domains. Five frozen trajectory families and six registered checkpoints are present for every specimen.

Every teacher fit excludes both the held-out target domain and the query source domain. Source true CAI appears only in privileged state-action teacher rows; policy-visible state rows contain only initial context, actually revealed measurements, exact costs, legal candidate geometry/cost, and strict-OOF predictions.

`1380` terminal checkpoint states have no legal next action under the exact 25% endpoint and therefore have no state-action rows. Statistical inference remains specimen/domain-level; state-action rows are training samples, not independent experimental replicates.
