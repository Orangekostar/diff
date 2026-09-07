# G1 FIELD Policy Result

Status: `G1_FIELD_POLICY_NO_GO`

The preregistered effect is fixed-baseline AUEBC minus learned-policy AUEBC, so
positive values favor the learned policy.

| Quantity | Result |
|---|---:|
| Fixed strongest source-selected baseline | SURFACE_FOCUS in 6/6 outer folds |
| Fixed AUEBC | 0.002047837067539446 |
| Learned AUEBC | 0.002120618581802132 |
| Privileged oracle AUEBC | 0.0019655610066778156 |
| Fixed minus learned | -0.0000727815142626863 |
| Synchronized 95% bootstrap CI | [-0.00008689672989010225, -0.00006182037360547485] |
| Improved held-out domains | 0/6 |
| Oracle-gap closure | -0.8846013469833925 |

Domain effects were negative in all six held-out domains: `74t7kcdgkr`
-0.00010953931745652554, `cgtnjyggtm` -0.00006932465420947177,
`w68dtmpfyf` -0.00003572992830907941, `xcmzfsbd9t`
-0.00008515543029735685, `yfxyg8jm46` -0.00010262372213405563, and
`ykhs7s2dck` -0.00003431603316962853.

The learned observable FIELD policy is statistically worse than the strongest
source-selected fixed policy and does not close the required 20% privileged
oracle gap. The FIELD component therefore fails all three policy-GO conditions.
