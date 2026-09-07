# G1 CAI Policy Result

Status: `G1_CAI_POLICY_NO_GO`

The preregistered effect is fixed-baseline AUEBC minus learned-policy AUEBC, so
positive values favor the learned policy.

| Quantity | Result |
|---|---:|
| Fixed strongest source-selected baseline | RANDOM in 4 folds; CENTER_FIRST in 2 folds |
| Fixed AUEBC | 0.02859496549734356 |
| Learned AUEBC | 0.029653166484753902 |
| Privileged oracle AUEBC | 0.007802877380960266 |
| Fixed minus learned | -0.0010582009874103384 |
| Synchronized 95% bootstrap CI | [-0.0020696313885150072, -0.000059502896739864196] |
| Improved held-out domains | 2/6 |
| Oracle-gap closure | -0.0508944066361726 |

Fixed methods by outer target were RANDOM for `74t7kcdgkr`, `w68dtmpfyf`,
`xcmzfsbd9t`, and `yfxyg8jm46`, and CENTER_FIRST for `cgtnjyggtm` and
`ykhs7s2dck`. Domain effects were -0.0007862972770161765,
-0.0017194494528372086, -0.002738490103816066, +0.0004135047827370371,
-0.0018737163157090504, and +0.0003552424421794334 in the registered domain
order.

The learned observable CAI policy is statistically worse overall, improves only
two held-out domains, and has negative oracle-gap closure. The CAI component
therefore fails all three policy-GO conditions.
