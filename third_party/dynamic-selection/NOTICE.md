# Dynamic Selection Source Notice

- Upstream: `https://github.com/iancovert/dynamic-selection`
- Fixed commit: `e2b6f7403fdac4d217ac2ec5dea96acd60240b60`
- Upstream license: MIT, preserved in `LICENSE`
- Relevant upstream symbols: `dynamic_selection.utils.ConcreteSelector` and
  `dynamic_selection.greedy.GreedyDynamicSelection.fit`

The v3 adapter does not vendor the upstream datasets, notebooks, trainers, or
package. `cmc_bbdm.cai_agent_v3.gdfs_training` reimplements only the bounded
grouped Concrete selection idea for 64 acquisition cells. Unlike upstream
joint fitting, the v3 CAI predictor is frozen; only selector parameters are
optimized. The selector observes hard-visible state, while full hidden C-scan
tokens are exposed only to the documented training-only soft-mask relaxation.
VALID and any conditional TEST execution use hard acquisition.
