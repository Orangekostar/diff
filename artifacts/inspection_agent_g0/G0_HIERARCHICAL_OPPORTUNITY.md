# G0-B Hierarchical Acquisition Opportunity

Status: FIELD and conditional CAI hierarchical headroom gates pass.

| Task | Fixed AUEBC | Oracle AUEBC | Fixed-minus-oracle | Relative improvement | 95% CI | Domains |
|---|---:|---:|---:|---:|---|---:|
| FIELD | 0.001162681470 | 0.001072049934 | 0.0000906315358 | 7.795044% | [0.0000826309228, 0.0000994948969] | 6/6 |
| CAI | 0.026682472404 | 0.005833683797 | 0.020848788607 | 78.136645% | [0.019150257953, 0.022589405744] | 6/6 |

FIELD passes the registered alternative magnitude condition through its 48.24%
reference-quality sufficiency-budget reduction; CAI passes directly through its
relative AUEBC improvement. These are privileged teacher opportunities, not
learned-policy results.

FIELD uses 28,889 actions: 61.14% FOCUS/BROADEN and 38.86% REFINE. Exact pixel
budget is 7.08% FOCUS/BROADEN and 92.92% REFINE. Once FIELD enters REFINE, its
formal trajectory does not return to BROADEN. CAI uses 17,873 actions: 49.82%
FOCUS/BROADEN and 50.18% REFINE; exact pixel budget is 3.69% versus 96.31%.
Unlike FIELD, CAI repeatedly switches between BROADEN and REFINE.

`FIXED_UNIFORM_THEN_MAVIS` is a
`METADATA_AUGMENTED_UPPER_BOUND`, not a gate-eligible deployable comparator. Its
equal-domain AUEBC is 0.001226035237 for FIELD and 0.027918914020 for CAI; the
label denotes input privilege, not performance superiority.

Authority: `hierarchical_trajectories.parquet` SHA-256
`6b286eab5080a66cec804677944de14417f6f5c1d7b645df25348abe40bee49f`.
