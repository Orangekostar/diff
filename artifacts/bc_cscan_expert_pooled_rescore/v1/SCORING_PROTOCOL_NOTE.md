# Scoring Protocol Note

The proxy and reviewed evaluations reuse identical frozen methods, actions, reports, and STOP events, but they do not share an identical scoring protocol.

LOCATE reviewed scoring uses the expert certain-region bounding box, rejects a full-frame prediction, and enforces the registered measured-support rule. CHARACTERIZE reviewed scoring uses expert certain/uncertain regions, an uncertainty-aware area interval, and per-predicted-component measured support. Historical proxy CHARACTERIZE scoring instead compared with the full-input Reader target and did not apply that component-support check.

This pooled set contains 0 references with non-empty uncertain regions. Differences in the proxy/reviewed comparison are therefore reported as changes under both reference and protocol, not as a causal effect of annotation alone. All BC/P8 reviewed comparisons use the same reviewed evaluator.
