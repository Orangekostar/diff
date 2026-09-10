# CAI Active Image v2 Source Bindings

Protocol: `paper_v3/configs/cai_active_image_v2.yaml` (`5ed1911bfe029a40962a4a1f391dc1ced1ddc6cff7fe8bb1f0953f69c8ce9726`)

| Source | Path | SHA-256 |
|---|---|---|
| learned_cscan_config | `paper_v3/configs/learned_cscan_same_perception.yaml` | `12268dcaf470f769c326007702f7b0b6f4a13cf326447eef652a6dda972b5662` |
| vlm_cache | `results/learned_cscan_same_perception/surface_percepts.jsonl` | `85f400119093b362e2eb76489c9ae158ecdd453ce84ad04b6dfab7f04384693e` |
| vlm_manifest | `results/learned_cscan_same_perception/perception_manifest.json` | `b87b85240ee96c31b7fd9f8b46e0ca2338c7e11065fe531bd1f9e286aa942454` |
| cai_mpa_authority | `results/multiview/e1_audit/oof_predictions.csv` | `1d65d9f7c40f561727329ab6b7f922973765cc57f6ae36b5a94c572b20e8b054` |
| resnet18_weights | `paper_v3/assets/resnet18-f37072fd.pth` | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |

External image root binding: `external:cmc_damage_inference`; supply its local path with `--source-root`.
The external ResNet18 weight copy was checked against the registered `paper_v3/assets/resnet18-f37072fd.pth` digest before encoding.
