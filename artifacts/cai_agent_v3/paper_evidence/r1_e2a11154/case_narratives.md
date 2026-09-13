# 三个预先冻结VALID案例

同k仅描述过程，数值效率对照用共同b。案例不按新结果筛选，保留q24-48。

## 74t7kcdgkr:c8-16

冻结表面/VLM首步候选记录为 HIGHEST_RELIABLE_CONFIDENCE_C0；调用 1 在无C-scan状态选 cell 36，取得 1848 原生像素，成本 0.01553254。预测由 301.366425 变为 333.240997 MPa；分析侧标签 264.163452，该次误差减少 -31.874573 MPa。随后调用 2 的规则为 C0_RELEASED_AFTER_FIRST_ACTION，选择 cell 48；这证明逐步记录，不证明模型理解了损伤边界。

不利部分：第 1 次观测（cell 36）误差减少 -31.874573 MPa；负值即误差增加。末预测 239.994385，绝对误差 24.169067 MPa。没有记录语言思维链或注意力；离开C0不称纠错，不能据关联断定C0造成误差。原七张图和所有同成本对照见 case_figure_reuse.csv、case_same_cost_states.csv。
## cgtnjyggtm:q24-48

冻结表面/VLM首步候选记录为 HIGHEST_RELIABLE_CONFIDENCE_C0；调用 1 在无C-scan状态选 cell 36，取得 7056 原生像素，成本 0.01550940。预测由 302.099304 变为 213.671692 MPa；分析侧标签 136.680008，该次误差减少 88.427612 MPa。随后调用 2 的规则为 C0_RELEASED_AFTER_FIRST_ACTION，选择 cell 48；这证明逐步记录，不证明模型理解了损伤边界。

不利部分：第 3 次观测（cell 28）误差减少 -43.722580 MPa；负值即误差增加。末预测 242.696823，绝对误差 106.016815 MPa。没有记录语言思维链或注意力；离开C0不称纠错，不能据关联断定C0造成误差。原七张图和所有同成本对照见 case_figure_reuse.csv、case_same_cost_states.csv。
## w68dtmpfyf:q16-29

冻结表面/VLM首步候选记录为 HIGHEST_RELIABLE_CONFIDENCE_C0；调用 1 在无C-scan状态选 cell 27，取得 7140 原生像素，成本 0.01569403。预测由 326.637573 变为 342.268188 MPa；分析侧标签 338.548340，该次误差减少 8.190918 MPa。随后调用 2 的规则为 C0_RELEASED_AFTER_FIRST_ACTION，选择 cell 24；这证明逐步记录，不证明模型理解了损伤边界。

不利部分：第 10 次观测（cell 28）误差减少 -13.061676 MPa；负值即误差增加。末预测 316.461456，绝对误差 22.086884 MPa。没有记录语言思维链或注意力；离开C0不称纠错，不能据关联断定C0造成误差。原七张图和所有同成本对照见 case_figure_reuse.csv、case_same_cost_states.csv。
