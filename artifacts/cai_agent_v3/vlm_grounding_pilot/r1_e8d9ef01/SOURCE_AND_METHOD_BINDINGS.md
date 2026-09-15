# 源码与本轮方法绑定

基点e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750；任务规范完整副本见spec/。本轮6×4是开发诊断，非论文方法更新。

| 依据 | 实际复用与边界 |
|---|---|
| learned_cscan/perception.py | 原SURFACE_PERCEPT_PROMPT/schema/parser/FORMAT_REPAIR及原context；新增合同仅拦跨region重复及空regions与no-cue矛盾。未改生产parser或旧resolve。 |
| vlm_cscan/runtime.py | 每源一次render_surface_inputs(max_edge=1024)，原PNG压缩hash；全部六件clean/R0与历史request哈希一致。R1从clean复制，保留同网格线，显式字体/内缩/面板。 |
| cai_agent_v3/vlm_perception.py | 实际_MODEL_PATH/revision、_features只读；不运行run_vlm_perception。 |
| vlm_cscan/vlm.py | 原QwenVLBackend.load；独立runner复用等价infer消息、模板、processor与generate参数，增加落盘和计数而非新科学配置。 |
| cai_active_image/environment.py | 登记C-scan宽高建立NativeCellGrid；仅legal_mask(set(),endpoint_budget=.25)，不打开C-scan图。 |
| cai_agent_v3/policy.py | CPU vlm_first_action_mask，action_count=0、use_vlm=True；最高medium/high与legal相交或原fallback，非执行动作。 |
| new_protocol/candidate_queue.csv | 只将白名单身份/表面路径hash/登记尺寸列投影入状态，三VALID固定、三域TRAIN固定SHA256排序；不依据标签/旧质量选样。 |
| new_protocol/vlm_actor_features_fit.csv、vlm_surface_percepts.jsonl | 精确六件cache key只读H00，完整历史最终raw单列。H00不能代替当前A，也无历史首答重建。 |
| feature_bank_manifest.json | encoder_execution_root连接外部源路径，源hash核对。未打开内部图或CAI工作簿。 |

实际模块__file__全部绑定当前工作树，见experiment_lock.json::modules。系统torch2.12.1+cu130、transformers4.49.0、Pillow12.3.0与前诊断一致；不升级依赖。模型revision固定cc594898137f460bfe9f0759e9844b3ce807cfb5，bf16/SDPA/greedy/use_fast=False/200704–1003520 pixels/batch1/maxnew500，clean先编号后，原chat template，不增加vision ID。

P1为固定文字＋原序列化schema，改变角色/坐标/对应置信等一组规则；R1改变标签可读性/位置一组规则，不能分解到单句/字号的因果贡献。字体实际框64×6全部在所属格内；R0/R1在声明标记mask外像素相同。标签仍遮挡部分表面，原clean保留，不预设R1更优。

运行前的源/缓存关键hash及ledger原长度/前缀hash见protected_before.json；源码、旧DIAG/W2/W3/论文仍只读。用户观察只存USER_HYPOTHESES.md，不进入prompt。没有人评时不得从集合变化宣称定位、损伤或CAI改善。
