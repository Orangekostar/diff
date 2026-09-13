# W2正式重放有限需求审查

审查方式：SELF_CHECK_WITH_EXTERNAL_NUMERIC_ORACLE。主代理负责代码差异、科学契约及最终决定；不声称独立审查者。
依据：本轮执行规范、review A1–A8、原v3 metric_reference/golden、实际源代码和临时合成产物。

| ID | 原条款 / 实际symbol | 独立预期与实际证据 | P0状态 / 后续 |
|---|---|---|---|
| A1 | §3 / ReplayContext.begin_stage/check_allocation | 22264+12000=34264；模拟A6000后B6000可分配，超1拒绝；Actor代码无diff | PASS |
| A2 | §1.2/2 / resolve_output、run_predictor_candidates、OOF | 旧路径拒绝；合成checkpoint只写新RUN；prepare真实project_root只读六shard | PASS；最终核对旧目录diff |
| A3 | §4.2/5.3 / evaluate_predictor、validation_identity | 独立golden A=8、尾段A=6通过；prepare逐数组一致；成本float64/rint/1e-12未改；A六域等权，gate物理池化 | PASS |
| A4 | §5/6 / _train_candidate、_model、_sample_training_masks | diff未改结构、seed、Huber、AdamW、mask混合、2000/250/patience4；无旧权重warm start | PASS；首真实250核对 |
| A5 | §5.1/5.2 / CheckpointArchive、ReplayContext.checkpoint | 合成非赢家保留与缺失拒绝通过；optimizer.step禁用，latest RNG/optimizer可读；每时点预测仅写一次 | 合成PASS；真实NOT_APPLICABLE_YET |
| A6 | §5.3/6 / choose_common_predictor、require_predictor_selection_evidence | 现有gate/结构规则未改，OOF只读新RUN；fit组与常量原函数复用、prepare固定分折一致 | 代码PASS；真实NOT_APPLICABLE_YET |
| A7 | §3.1 / update_usage、ReplayContext | STARTED2000完成1500仅计1500、另中断job占2000；PROGRESS actual=0；历史750仍占用 | PASS；最终核对真实ledger |
| A8 | §7–9 / summarize / Git交付 | summarize只读预测无推理，旧史不覆盖；训练前commit待下一步，真实模型和最终push尚未发生 | NOT_APPLICABLE_YET |

P0必要检查无FAIL。真实归档/科学数值与最终交付未预先宣称通过。
实际检查命令：
```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_cai_agent_v3_w2_replay.py tests/test_cai_agent_v3_w2_recovery.py::test_training_loop_archives_each_synthetic_selection_point_without_updates tests/test_cai_agent_v3_w2_recovery.py::test_archive_preserves_losing_weights_and_replays_all_candidates tests/test_cai_agent_v3.py::test_v3_metric_matches_independent_golden_cases
python -m ruff check src/cmc_bbdm/cai_agent_v3/w2_replay.py src/cmc_bbdm/cai_agent_v3/w2_replay_results.py src/cmc_bbdm/cai_agent_v3/predictor_training.py tests/test_cai_agent_v3_w2_replay.py scripts/run_cai_agent_v3_w2_replay.py
```
实际输出：`9 passed in 5.85s`；`All checks passed!`。继承前轮28项结果，不运行Actor前向或全库测试。prepare-run已执行，固定输入计数/VALID身份见RUN/input_reuse_manifest.json。


## A完成检查

A4/A5/A6 PASS：首250运行时证据见first_checkpoint_A_MEAN_SC.json；三模型21份完整归档，选中权重逐tensor与对应参选文件相等。`audit_saved_evidence.py A`实际执行通过，逐状态保存预测独立复算所有参选指标，最大绝对差1.42e-14；未额外前向。见saved_evidence_review_A.json与RUN/preparation_conditions_A.csv。
三模型准备均通过，P_all按固定A为MEAN_SC@1750。A7 PASS：实际5250，GPU209.116秒，累计27514；B自身6000可以分配，不再次要求12000。


## B完成及最终需求核对

| ID | 最终证据 | 结论 |
|---|---|---|
| A1 | A5250/B5750；累计33264<34264；各阶段<6000；旧Actor/gates/models文件无diff | PASS |
| A2 | 新RUN全部产物；git diff 292b1c74 -- new_protocol为空，旧失效产物保留；入口main未跟踪docs保留 | PASS |
| A3 | 44点保存预测独立复算，最大差1.42e-14；硬成本float64、固定VALID身份一致；未改阈值/汇总口径 | PASS |
| A4 | 六个固定seed/结构，fit常量独立核对、参数量记录；A实际2000/1750/1500，B2000/1750/2000，均正常完成或合法早停 | PASS |
| A5 | 44点完整归档、44份同次预测、6赢家逐tensor匹配、6latest optimizer/RNG；首250即时证据均真实 | PASS |
| A6 | 新P_all=MEAN_SC@1750；B按新RUN证据进入，3折fit/query组隔离与常量核对通过；OOF query覆盖TRAIN161、无TEST key | PASS |
| A7 | 旧ledger77行逐字保留；新6次预留逐一结算，无未结算新job；actual11000，累计33264，GPU323.883882秒<21600 | PASS |
| A8 | 真实VALID/OOF表和两PNG、final_manifest/结果指针/交接齐；旧史不改、无下游操作；Git跟踪与push待最后核验 | LOCAL_PASS / REMOTE_PENDING |

`audit_saved_evidence.py B`实际通过；同A无模型前向或优化。OOF三折真实准备均通过，但无工程误差限，不宣布工程达标或Agent收益。两PNG已视觉检查，标识VALID与TRAIN OOF，单位/图例完整、无裁切。
规范§0–9核对：任务续接、隔离、C01–C10薄适配、34264/12000/6h授权、P0、A、条件B、结果表图、有限review均完成。§9剩余动作仅实际跟踪所有模型和push/SHA核验。历史37份权重仍缺，不伪造补齐；新44点不追认旧W2–W4。
