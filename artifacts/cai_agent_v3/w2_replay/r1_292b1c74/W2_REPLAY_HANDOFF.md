# W2正式重放交接

## 任务绑定

来源：用户指定 `/home/ww/diff/docs/W2正式重放/CAI_V3_W2_FORMAL_REPLAY_CODEX_PACKAGE.zip`。10个文件已逐字复制并核对。
已实际读取主规范：`/home/ww/diff/.worktrees/cai-vlm-agent-v3-controlled-reuse/docs/cai/w2_formal_replay/CODEX_CAI_V3_W2_REPLAY_EXECUTION.md`；配套同目录review、SOURCE_BINDINGS、授权JSON、README，以及原 `docs/cai/v3/` 科学规范、metric_reference/golden；恢复交接/证据/验证日志与真实全局ledger均按绑定读取。
分支：research/cai-vlm-agent-v3-controlled-reuse；入口292b1c74是00ac8fb之后的恢复提交。训练前提交95be0cf已绑定本轮代码、授权、任务和P0review；两个GPU session记录实际完整执行SHA。没有reset、PR、merge或force push。

## 完成状态

- implementation_status：W2_EXECUTION_COMPLETE，薄入口接入原训练循环；完成任务不代表Agent或工程成功。
- checkpoint_selection_status：6个新run COMPLETE，44个实际参选权重完整，6个赢家实际重载核对，44份逐状态预测可复算。
- predictor_readiness：PREDICTOR_READY，MEAN_SC@1750，VALID A=48.51707710207723。
- reward_readiness：REWARD_MODELS_READY，OOF fold0/1/2分别选1250/750/1000，各自准备条件通过。
- 真实新增更新11000（A5250+B5750）；GPU活动323.883882秒；累计上界33264/34264，未知750仍占用。
- 旧W2–W4科学失效保留，37份历史权重未恢复。新W2不解锁旧Actor或GDFS；新VLM/TEST/Actor任何前向均0。
- 本轮授权已足够完成固定W2任务，不需追加额度；下一策略任务仍需另行授权，不能花掉本轮余额自启。

## 修复和证据

predictor_training.py增加兼容context与同次评价details回调；w2_replay.py实现真实根目录、输出隔离、固定输入、阶段额度、一次reservation结算、每点恢复快照及首250核实；w2_replay_results.py仅从已存预测生成结果。训练前审查修正了赢家重载可能覆盖最后时点预测的问题；已有独立保存/固定VALID逻辑直接复用，未重做float32修复。
P0 9项定向测试与Ruff通过，继承旧28项，不跑全库；A/B实际保存证据审查通过，最大数值差1.42e-14，未额外前向。六个首250产物与saved_evidence_review_A/B.json记录实际核对。审查方式SELF_CHECK_WITH_EXTERNAL_NUMERIC_ORACLE，主代理最终负责，不虚构独立reviewer。

## 产物与复用

RUN=`results/cai_agent_v3/w2_replay/r1_292b1c74/`；ART=`artifacts/cai_agent_v3/w2_replay/r1_292b1c74/`。
入口指针ART/RESULT_POINTER.json → RUN/final_manifest.json。比较表predictor_comparison.csv、oof_comparison.csv；逐条件表preparation_conditions_A/B.csv；进度predictor_training_progress.csv、oof_predictor_training_progress.csv；OOF折表及10,883条query状态；域/成本诊断predictor_cost_metrics.csv；query诊断oof_query_metrics.csv；两张PNG。具体结果和口径见W2_REPLAY_RESULTS_AND_BOUNDARIES.md。
models/predictor_*.pt与reward_predictor_mean_sc_fold*.pt为6份新赢家；models/selection_history/*/selection.json与update_*.pt为44点完整选择证据，latest_training_state.pt为6份最小容错状态。candidate_state_predictions/保存44份同次VALID前向NPZ，未上传任何临时合成测试模型。
数据名单/split/标签来源、六shard、精确VALID库、VLM缓存（含失败项）只读复用；未重做W0/W1/编码/调用。完整Ridge复用原输入身份引用，旧部分Ridge不作新的exact比较或准备判定。本轮全部新模型从头训练，旧模型不作warm start。旧new_protocol目录与原失效审查无修改。

## 已实现命令

在实际工作树执行，训练已完成，不应再优化同一job。正常完成同身份job可只读复用；不完整job会拒绝自动重启。
```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src python scripts/run_cai_agent_v3_w2_replay.py prepare-run
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u scripts/run_cai_agent_v3_w2_replay.py train-candidates
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u scripts/run_cai_agent_v3_w2_replay.py train-oof
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src python scripts/run_cai_agent_v3_w2_replay.py summarize
```
以上四命令实际执行完成，训练stdout见train_a.log/train_b.log。summarize仅保存预测读表/绘图，不重评分模型。审查脚本audit_saved_evidence.py A/B已分别执行，不需反复运行全量哈希或前向。

## Git交付

仅本轮代码/规范/任务/新RUN/ART/global ledger追加纳入提交；原main工作树仍9c2d0f1，三个原未跟踪docs目录保留。全部真实.pt与预测NPZ按新RUN定向force-add（不是force push），没有复制六shard或原图。最终push与local/upstream/remote三方SHA在最终回复核对；不将本文件自身提交号循环回填。
