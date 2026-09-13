# W3 seed1受控实验交接

## 任务绑定与执行来源

任务ID `W3_VALID_PILOT_R1_0e11452a`；实际工作树 `/home/ww/diff/.worktrees/cai-vlm-agent-v3-controlled-reuse`，分支 `research/cai-vlm-agent-v3-controlled-reuse`。
主规范 `/home/ww/diff/.worktrees/cai-vlm-agent-v3-controlled-reuse/docs/cai/w3_valid_pilot/CODEX_CAI_V3_W3_PILOT_EXECUTION.md`；同目录 REVIEW、AUTHORIZATION、SOURCE_BINDINGS、HAND_COMPUTED_CASES、basis 为配套。来源 `/home/ww/diff/docs/W3 seed1受控实验/CAI_V3_W3_VALID_PILOT_CODEX_PACKAGE.zip`，13份文件逐字一致。

输入基点 `0e11452ac6590b3b2b694364bd4d1cef7c9315cf`；训练前主体代码提交 `27ceee5`，实际训练代码SHA `0f18001c8fa25e83b6fd3ece2fa6c7cd74e6c5ff`（仅后续修正冻结CSV行尾）。训练之后源码/脚本/测试没有变化。后续 W2 零训练复核提交 b00c2e8 保留；本次明确继续 W3 原目标，完成余下 review/上传，不重启已完成 job。

DATA=`results/cai_agent_v3/new_protocol/`、W2=`results/cai_agent_v3/w2_replay/r1_292b1c74/` 只读；RUN=`results/cai_agent_v3/w3_pilot/r1_0e11452a/`；ART=`artifacts/cai_agent_v3/w3_pilot/r1_0e11452a/`。全局账目 `results/cai_agent_v3/compute_ledger.jsonl` 仅追加，旧137行字节前缀保持，W3视图37行，无清零或未结算新预留。

## 实现与真实完成证据

复用原 actor_training 优化/采样/OOF路由/数学，通过兼容可选路径和context接入新W2；seeded_actor在构造前设置Python、NumPy、Torch CPU与可见CUDA随机源。五方法信息开关、2层4头128维空间结构、64共享logits真实静态保持；训练批16/AdamW3e-4/wd1e-4/clip1、cost-to-go/终点0.25/value0.5/entropy0.01→0保持。

actor_selection保存每个实际checkpoint权重及同次完整VALID轨迹；固定环境身份与各自动态路线分别绑定，不能要求跨checkpoint动作相同。latest含模型/optimizer、NumPy Generator及全局/Python/Torch CPU/CUDA RNG、update/best/stale、runID/输入身份；异常不自动重启，不声称已验证自动续训。每个job首250立即检查真实hash/轨迹/ledger/RNG，再继续同进程；五份 `first_250_*.json` 全部 FIRST_250_CONFIRMED。

五个已授权 job 全部达到各自上限，无失败或重试：主/无VLM/开环/均值各1250，静态750。training_seed依次2026091301/02/03/05及静态04；见 initialization_summary.csv 的完整初始hash/结构/参数量。23个实际参选点全部保留（5+5+5+3+5），5赢家从真实归档重载，并复用其同次轨迹。共33份.pt：23候选+5赢家+5latest，23份候选轨迹gzip；最终650条episode，无删除样本。

P_all及OOF实际消费权重和来源见 INPUT_BINDINGS.json；P_all MEAN_SC@1750，OOF@1250/750/1000。队列276/259组，TRAIN161/152组，VALID50/48组，TEST65/59组；本轮学习/前向/评分仅TRAIN与VALID。211 VLM记录全部保留（205可用+6不可用），新VLM调用/编码/W2训练/GDFS/STOP/seed2、3/TEST预测或标签评分均0。

w3_results只读保存轨迹汇总和绘图，不调用模型。固定四方法400条轨迹只执行一次，三件原先冻结VALID案例共21张独立PNG；case_manifest记录名单，preparation_revision记录任何新评分前对原已有名单的绑定修正。

## 方法结果与解释

| 方法 | A (MPa) | early A | 终点 MAE | 所选 update | 实际更新 |
|---|---:|---:|---:|---:|---:|
| CENTER_FIRST | 48.475341 | 51.820645 | 48.026569 | 0 | 0 |
| GEOMETRY_SPREAD | 47.310645 | 51.879792 | 46.909917 | 0 | 0 |
| LEARNED_STATIC_TRUE | 47.976721 | 52.945606 | 46.993118 | 250 | 750 |
| NO_VLM_SPATIAL_FEEDBACK | 43.597088 | 48.770837 | 42.384681 | 1000 | 1250 |
| RANDOM | 49.594392 | 54.116690 | 46.886383 | 0 | 0 |
| SERPENTINE | 48.911416 | 52.170184 | 48.012802 | 0 | 0 |
| VLM_MEAN_FEEDBACK | 45.262566 | 50.389752 | 43.610874 | 250 | 1250 |
| VLM_SPATIAL_FEEDBACK | 45.110377 | 49.642437 | 44.285799 | 250 | 1250 |
| VLM_SPATIAL_OPEN_LOOP | 46.500981 | 51.047975 | 44.821928 | 750 | 1250 |

Δ_fixed=+2.200268，Δ_feedback=+1.390605，Δ_vlm_early=−0.871600，Δ_spatial=+0.152189 MPa。原pilot条件true，但VLM早期方向为负且无VLM整体A更低；不宣称VLM增益、独立确认或工程达标。完整限制见 RESULTS_AND_CLAIM_BOUNDARIES.md，逐域/配对及绝对成本点表全部保留。

## 资源、检查与完成条件

W3真实新增 **5750/5750更新**；全局保守已用 **39014/40014**（保留旧未知750），剩余1000仍属W2、不挪用。新增GPU **2595.753675/21600秒**，固定8.415237+训练2587.338438秒，约43.26分钟；仅物理GPU1（逻辑cuda:0），CPU≤4。旧GPU窗口部分时长未知，不能由本窗口推算旧余额。

| 类别 | 累计上限 | 保守已用 | 剩余 |
|---|---:|---:|---:|
| 共同预测器 | 11000 | 10250 | 750 |
| OOF预测器 | 11750 | 11500 | 250 |
| 主/无VLM/开环 | 11250 | 11250 | 0 |
| 静态（含旧未知750） | 2250 | 2250 | 0 |
| 均值诊断 | 2500 | 2500 | 0 |
| GDFS诊断 | 1250 | 1250 | 0 |
| 历史优化预检 | 14 | 14 | 0 |

review方式为 SELF_CHECK_WITH_EXTERNAL_NUMERIC_ORACLE，主代理负责最终审查，没有捏造独立reviewer。P0 20项定向CPU行为检查和Ruff通过；具体命令/独立预期见 PREFLIGHT_REVIEW.md 及 REQUIREMENTS_REVIEW.json。实际首250及所有参选轨迹由 audit_saved_results.py 核验，最大A/J重算差5.684e-14；五赢家逐tensor一致，缺非赢家的合成归档被拒绝。最终成本点/R²/域表另以保存数据复算，最大数值差9.095e-13；有限账目、只读目录、文件数量与21图视觉检查见 FINAL_SCOPE_REVIEW.json。没有全库回归或上游模型前向重评分。

### 实际命令与输出

以下在上述真实工作树执行；三个运行/汇总命令已完成，**不要为复现交接而重新训练**。

```bash
# 已执行一次；本W3任务的固定路线评价和五个正式训练job
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u scripts/run_cai_agent_v3_w3_pilot.py run-fixed
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u scripts/run_cai_agent_v3_w3_pilot.py train-pilots
# 已执行；仅保存轨迹汇总和最多三件图稿，无模型前向
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src python scripts/run_cai_agent_v3_w3_pilot.py summarize
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src python scripts/run_cai_agent_v3_w3_pilot.py export-figures
# 已执行的保存证据检查；无模型前向或更新
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src python artifacts/cai_agent_v3/w3_pilot/r1_0e11452a/audit_saved_results.py final
```

run_fixed.log/train_pilots.log/final_audit.log/export_figures.log均为实际退出0的对应输出。CLI --help已实际核对，仅prepare-run/run-fixed/train-pilots/summarize/export-figures五入口。再次summarize会重写生成表和将final_manifest的review状态设回待审；它不是自动最终验收命令，不会训练。

### Git交付

本轮按RUN精确 `git add -f` 上传全部33份真实模型、23份同次候选轨迹、固定与最终轨迹、表格/21张图；没有合成模型进入正式目录。最大单文件<50MiB。原W2/特征已有远端对象不重复复制，主工作树未跟踪docs保持。结果代码及协议来源SHA如上；实际结果提交和三方远端核对记录在 GIT_DELIVERY.json，最终含该记录的SHA在回复报告，避免自引用提交循环。

本轮无剩余训练任务；达到pilot规则不自动授权GDFS/TEST/扩seed。交付与协议状态以最终review和Git核对为准，科学结论始终限于VALID pilot。
