# W3执行计划

Goal：完成本包固定W2五seed1的真实VALID pilot并上传全部证据。
Architecture：复用actor_training原训练和评价数学；兼容新增W2来源、trace与context；新w3_pilot只编排本授权，新actor_selection保存动态路线，新w3_results只读统计/图稿。
Spec：docs/cai/w3_valid_pilot/CODEX_CAI_V3_W3_PILOT_EXECUTION.md。主代理负责实现和review；不另建实验平台。

- [x] 路径/seed/轨迹薄适配（actor_training.py，tests/test_cai_agent_v3_w3_pilot.py）：新增seeded_actor，设seed后构造；_load_oof_predictors显式predictor_root；_evaluate_one在实际调用位置记录callback，evaluate_policy打包同次trace；_train_actor接context保存/重载，不增加评价前向。先用初始化次序、真实固定路线callback和不同轨迹归档反例失败，再修。
- [x] actor_selection.py：每点原子权重+csv.gz，固定环境identity、允许不同路线、完整schedule/早停，保存latest优化器/RNG，缺非赢家拒绝；合成190/194/198轨迹A8，另一轨迹A2，后者获选，删前者后拒绝。
- [x] w3_pilot.py及薄scripts入口：prepare固定输入/四模型hash/一次上游证据检查/case list/授权；每job5750总量与各方法cap、40014累计、21600秒窗口；单run预留完成结算，禁止未完成自动重试；固定方法只跑一次；五job调用原_train_actor。合成预算超1拒绝与完成复用检查。
- [x] w3_results.py：保存轨迹重算A/early/J、repeat损失均值/六域等权及终点池化/域表/配对差/gate；y200且190/210的MAE为10，不为0；图稿读取最多3个冻结案例原图与保存动作，不前向。
- [x] P0 review A1–A7必要项通过后commit；真实P1、P2首250与全参选检查；A8核对650最终episode、≤23候选、无TEST、全部权重tracked和push三方SHA。STATE保留历史，更新真实资源；不因负结果加seed或改门槛。
