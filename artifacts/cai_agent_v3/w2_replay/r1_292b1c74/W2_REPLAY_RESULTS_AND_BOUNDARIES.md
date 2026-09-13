# W2正式重放结果与边界

本轮ID：W2_EXACT_COST_REPLAY_R1_292b1c74；科学执行代码基点95be0cf。只执行W2；未新增Actor前向/训练、GDFS、VLM调用、CNN编码或TEST预测/评分。

## 真实结果

W2-A：PREDICTOR_READY，P_all=MEAN_SC@1750；三候选均通过准备条件，按固定VALID A选择。
W2-B：REWARD_MODELS_READY，三个固定MEAN_SC OOF均通过准备条件。所有模型从头初始化，未热启动旧权重。
下表指标全部来自同一外部VALID50试样；A按每试样四路线再六域等权，其他MAE/MSE/RMSE/R²按物理试样池化。N列共同预测器为TRAIN/VALID，OOF为fit/query；OOF行中的指标仍为VALID，不是query成绩。

| 候选/折 | fit/query N | selected/actual update | VALID A | zero MAE | full MAE | center MAE | geometry MAE | full RMSE/R² | 准备 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MEAN_SC | 161/50 | 1750/2000 | 48.517077 | 58.550993 | 41.690010 | 48.026566 | 46.909917 | 53.944040/0.710536 | PASS |
| SPATIAL_SC | 161/50 | 750/1750 | 55.077337 | 59.657537 | 47.433969 | 52.742374 | 53.760256 | 61.897735/0.618884 | PASS |
| SPATIAL_C | 161/50 | 500/1500 | 66.434711 | 74.298280 | 59.088016 | 60.543348 | 62.090438 | 78.542156/0.386361 | PASS |
| OOF fold0 | 104/57 | 1250/2000 | 54.113979 | 64.459360 | 45.323202 | 51.560393 | 52.915826 | 57.818382/0.667463 | PASS |
| OOF fold1 | 108/53 | 750/1750 | 53.792081 | 56.021833 | 50.433891 | 53.674196 | 52.276100 | 66.499877/0.560105 | PASS |
| OOF fold2 | 110/51 | 1000/2000 | 48.345097 | 60.505811 | 35.586230 | 47.361261 | 44.833427 | 49.335497/0.757882 | PASS |

逐条件的真假、各自fit常量参照及归档完整性见RUN/preparation_conditions_A.csv与preparation_conditions_B.csv。三个OOF fit/query组数分别97/55、102/50、105/47，query互斥并覆盖TRAIN全部161个试样；fit组不含query组。每折常量只取自身fit，独立复核通过。固定分折与原交接一致。

## 单独的OOF query诊断

以下仅为TRAIN内部query折完整64格预测，未用于checkpoint选择；不等同TEST。

| 折 | query N | full MAE (MPa) | full RMSE (MPa) |
|---|---:|---:|---:|
| 0 | 57 | 50.839676 | 77.403678 |
| 1 | 53 | 52.026560 | 69.347133 |
| 2 | 51 | 31.469682 | 45.553455 |

共导出10,883条OOF query状态（零/四路线前缀/完整输入），其中完整输入161条，未包含TEST key。
两张单独PNG：RUN/valid_error_cost.png（VALID固定前缀的池化MAE，5个显示成本点连接线不是新采样），RUN/oof_full_predictions.png（TRAIN OOF query完整64格，按折着色）。均直接使用保存预测，未追加模型前向；已检查坐标、单位、图例和裁切。

## 资源与选择证据

A实际5250，B实际5750，本轮共11000；无失败/重试/额外优化预检。本轮44个参选权重（A21+B23）、6个最终选择权重、6份latest_training_state，共56个.pt文件；44份同次VALID预测NPZ。所有实际候选均保留，不按历史43个时点凑数。
每模型保存完整seed/fit组/常量、固定VALID身份、原生成本定义和评分；latest状态有model/optimizer/numpy Generator/Torch CPU及可见单GPU RNG、update/best/stale/runID。没有使用或声称完成通用续训CLI。
从保存预测独立复算所有44点，指标最大绝对差1.42e-14；未改变1e-12成本合法性/改善阈值或1e-8结构平局规则。六份最终权重逐tensor等于各自选中归档文件，正常早停/完成计划均完整。原生rint像素格、float64硬成本、模型float32输入未改。
历史已知21514+未知上界750保留；累计33264/34264，余额1000。本轮GPU活动323.883882秒（0.089968小时），低于21600秒；physical GPU1，CPU≤4，两个session完整计时。旧GPU时长仍只有下界，不将其宣称精确累计。
原账目前77行逐字保留，本轮追加记录见RUN/replay_ledger_view.jsonl；6个STARTED各预留2000，与6个COMPLETED逐一结算，PROGRESS actual=0，无漏记或双计。

## 科学解释限制

实现和新归档验收通过，预测器/OOF只达到原有后续资源准备条件。没有工程误差上限，本轮不宣称工程达标、不验证VLM或闭环Agent收益。上述新VALID full MAE不与旧v2 TEST或历史约41 MPa直接比较。
旧37份历史checkpoint仍未找回；本轮是新训练证据，未追认历史选择。旧W2–W4继续失效，旧策略/图表不因新W2就绪而恢复有效。
W3、GDFS、多seed、VLM新调用和TEST仍为NOT_AUTHORIZED_W2_ONLY。剩余1000不是下游授权，也不自动用于额外模型搜索。后续若执行策略阶段，须单独绑定新任务与额度并显式引用本RUN。
