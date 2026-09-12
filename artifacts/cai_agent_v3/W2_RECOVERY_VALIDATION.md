# W2恢复定向验证

所有测试使用CPU≤4线程。合成权重在pytest tmp目录，接线测试禁用optimizer.step；实际训练/VLM/TEST=0。

```bash
python -m pytest -q -p no:cacheprovider tests/test_cai_agent_v3_w2_recovery.py tests/test_cai_agent_v3.py -k 'w2_recovery or metric_matches_independent or prediction_at_budget or exact_cost_checkpoint or native_cost_adapter or validation_routes_retain or policy_legality_uses_exact or failed_predictor_gate or resource_limit or failed_policy_expansion'
```

退出码：0

```text
...........................                                              [100%]
27 passed, 24 deselected in 7.00s
```

```bash
python -m ruff check src/cmc_bbdm/cai_agent_v3/checkpoint_selection.py src/cmc_bbdm/cai_agent_v3/predictor_training.py src/cmc_bbdm/cai_agent_v3/actor_training.py tests/test_cai_agent_v3_w2_recovery.py scripts/audit_cai_agent_v3_w2_recovery.py
```

退出码：0

```text
All checks passed!
```

```bash
git diff --check
```

退出码：0

```text
```

独立参照已执行：`python docs/cai/v3/metric_reference.py --self-test`，退出0，7个fixture通过；production手算A=8/A=6及无未来插值在上述pytest中核对。未重复已完成的真实旧模型评分。

初始失败记录：3项因归档模块不存在失败；非整除测试最初错误假设floor分格，核对冻结的nearest-even协议后修正独立格宽预期，未修改生产分格或放宽容限。阶段入口测试曾错误假设所有阻断返回包含actual_optimizer_updates，后按实际API检查阻断状态、禁止数据/模型加载及无模型/ledger输出；没有为测试修改生产API。

行为证据：10/2/6 MPa的三份合成checkpoint全部重新评分后选择第二份；删除首个非赢家使重评失败；VALID特征/标签/mask/成本变化失效；真实MEAN_SC合成权重保存与重载逐tensor相同；四次无改进早停保留全部五个时点；原生成本类型错误在构造optimizer前被拒绝；READY但缺真实归档不能加载下游回报模型；22,264已用的临时ledger使W2在加载数据前拒绝。

## 中断记账修复后的最终验证

预留回归在修改前得到22,264（错误漏计），修改后未完成为24,264、完成1500步后为23,764。均为临时ledger，无真实优化。

```bash
python -m pytest -q -p no:cacheprovider tests/test_cai_agent_v3_w2_recovery.py tests/test_cai_agent_v3.py -k 'w2_recovery or metric_matches_independent or prediction_at_budget or exact_cost_checkpoint or native_cost_adapter or validation_routes_retain or policy_legality_uses_exact or failed_predictor_gate or resource_limit or failed_policy_expansion'
```

退出码：0

```text
............................                                             [100%]
28 passed, 24 deselected in 6.71s
```

```bash
python -m ruff check src/cmc_bbdm/cai_agent_v3/checkpoint_selection.py src/cmc_bbdm/cai_agent_v3/predictor_training.py src/cmc_bbdm/cai_agent_v3/actor_training.py tests/test_cai_agent_v3_w2_recovery.py scripts/audit_cai_agent_v3_w2_recovery.py
```

退出码：0

```text
All checks passed!
```

```bash
git diff --check
```

退出码：0

```text
```
