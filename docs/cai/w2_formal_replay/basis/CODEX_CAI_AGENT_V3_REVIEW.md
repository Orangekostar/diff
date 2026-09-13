# Codex审查任务：CAI Agent v3有限需求验收

用途：在新执行上下文中交给审查者；也供实现者运行独立数值参照。不是再写一遍实现计划，不承担训练，不扩大测试范围。

## 1. 必须读取的输入

1. `CODEX_CAI_AGENT_V3_EXECUTION.md`全文；
2. `CAI_AGENT_V3_SOURCE_BINDINGS.md`中的事实/推断边界；
3. 本包`metric_reference.py`、`golden_cases.json`；
4. 实际配置、变更代码、`IMPLEMENTATION_BINDINGS.md`、当前阶段真实产物；
5. 原v2仅用于核对修复前基线，不能把其配置当本轮需求依据。

来源优先级：用户当前授权 > 本轮完整执行规范 > 配套独立数值预期。实现方自己写的配置、测试、completion checklist都不是新的需求授权。

审查者不修改任务文本、golden预期、生产代码或结果来制造PASS；只报告精确问题和最小修复要求。无法读取的文件明确写“未读/缺失”，不依据标题推断实现。

## 2. 有限审查的12个项目

| ID | 真正要验证的行为 | 必须使用的证据/反例 | 拦截什么 |
|---|---|---|---|
| A01 | 先前要求的左端阶梯面积、尾段、终点.25 | 手算A=8/J=8.5、尾段A=6/J=6.5，调用实际production与独立reference | 防止代码与测试共同改成梯形6 |
| A02 | 两seed先算各自损失，再平均 | y200，pred190/210必须独立MAE10；报告R²逐seed而非预测平均 | 防止无成本计价的集成 |
| A03 | 训练、checkpoint与评价目标有明确一致关系 | 同一合成状态序列逐项比较d_t、terminal、G_t；检查.25不是1 | 防止训练另用后误差或终点权重 |
| A04 | 完整候选与来源分组 | 读真实cohort/crosswalk/split；同原截图面板同组；实际N而非assert旧60 | 防止静默缩水和源图跨split |
| A05 | 确实有VLM，且映射一次 | 真缓存identity、highest-confidence C0、无可靠分支、第二步解锁；区分失败与无cue | 防止CNN冒充VLM或首步选错集合 |
| A06 | 先选动作后揭示，没有不可见输入旁路 | 改隐藏token/标签，保持可见状态，实际推理输出不变；gather或mask发生在全局计算前 | 防止提前看完整图、OOF标签侧漏 |
| A07 | 空间网络实际进入前向 | model repr和forward hook确认2层4头128宽contextualizer被调用；明确哪些token带坐标 | 防止定义未使用Transformer或换成MLP不说明 |
| A08 | 反馈能到达合法候选评分 | 在固定几何下检查“已测内容→未测合法候选logits”的梯度/通路；不能只检查已非法格子分数变动 | 防止历史/反馈虚接；不要求每个扰动argmax都改变 |
| A09 | 两个消融及真实静态保持其含义 | open-loop改变内部内容/预测输出不变；no-VLM改VLM不变；static换全部图像，在相同几何/预算下logits与排序不变 | 防止表面依赖Actor冒充static |
| A10 | 原生计费和不同episode终止正确 | 非8整除小图全覆盖；重复测量禁止；一件装不下不终止其他件；尾段只计一次 | 防止预算不等、批内提前结束 |
| A11 | 阶段关口真的控制入口 | 注入预测器NOT_READY或pilot失败；验证不创建Actor/后续seed/TEST任务。检查OOF组隔离及每样本采样权重 | 防止“全跑完才看VALID” |
| A12 | 统计、来源、资源与交付真实 | 小型不等组例子检验group重采后按物理N加权；独立运行而非重复N；累加失败run；实际远端SHA | 防止模糊PASS和选择性资源记账 |

不额外开展渗透、fuzz、压力、全仓库回归或新模型搜索。每个项目可用少量参数化案例；测试总数不是验收目标。

### 关于A08的边界

初始随机权重上反馈梯度非零只证明结构可达，不证明训练完成后有任务价值。完整结果中需另报真实VALID的反馈/无反馈差异。相同已测集合下，纯顺序历史未必是这套无移动成本环境的必要信息，不应为了通过测试强迫每次变历史就变动作。

空间图块交换诊断应保留相同已测位置、相同特征集合/均值，交换内容与位置的对应，并避免让“当前CAI标量变化”成为唯一通路。结果是机制诊断，不当独立性能证明。

### 关于GDFS

仅在本轮实际执行该pilot时追加到A06/A07检查，不新增一套审计：
- 原始源码同时训练predictor；本适配必须真的冻结它。
- selector看hard-visible；soft full-token分支仅用于TRAIN损失，并在方法说明中披露。
- eval为hard已测集合，已测mask/成本准确；训练soft score不能当已测数据成本。
- 反向梯度到selector，参数冻结的predictor本身不更新。
- VALID mask固定；不从外部代码继承随机变化验证集。

## 3. 审查输出

生成`REQUIREMENTS_REVIEW.md`和简短JSON，每项包含：

```text
requirement_id
stage
requirement_source_section
actual_code_path_and_symbol
independent_expected_behavior
observed_evidence / command
status: PASS | FAIL | NOT_APPLICABLE_YET | BLOCKED_INPUT
severity: BLOCKER | NOTE
approved_deviation_reference（没有则null）
```

总体状态分开：
- `protocol_conformance`：必须项是否满足，不被模型成绩左右；
- `scientific_evidence`：按真实计算结果，不因验收PASS自动SUPPORTED；
- `review_mode`：INDEPENDENT_CONTEXT_REVIEW或SELF_CHECK_WITH_EXTERNAL_NUMERIC_ORACLE，按真实执行填写。

未到达的训练阶段写NOT_APPLICABLE_YET，不谎报已完成；已到阶段缺关键检查则FAIL/BLOCKED，不能用NOT_APPLICABLE逃避。

## 4. 三个时点与是否允许继续

A. 长训练前：A01–A10的接口/小样本检查、数据范围和真实模型结构完成。核心FAIL不得训练。
B. 预测器完成后：读真实VALID与OOF，不仅看状态字段。满足主文件条件才允许Actor。
C. 总结前：复核A01/A02/A11/A12、模型与数据身份、阶段实际执行及GitHub上传。

审查发现科学输入/公式改变导致旧run无效，应保留并披露，计入累计资源；不为了交付赶时间擅自重跑到超额。

只修复受影响项，只重查相关检查；不重复整个项目。没有可用独立审查上下文时必须如实报告，不捏造另一个reviewer的意见或签名。
