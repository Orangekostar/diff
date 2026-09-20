# Actor C0 机制诊断发现

## 范围

六个固定规则 VALID 案例：`cgtnjyggtm:q24-48`、`74t7kcdgkr:c8-16`、`w68dtmpfyf:q16-29`、`xcmzfsbd9t:c24-9t`、`yfxyg8jm46:c16-24`、`ykhs7s2dck:q8-22`。本结果只描述已冻结的 C/N Actor、共同 P_all 和这些状态；不代表重训收益、全 VALID 总体、TEST 表现或因果损伤定位。

## 首步外部门控

- `C0_CHANGED_THIS_DECISION`: 3/6。
- `C0_ONLY_CHANGED_DISTRIBUTION`: 2/6。
- `C0_NOT_ACTIVE_ON_CASE`: 1/6。

外部门控并非在每件都改变确定性首动作；即使首动作不变，被排除概率质量仍可能非零。

## 仅关闭 C0 的冻结策略干预

- `cgtnjyggtm:q24-48`: A(C_NATIVE)-A(C_NO_C0)=0.000000 MPa，终点误差差=0.000000 MPa，`EXACT_NATIVE_REUSE`。
- `74t7kcdgkr:c8-16`: A(C_NATIVE)-A(C_NO_C0)=0.000000 MPa，终点误差差=0.000000 MPa，`EXACT_NATIVE_REUSE`。
- `w68dtmpfyf:q16-29`: A(C_NATIVE)-A(C_NO_C0)=-4.487909 MPa，终点误差差=0.000000 MPa，`FULL_MASK_ONLY_REPLAY`。
- `xcmzfsbd9t:c24-9t`: A(C_NATIVE)-A(C_NO_C0)=0.000000 MPa，终点误差差=0.000000 MPa，`EXACT_NATIVE_REUSE`。
- `yfxyg8jm46:c16-24`: A(C_NATIVE)-A(C_NO_C0)=0.040043 MPa，终点误差差=0.909027 MPa，`FULL_MASK_ONLY_REPLAY`。
- `ykhs7s2dck:q8-22`: A(C_NATIVE)-A(C_NO_C0)=-0.456159 MPa，终点误差差=-2.669189 MPa，`FULL_MASK_ONLY_REPLAY`。

三件实际闭环干预的面积差方向混合（最小 -4.487909，最大 0.040043 MPa）；不能把 C 与 N 的既有差异归因于 C0，也不能推出 C0 必然有益或有害。正差仅表示该案例中关闭 C0 后误差面积更小。

## 固定状态先验通道

54 个共同物理状态中，四个 VLM 输入置零后有 5 个 top-1 改变；TVD 中位数 0.015691，最大 0.026723。这是同权重同状态的通道敏感性，不是 NO_VLM 重训对照。

## 表面特征与 Attention

- 72/72 个固定 10% 特征扰动产生非零 target log-probability 变化，最大绝对变化 0.374235。
- direct/total forward 最大 logit 差 `1.9073486e-06`，P_all 当前预测最大复现差 `3.0517578e-05 MPa`；`via=total-direct` 仅表示经当前预测标量的局部路径。
- Actor attention 为 2 层 x 4 头 x 65 token；保留 query self-mass，rollout 明确为近似。它不是 Qwen attention，也不是表面因果归因。

## 结论边界

本诊断分别观察到了 C0 门控、固定状态先验通道和表面 direct/indirect 局部敏感性；三者可区分，但本六件结果不支持单向效果结论、显著性结论或重训结论。
