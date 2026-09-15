# 本轮源码、证据与设计绑定

核对日期：2026-09-15。研究分支读取时指向`e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750`。以下为实际读取的固定来源；执行时先确认当前工作树，不把聊天摘要或旧BC附件当当前实现。

## 一、仓库事实

| ID | 实际来源 | 已核实内容 | 本轮绑定 |
|---|---|---|---|
| S1 | [当前VLM感知入口](https://github.com/Orangekostar/diff/blob/e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750/src/cmc_bbdm/cai_agent_v3/vlm_perception.py) | clean/gridded生成后传入backend；Qwen固定revision；`_features`映射0、1/3、2/3、1；旧入口写new_protocol并依赖gate | 独立runner；复用特征纯函数，不运行旧入口 |
| S2 | [表面提示词、schema、parser和cache](https://github.com/Orangekostar/diff/blob/e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750/src/cmc_bbdm/learned_cscan/perception.py) | 直接cells输出、最多2×4；resolve内部硬编码SURFACE_PERCEPT_PROMPT；缓存只存最终raw text；旧display置换函数存在但当前路径不调用 | P0取实际字符串；新prompt显式传递；初次/repair完整保存；不套入旧置换 |
| S3 | [渲染函数](https://github.com/Orangekostar/diff/blob/e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750/src/cmc_bbdm/vlm_cscan/runtime.py) | ROTATE_270后缩放、编号图copy自clean、row*8+col、左上+2/+1绘字且没有显式font大小 | 旋转一次；R1只改数字标签，几何冻结 |
| S4 | [QwenVLBackend](https://github.com/Orangekostar/diff/blob/e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750/src/cmc_bbdm/vlm_cscan/vlm.py) | cuda:0、bfloat16、sdpa、use_fast=False、256–1280视觉token约束；greedy generate；VLMRawResponse只有文本和计数 | 加载复用；薄记录真实token-ID，不能把generate计成一次decoder forward |
| S5 | [首步C0](https://github.com/Orangekostar/diff/blob/e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750/src/cmc_bbdm/cai_agent_v3/policy.py) | 最高medium/high与legal相交；不可用/no-cue/fallback有分支；首步后释放 | 纯CPU候选集合诊断，不修改真实策略 |
| S6 | [已完成诊断交接](https://github.com/Orangekostar/diff/blob/e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750/artifacts/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53/VLM_CELL_DIAGNOSTIC_HANDOFF.md) | q24-48坐标/历史overlay一致；1次新attention forward、未generate完整回答；约29.10GiB峰值；多editable导入等环境记录 | 不重复attention；输入链复用；更保守单卡预估；完整新回答与历史分开 |
| S7 | [只读导出脚本](https://github.com/Orangekostar/diff/blob/e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750/scripts/vlm_cell_diagnostic/export_cpu.py) | source_root取encoder_execution_root；manifest/hash连接；强制本工作树import；若直接main会写旧目录 | 参考局部函数，不运行main、不覆盖DIAG |
| S8 | [固定三个案例](https://github.com/Orangekostar/diff/blob/e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750/results/cai_agent_v3/w3_pilot/r1_0e11452a/case_manifest.csv) | c8-16、q24-48、q16-29均为既有VALID例 | 三例直接绑定，另外三域TRAIN按固定hash规则选 |
| S9 | [候选队列](https://github.com/Orangekostar/diff/blob/e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750/results/cai_agent_v3/new_protocol/candidate_queue.csv) | 元数据列包含split、capture_group、surface path/hash、登记C-scan宽高及CAI标签列 | 只抽表面/身份/尺寸列；标签不读入实验状态；额外三件key待本地确定 |
| S10 | [诊断身份](https://github.com/Orangekostar/diff/blob/e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750/results/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53/identity.json) | q24-48源与渲染hash、cache key、旋转方向、四候选与历史首动作 | q24身份核对，不用first action=36认定任意图片 |
| S11 | [历史回答](https://github.com/Orangekostar/diff/blob/e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750/results/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53/raw_response.txt) | 第一region36/37 circle with white speckles；第二59/60 line；均medium | H00只读、用户观察假设不作为本轮答案注入 |
| S12 | [历史attention提取](https://github.com/Orangekostar/diff/blob/e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750/scripts/vlm_cell_diagnostic/replay_attention.py) | 切在首数字前；4层均值非完整回答；带重跑阻断 | 本轮不执行，attention不作为GT |

S9读取确认了字段与队列来源；本次没有将全276行或额外三件的图像在本机重新审计，没有预先得到新选出的三件结果。本包也没有在用户GPU执行任何生成。

## 二、官方文档支持，不能当成本例效果证明

| ID | 来源 | 可用于本轮的内容 | 不应推断 |
|---|---|---|---|
| O1 | [Transformers v4.49.0 Qwen2.5-VL文档](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/qwen2_5_vl) | 多图输入建议明确图像身份；processor分辨率、图像网格与generation接口 | 未标图像ID不是调用bug；加ID或改prompt不保证定位提升 |
| O2 | [Pillow ImageDraw官方文档](https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html) | 显式font、anchor、textbbox及字框计算 | 文档不证明某个24px字体对当前Qwen最优 |

为保持因素清楚，实际矩阵只改P1最终prompt文本，不额外开启add_vision_id或改chat template；R1的24px/10px为本轮预定呈现设计，非官方推荐性能阈值。本机实现以已安装版本为准，不能依据官网最新样例自动升级生产环境。

## 三、从证据到任务的判断

| 判断类型 | 判断 | 如何执行 |
|---|---|---|
| 源码事实 | 目前没有连续候选热图→cell的后处理链 | 检查完整文本格号，不把attention绘制为真值 |
| 源码事实 | prompt传递可能被旧cache.resolve硬编码覆盖 | fake backend验证真正收到的文本，独立cache签名 |
| 风险推断 | 小标签/边界附近文字可能增加格号对应歧义 | R0/R1有控制地比较，不能直接称问题已查明 |
| 风险推断 | 原confidence未明确区分线索与定位 | P1明确定义综合把握，记录候选及C0变化，不宣称校准 |
| 新设计 | 三个诊断例＋三个TRAIN域内固定hash例 | 冻结选择，不按结果换例，有限开发检查 |
| 新设计 | 2×2包含“只改编号图” | 增加一个必要对照而非扩成算法搜索，区分两类改动 |
| 新设计 | 当前环境A与H00历史分栏 | 不把环境/生成差异藏进P1的收益 |
| 必需人类输入 | 可见线索与格号是否匹配 | 用户看图/填CSV；未填不伪造准确率也不阻塞导出 |

## 四、旧附件的作用边界

本会话附带的Hasebe证据摘要记录24件的作者面积/深度可追溯，但无作者空间mask或原始定量阵列；它还明确本地形态学描述符是proxy。此材料只用于避免把旧内部proxy误充本轮表面GT。2026-09-09交接是旧BC阶段说明，不能替换当前CAI v3分支/任务，也不恢复LOCATE、CHARACTERIZE与STOP。

## 五、完成后能与不能说的结论

能报告：四种实际输入设计、原始回答、是否格式有效、候选位置如何改变、C0如何改变、用户逐例实际评分。不能自动报告：VLM定位全面修复、已证实模型只看数字、CAI性能提升、原无VLM优势已经解释、六件独立泛化确认。后续是否接入Actor是另一项决定，需保留新旧方法身份。
