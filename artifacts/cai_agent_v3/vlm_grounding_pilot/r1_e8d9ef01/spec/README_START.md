# 启动：VLM提示词与编号呈现的有限定位对照

本包用于执行小规模开发诊断，不修改现有CAI论文方法。主规范已明确分支、代码依据、6件案例选择、4配置、调用额度、结果图与GitHub交付，无需另外寻找旧提示词。

## 文件

- `CODEX_VLM_GROUNDING_PILOT_EXECUTION.md`：主执行规范，含P1完整文本与全部边界。
- `VLM_GROUNDING_SOURCE_BINDINGS.md`：固定代码/文档来源与事实—设计区别。
- `VLM_GROUNDING_REVIEW.md`：六项有限验收。
- `VLM_GROUNDING_SCOPE.json`：机器可读范围。
- `prompts/P1_SPATIAL_GROUNDING_ZH.txt`：P1正文；执行时按主规范追加原schema。

## 发给Codex

```text
请完整读取并执行附件 CODEX_VLM_GROUNDING_PILOT_EXECUTION.md。

在Orangekostar/diff的research/cai-vlm-agent-v3-controlled-reuse分支，
以e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750为已核对基点，
实际完成VLM表面线索与格号对应的有限对照，不只给修改建议。

固定6件：q24-48、c8-16、q16-29三个既有VALID例，
以及另三个域各一个按主规范hash规则选择的TRAIN例。
运行A原prompt+原图、B新prompt+原图、C原prompt+清晰编号、
D新prompt+清晰编号四配置；所有配置同模型、同方向、同clean。
原历史缓存另列H00，不替代本轮A。

本轮授权24个主Qwen完整回答、每项最多一次格式修复，
含失败总尝试不超过48；单卡累计1800秒、CPU最多4线程。
一次generate可能多次forward，计数如实区分。
训练、Actor/CAI/ResNet/Reader/STOP、新attention、新TEST均为0。
不要运行旧run_vlm_perception或硬编码原prompt的cache.resolve。

输出完整原始回答、真实输入、独立候选PNG、C0集合诊断、
可本地打开的index.html和作者评价模板。
未收到人评则标PENDING_HUMAN_REVIEW，但完成全部可做工作并推送。
禁止把23/31等用户观察送入prompt，禁止自动造GT或宣称CAI提升。

原论文、旧缓存、src生产代码和旧统计全部不改。
完成后将新代码、图、raw回答、HTML和交接MD实际commit/push，
核对local/upstream/remote SHA一致。不PR、merge或force push，
不做全库安全测试，不因未得到预期正结果而反复调整prompt。
```

## 预期拿到什么

重点看新`index.html`：同一试样的真实clean、原编号、新编号与四组候选，可直接分辨“提示词变化”“标签呈现变化”各带来什么。没有人类参考前，输出改变不是定位改善；不将开发诊断与原CAI结果混用。

本包自身只生成规范文件，没有运行Qwen、修改仓库或推送研究代码。字体和模型不随包分发。
