# 冻结证据接线

主规范：docs/cai/paper_evidence/CODEX_CAI_V3_PAPER_EVIDENCE_EXECUTION.md；本轮源码不修改既有训练/评价模块。

| 条款 | 新实际符号 | 来源/复用 | 独立验收 |
|---|---|---|---|
| C01/C03 同成本 | paper_evidence_math.held/aggregate/area | metrics.left_error_area_mpa；原bisect-right语义加.25查询上界 | A8/尾段6；190/210独立损失；.5拒绝 |
| C02/C06 轨迹 | paper_evidence.Evidence/validate_episode | 只读最终gzip；逐字段消费真实callback；不调用actor_selection模型接口 | 650/50/48/6与索引标签一致，篡改调用号拒绝 |
| C04/C05 完整参考 | paper_evidence_math.map_full；Evidence.__init__ | 原序feature index连接NPZ full_*，pointer核对同权重1750 | 反序标签错配拒绝，50key及原MAE/RMSE/R²复算 |
| P1 区间 | bootstrap_weights/weighted；Evidence.interval | 一次分域capture组重采，所有比较共用5000权重 | 组内成员重数相同；2/2/1物理权重得24；域/池化不同 |
| P2 等质量 | first_quality/saving；Evidence.quality | 群体曲线，固定和全事件网格分别求最小；21完整锚点 | .0625/.125/.0625；反弹/null/零分母/负收益保持 |
| C06/C08/C09 机制 | Evidence.mechanisms/texts | 原方法信息权限仅解释；真实路径与误差变化只读 | 原九方法/强无VLM/均值终点/差案例全部保留，无因果叙述 |
| C07 图稿 | paper_evidence.export | 六类matplotlib单图；引用21原PNG，不运行旧export | PNG/SVG/PDF可读，.25后无曲线；指标与CSV一致 |
| 隔离/交付 | analyze/薄CLI | 仅RUN/ART写入；输入开始/结束必要hash核对，无.pt/bank/训练入口 | 新8类测试、真实输出review、旧目录diff、实际push |
