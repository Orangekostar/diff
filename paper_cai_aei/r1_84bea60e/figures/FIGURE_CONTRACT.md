# 图稿契约

Python/matplotlib；AEI作者审阅稿，约183mm宽，文本≥8pt，PDF/SVG可编辑，PNG300dpi。仅两张新增图，不生成研究影像。

Fig1（schematic）：回答哪些信息可进入逐步策略。表面特征/一次缓存VLM→可见状态→Actor→合法格→状态更新→共同预测器；预测反馈回可见状态。环境隐藏缓存仅经请求揭示单格，y仅训练/评分。图中不画全X到Actor的边，无语言逐步规划、假STOP或硬件声称。
Fig4（quantitative single axes）：四个完成阶段的signed timing-weighted contribution，全部9方法×4箱；来自唯一已保存timing_contributions.csv，不再读事件/模型/抽样。按原repeat→domain聚合，50物理/48组，95%区间未计算所以不画error bar。四阶段色彩同时以行内位置标识；负值轴完整。不是因果百分比或箱内积分。

复用F1/F4/F5及原三病例PNG按原字节复制，F2/F3/F6用于SI。所有caption准确标来源和成本1.0全输入参照，原图不重新跑QA整套。新增两图单轴/流程图，不存在可比多panel，alignment不适用；运行源码/字体/碰撞检查并视觉查看。
