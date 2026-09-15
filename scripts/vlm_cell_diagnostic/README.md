# 单例VLM选格/坐标诊断

固定任务VLM_CELL_DIAG_R1_8778aa53，输出在results/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53/index.html，交接在artifacts/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53/VLM_CELL_DIAGNOSTIC_HANDOFF.md。

- export_cpu.py：只读恢复指定q24-48输入、缓存和坐标；不运行研究模型。
- replay_attention.py：严格前缀、单例Qwen诊断；现已有一次成功attempt，默认拒绝重复运行。不能作为批处理或自动重跑入口。
- plot_attention.py：只读已导出attention数组，共享色标、nearest显示。
- build_index.py：标准库静态HTML生成器，不需要服务端。
- validate_exports.py：四项有限检查，无模型计算。

已执行研究诊断Qwen forward=1，其他模型/训练/TEST/bootstrap=0。不用generate，不修改生产后端。科学解释、资源实际峰值和复现边界见交接。临时浏览器仅作页面QA，不是项目运行依赖。
