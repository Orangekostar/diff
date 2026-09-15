# VLM_GROUNDING_PILOT_R1_e8d9ef01 交接

**RUN_COMPLETE_REVIEW_PENDING / RESEARCH_PRODUCTION_UNCHANGED**。已实际完成6件×4配置生成、独立诊断与可视化，人评缺失不阻塞交付。原分支research/cai-vlm-agent-v3-controlled-reuse；起点e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750。

## 直接查看

- HTML：[index.html](../../../../results/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01/index.html)
- q24 A：[独立PNG](../../../../results/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01/overlays/cgtnjyggtm__q24-48/A_P0_R0.png)
- q24 D：[独立PNG](../../../../results/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01/overlays/cgtnjyggtm__q24-48/D_P1_R1.png)
- q24 D：[完整raw](../../../../results/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01/runs/cgtnjyggtm__q24-48/D_P1_R1/raw_1.txt)
- 作者：[评价模板](../../../../results/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01/human_review_template.csv)、[评价说明](../../../../results/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01/HOW_TO_REVIEW_ZH.md)

相对路径从本文件所在ART返回工作树ROOT；HTML纯本地相对资源，无CDN/后端。真实输入是clean/R0/R1，叠图另标diagnostic reference。每配置独立PNG、序数置信图、64格CSV、C0集合JSON、首次和必要repair的raw/token-ID/input/chat状态均有保存。

## 实际执行

案例：三个固定VALID q24-48、c8-16、q16-29；三个固定hash域内TRAIN c24-43、c16-20、q8-3。无替换样本。A原prompt原编号，B新prompt原编号，C原prompt清晰编号，D新prompt清晰编号，H00历史只读。

27次generate=24主+3格式修复；全部最终合同有效，21首次有效。3修复均首次跨region重复格，原parser-valid但contract-invalid。2508实际输出token、2508实际Qwen模型顶层forward（轻量pre-hook），没有额外attention输出或前向。一次加载GPU0、CPU≤4、batch1；总332.806秒（包含加载及全部生成/修复），峰值29.100GiB，均在1800秒/单项120秒额度内。训练和所有其他研究模型/TEST/bootstrap为0。资源、attempt与ledger一行相互对应。

P0/R0全部六件精确历史hash，六件新A最终raw与历史一致。P1/R1引起多件候选变化，但没有定位GT与作者评分，不宣称准确率或CAI改善。详情RESULTS_AND_NEXT_DECISION.md。用户目视假设只在OUT/USER_HYPOTHESES.md，不进入模型prompt。

## 代码与有限验证

scripts/vlm_grounding_pilot包含common、prepare、run、report、html_view、validate及有限test_contracts。模块来源绑定当前工作树；不修改src/生产cache/原论文/旧统计。纯CPU复用原_features、NativeCellGrid合法集合和vlm_first_action_mask；空已测集合、action_count=0、预算0.25，C0不是新动作。

Q1—Q6见REQUIREMENTS_REVIEW.md、cpu_preflight.json、first_case_wiring.json、finite_validation.json、html_open_check.json及Git交付记录。真实浏览器使用已存在的临时headless shell，查看首页/首件、验证全部本地链接与图像加载、候选隐藏及实际输入切换、桌面和390px移动布局。PNG中文显示正常；本机字体weight fallback为500不改变模型输入。

报告复核修正一项纯表示问题：历史Python tuple与新JSON list直接比较误报内容不同；回归测试先复现失败，再采用同一JSON表示比较。没有改raw、候选、模型参数或增加调用。Ruff仅检查新脚本，不运行全库测试。

已完成运行的prepare/run会拒绝覆盖lock或再次加载；后续只运行report.py、validate.py --final等CPU查看命令，不能删state重复抽样。没有未知人评时直接保持PENDING，不自动上线新prompt或替换旧缓存。

Git只stage本任务CODE/OUT/ART、任务卡和全局ledger新增一行；模型/字体/环境未提交。真实结果SHA、文件追踪及最终三方一致由GIT_DELIVERY.json和最终回复记录。
