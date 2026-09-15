# 主代理定向复核：VLM_GROUNDING_PILOT_R1_e8d9ef01

主规范为spec/CODEX_VLM_GROUNDING_PILOT_EXECUTION.md；本表覆盖W0—W4和Q1—Q6，实际运行计数与交付SHA由同目录结果/交接记录闭合。

| 要求 | 权威证据与核对范围 |
|---|---|
| Q1 6件身份/禁止选择泄漏 | cases.csv、prepare.select_cases；三个固定VALID，另三个域TRAIN按指定完整specimen_key SHA256排序最小项；case/输入在生成前冻结。源路径来自encoder_execution_root，source hash匹配，全部六件R0/clean与历史hash相等。没有打开TEST或C-scan图。 |
| Q2 四配置真接线 | experiment_lock 24个独立完整签名，cpu_preflight实际fake backend收到的prompt及PNG hash；真实runs/input_N.json、prompt_N.txt、chat_N.txt；first_case_wiring验证先q24四配置。P1固定包文本＋原schema，未调用resolve/旧主入口，未增加system/vision-ID。 |
| Q3 标签几何 | 384个label_boxes，font路径/hash/size/inset，label_change_mask外像素相等；奇数773×899合成64彩色ID图验证0..63及中心归属，64字框均在cell内；实际processor尺寸980的q24 R1人工观察可读。R1没有二次旋转。 |
| Q4 生成与额度 | gpu_session、attempts.jsonl、各raw/token-ID/attempt状态，首答和修复完整保留；原parser与局部contract-valid分开。测试覆盖valid no-cue不修复、无效最多一次修复、失败/STARTED/完成不重发、签名变化拒绝复用。真实forward pre-hook计数，完整generate不冒充单次forward。单卡累计1800秒、单次120秒、batch1、CPU≤4，无其他研究模型。 |
| Q5 图/数表/人工边界 | 每件clean/R0/R1真实PNG、四新配置及H00独立叠图/序数图/64格表/C0 JSON；24 summary、30配对变化、6 H00-A比较；HTML支持隐藏候选/真实输入/原尺寸/raw下载。首次解析与repair-dependent分列，H00明确历史最终raw。PENDING人工CSV和HOW_TO_REVIEW，无自动GT/IoU/赢家。 |
| Q6 冻结与交付 | protected_before+finite_validation关键源码/cache哈希，最终git diff仅新CODE/OUT/ART、任务卡及ledger一行；旧ledger前缀保持。Git实际提交/上传验证见GIT_DELIVERY与最终交接，不提交模型/字体/环境。 |

主代理保留选样/几何/状态机/模型接线/C0/统计语义与最终复核。唯一机械worker（gpt-5.6-luna max）按固定payload合同完成html_view.py，主代理全文件审阅并纠正H00原始回答标签，增加no-cue与首次内容分栏；科学处理均主代理实现。未因首件输出改变P1/R1或选择。

本轮未决项只有真实作者定位评价及以后是否值得开展另行授权验证。代码检查/格式通过/候选变化均不能代替定位改善。新confidence含位置把握，C0大小变化不等于CAI收益；没有替换旧先验或论文数字。
