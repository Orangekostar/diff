# AEI格式核实记录

2026-09-14，官方Guide for Authors通过web.open定向访问返回403：
https://www.sciencedirect.com/journal/advanced-engineering-informatics/publish/guide-for-authors

状态：JOURNAL_FORMAT_VERIFICATION_PENDING。文章类别、摘要精确限额、引用样式、highlights/graphical abstract是否必需、匿名方式、模板和期刊专属声明要求均UNKNOWN；不得以本文件的计划替代官方规则。

本任务配置：英文六主章，正文目标6000–7500词，摘要目标200–230词；均为作者写作目标。采用现有LaTeX工具链生成作者审阅稿；不声称submission-ready。4条highlights、cover letter、可用性/AI声明作为草稿交付，作者身份和声明保留待确认。


已核实的出版社层面AI政策：2026-09-14实际读取 https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals 。实质写作辅助须披露、作者须审核并负责；研究AI用途记录在Methods；解释性示意图的AI辅助须说明；不能使用生成式AI制作投稿图形摘要。这里不把出版社通用政策等同于已核实的AEI具体格式。

可用模板检查：现有工具链未发现elsarticle.cls，使用已安装article + LuaLaTeX生成审阅稿。采用编号引用作为草稿选择，不宣称已核实为AEI必需格式。未安装字体或大规模系统包；Pandoc仅安装到临时工具目录。
