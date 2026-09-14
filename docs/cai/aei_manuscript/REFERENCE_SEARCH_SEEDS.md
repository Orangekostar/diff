# 与本稿直接相关的检索种子和引用边界

不是可直接无脑复制的最终bibliography。Codex需要将实际读到的范围、元数据和支持句登记，验证后才写入references.bib。不要运行新的外部算法，也不要求覆盖固定篇数。

| ID | 论文/来源 | 已核实内容和用途 | 尚需核实/不能推断 |
|---|---|---|---|
| L01 | Learning to Maximize Mutual Information for Dynamic Feature Selection；PMLR202，2023 | 官方会议页面题目/作者/摘要/BibTeX；用于任务驱动动态信息获取的概念关系 | 回归条件与定理如要具体使用须读论文对应段；不能把GDFS保证转到本文RL绝对误差目标 |
| L02 | Autonomous ultrasonic inspection using Bayesian optimisation and robust outlier analysis；MSSP145，106897，2020 | 作者机构库摘要/元数据；顺序选择损伤指示较高位置，提供已存在主动超声的相关工作 | 本次未读取其全部方法/实验；硬件/成本精确细节需原文，不能自称其CAI MAE对照 |
| L03 | Deep Learning for Predicting Impact Energy and Compression After Impact Strength of Composite Materials Using C-Scan Images；AEI72，104518，2026 | 作者机构库摘要说明C-scan图像→CAI；DOI/venue/题目可核实，最直接期刊邻近工作 | 机构库作者列表/排序可能不是出版社完整作者顺序，最终BibTeX须publisher/Crossref确认；不拼其R²与本项目不同划分数字 |
| L04 | Hasebe相关2022与2025 Data in Brief资料 | 旧来源追溯给出PMC入口，可继续找作者数据说明与数据集版本 | 本次PMC页面未取得完整正文，title/DOI/域映射不能凭记忆填写；按实际使用数据来源核实 |
| L05 | 实际使用的ResNet、Transformer、策略梯度与Qwen技术报告 | 用于Methods中成熟组成部分的来源，来自当前实现 | 只引用真正使用的技术，不为热词填引文，不把库实现称新算法 |

## 精确入口

L01：
https://proceedings.mlr.press/v202/covert23a.html

该官方页面提供的作者为Ian Connick Covert、Wei Qiu、Mingyu Lu、Na Yoon Kim、Nathan J White、Su-In Lee；页码6424–6447。取网页现有BibTeX比根据二手引文手抄更可靠。

L02：
https://eprints.whiterose.ac.uk/id/eprint/160984/
https://strathprints.strath.ac.uk/72351/
https://doi.org/10.1016/j.ymssp.2020.106897

元数据作者：R. Fuentes、P. Gardner、C. Mineo、T.J. Rogers、S.G. Pierce、K. Worden、N. Dervilis、E.J. Cross。实际格式按最后核实的原文。

L03：
https://ideaexchange.uakron.edu/university_research/55/
https://doi.org/10.1016/j.aei.2026.104518
https://www.sciencedirect.com/science/article/pii/S1474034626002107

本次ScienceDirect正文访问受限，机构库可读摘要。不得从摘要图像/重要性描述直接推断本文agent的物理机制。这个论文可以帮助建立“已有C-scan图像的回归”到“获取图像信息的决策”的研究关系，但不要虚构它没有研究过的模块细节。

L04：
https://pmc.ncbi.nlm.nih.gov/articles/PMC9294053/
https://pmc.ncbi.nlm.nih.gov/articles/PMC11999467/

以上为已有追溯入口，不是本次完整读过的正文。先与当前DATA manifest中的作者数据URL/版本核对；元数据缺失只局部标记。

## 有限检索与引用验收

1. 先查repo中已有.bib/来源说明。针对最近邻论文每篇至多两种正常公开获取路线；不要绕过权限/验证码、不购买访问、不不停重试403。
2. 方法比较必须有正文或作者明确摘要依据；仅标题不足以断言对方模型/数据分割。记录read_extent=metadata/abstract/sections/fulltext。
3. 模型算法与数据集分开引用；引用别人的论文不等于声明复现了它。
4. 不设AEI同刊引文比例、引用数量配额或“今年必满10篇”。以本文真正的论证需求决定；避免将相邻但不相关工作纳入。
5. AEI文章作为叙事参考时不复制其句式/摘要，不仿造图。可记录一两项组织经验，但六章结构已由用户确定。
6. 所有引用导出真实bib条目；未核实的占位留在AUTHOR_INPUTS而不是造一个看似真实DOI。全文引用均能对应bib，正文中不使用ChatGPT的turn/filecite标记。
