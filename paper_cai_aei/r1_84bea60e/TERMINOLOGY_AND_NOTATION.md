# 术语与符号

| Canonical term | Symbol / meaning | 禁止混用 |
|---|---|---|
| compression-after-impact strength | y, MPa | accuracy、mask成功率 |
| surface image / C-scan image | S / X | 原始声学阵列（本稿未使用该输入） |
| acquired cell set | Omega_t, 8×8共64格 | 声学采样点 |
| acquisition fraction | c_t, 唯一原生像素/全图像素 | 时间/费用/步数比例 |
| acquisition cap | B=0.25 | 每件恰好采25% |
| observed mask | M_t | 损伤真值mask |
| CAI predictor | frozen MEAN_SC@1750, P_all | spatial Actor、critic |
| policy / actor | VLM_SPATIAL_FEEDBACK | VLM逐步语言规划、BC |
| surface VLM prior | ordinal region cue / C0 | 物理概率、损伤GT |
| capture group | 来源关联组 | 独立重复 |
| normalized error area | A(B), MPa, lower better | pooled endpoint MAE |
| early error area | A(0.0625) | 全程A |
| trajectory objective | J=A(0.25)+0.25 e_T | 推断时真实标签输入 |
| timing-weighted contribution | g_t=(1-c_t/B)(e_(t-1)-e_t) | 因果百分比、箱内积分 |
| matched empirical quality | first group-MAE passage on stated grid | 部署STOP、单件oracle停止 |
| selected validation set | 50 physical / 48 groups / six domains | independent test set |

运行数650与事件10227不改变物理N。Random先每episode损失→试样内repeat均值；A再域内均值→六域等权，MAE池化物理试样。全文方法显示名与原ID一一对应，主方法身份不因对照结果改变。
