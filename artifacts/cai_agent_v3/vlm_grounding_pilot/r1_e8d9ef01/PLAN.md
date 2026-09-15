# VLM_GROUNDING_PILOT_R1_e8d9ef01 执行计划

基点 e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750；原分支既有隔离工作树干净。用户授权按spec直接执行，无额外确认。

1. W0：common.py绑定实际源码；prepare.py只读白名单元数据，固定3 VALID与3域hash最小TRAIN，生成同clean的R0/R1、字框/mask、原prompt和固定P1、历史H00；冻结24项完整签名。
2. W1：run.py独立状态机，原parser加跨region重复合同，最多一次原格式修复；先写stub接线/签名隔离/修复/resume与奇数尺寸几何测试，真实模型调用为0。
3. W2：CPU测试通过后GPU0单次加载原backend；显式传prompt并记录token IDs、输入、真实forward计数。单项120秒，累计1800秒，先q24-48四项接线检查后剩余20项；失败保留，不改配置。
4. W3：report.py仅CPU复用_features与vlm_first_action_mask，登记尺寸/空测量/预算.25形成legal；导出独立PNG、64格CSV、24行summary、30行配对变化、H00-A比较、静态HTML与PENDING人评模板。
5. W4：Q1—Q6一次有限核验，Ruff限新脚本、真实浏览器首页及一个案例；冻结文件hash/ledger前缀检查。追加真实账目，交接/结果commit push；验证三方SHA与实际文件追踪。

生产src、旧DIAG/new_protocol/W2/W3/论文只读；无训练/其他研究模型/新attention/TEST。完整回答生成次数不等同forward次数。作者未评不阻塞交付且不推断定位改善。
