# 论文证据 P0 有限检查

Review mode: SELF_CHECK_WITH_EXTERNAL_NUMERIC_ORACLE。按用户治理由主代理审查，未委派科学决策。

实际命令：
```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_cai_agent_paper_evidence.py
python docs/cai/paper_evidence/paper_evidence_reference.py --self-test
python -m ruff check src/cmc_bbdm/cai_agent_v3/paper_evidence.py src/cmc_bbdm/cai_agent_v3/paper_evidence_math.py scripts/build_cai_agent_paper_evidence.py tests/test_cai_agent_paper_evidence.py
```

结果：8 passed in 2.07s；包内独立9例PASS。初始预期失败为数值模块不存在；新增实现后通过。Ruff发现37处字典写法/导入格式问题，限新增四文件修正后0错误，没有更改数值预期。

A1–A5合成例覆盖原序索引、标签错配、repeat先损失、组成员权重、域等权/池化、阶梯及边界、full单点、双方最早成本/未达/反弹/零分母。真实trace字段不一致会拒绝；MD/TeX保留负值和NA。
CLI --help实际成功，仅analyze/export。源码没有模型构造、checkpoint加载、训练/旧summarize调用或GPU探测；只读已存最终预测和必要来源。真实数据检查将在一次analyze中执行，现不声称真实50全输入结果已复算。
