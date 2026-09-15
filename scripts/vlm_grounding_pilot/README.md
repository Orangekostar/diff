# 有限VLM定位对照

工作树根执行。生产源码/历史缓存只读；先选样冻结，再CPU预检，再唯一GPU session，最后从已保存回答生成报告。

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python scripts/vlm_grounding_pilot/test_contracts.py
python scripts/vlm_grounding_pilot/prepare.py
python scripts/vlm_grounding_pilot/validate.py
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=5s 1800s python -u scripts/vlm_grounding_pilot/run.py
python scripts/vlm_grounding_pilot/report.py
python scripts/vlm_grounding_pilot/validate.py --final
python -m ruff check scripts/vlm_grounding_pilot
```

已有experiment_lock时prepare拒绝覆盖，已有gpu_session时run拒绝再次加载或重置额度。本轮真实运行完成后只需最后三个CPU命令；不要删除lock/session来获取新调用。execute_job对已完成/失败不重发；STARTED无完整结果归INTERRUPTED，保留已开始尝试。格式不合法最多一次原修复；跨region重复另记contract-invalid。不为低置信、no-cue或语义疑问重试。

实际生成使用原QwenVLBackend.load及其等价双图chat/processor/generate调用。额外仅记录输入、输出ID和forward pre-hook计数，不改变注意力或请求额外分数。单项120秒信号定时器+外部总1800秒看门狗，异常后不再次加载。报告仅调用_features、NativeCellGrid.legal_mask和vlm_first_action_mask的CPU函数，无Actor预测。新置信图是解码后序数值，不是注意力/损伤概率。

R1字体为本机DejaVu Sans，尺寸/路径/hash在各输入render_metadata.json中；PNG报告中文采用本机WenQuanYi Zen Hei。两者都不随仓库分发。输出根results/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01，原始图片≤1024，完整source仅存路径/hash。HTML支持file://，无网络依赖。
