# 项目进展

## 2026-11-05

- 已核验 `snorm_optimization_v2` 与 `backup/snorm_baseline_20261104` 均指向 `a98e4fb069d382740ff0a81734f432f5285dcce2`。
- 已从该 SHA 创建 `feature/ahdr_structural_v3`；仅开发分支发生后续提交。
- 已创建 `BASELINE_MANIFEST.md`，明确 GitHub 冻结不包含服务器未提交代码、checkpoint、数据集和训练状态。
- 已增加默认关闭的 `diagnostics/` 统计内核与六步记录器，不改模型执行路径。
- CPU 隔离测试结果：3 passed；测试了输入不变性、禁用无副作用和六步记录。
- 待办：接入固定 Fog/Night 样本；记录 epoch 边界状态；增加预测一致性测试；诊断完成后再实现 AHDR 结构方案。
- 未启动训练，未产生或宣称精度提升。
