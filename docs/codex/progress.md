# 项目进展

## 2026-11-05

- 已核验 `snorm_optimization_v2` 与 `backup/snorm_baseline_20261104` 均指向 `a98e4fb069d382740ff0a81734f432f5285dcce2`。
- 已从该 SHA 创建 `feature/ahdr_structural_v3`；仅开发分支发生后续提交。
- 已创建 `BASELINE_MANIFEST.md`，明确 GitHub 冻结不包含服务器未提交代码、checkpoint、数据集和训练状态。
- 已增加默认关闭的 `diagnostics/` 统计内核与六步记录器，不改模型执行路径。
- CPU 隔离测试结果：3 passed；测试了输入不变性、禁用无副作用和六步记录。
- 待办：接入固定 Fog/Night 样本；记录 epoch 边界状态；增加预测一致性测试；诊断完成后再实现 AHDR 结构方案。
- 未启动训练，未产生或宣称精度提升。


## 2026-11-06

- 已将现有 recorder 接入 SNorm 真实 eval 六步 AHDR 路径；默认关闭且不改变 legacy 输出公式。
- 已区分相邻步骤 `step_delta` 与相对 reference 的 `reference_delta`，输出 JSON、CSV、PNG 和可选 PT。
- 已按 run/sample/call 隔离输出，并兼容原 `SNORM_DEBUG_SAVE`。
- 已按 resize 后 GT boxes 构造 all/small/regular mask；small 默认面积阈值为 32×32。
- 已增加独立 SNORM_STEPS=0～6 检测 sweep 入口；不会进入或修改 VOC evaluation。
- RPN proposals/ROI logits 尚未由公开 eval 返回暴露，当前明确记录 unavailable。
- 已增加可逆 epoch 边界探针；确认 `memory_index=0` 时当前实现可能用零 memory 覆盖 alpha，未修改原更新逻辑。
- 已记录 Guided-AHDR v1 单一机制设计，等待 Fog/Night 真实诊断后再实现。
- 本轮代码未在本地或 GPU 执行，全部测试与推理待用户服务器验证。
