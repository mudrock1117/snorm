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


## 2026-11-06 服务器验证前兼容性加固

- 已按 FasterRCNN 实际 eval 返回强制 batch=1，并校验 detection list 与 boxes/scores/labels。
- 已核对测试数据为 Fog/Night 两个独立 loader、8 元组 batch、resize 后 GT 坐标。
- 已支持多种历史 checkpoint 容器和 module 前缀；关键检测参数缺失默认停止。
- 已移除 Python 3.7-only future annotations，并替换旧 PyTorch 不兼容 API。
- step sweep 现在恢复逐模块 train/eval 标志、环境变量及 Python/NumPy/PyTorch/CUDA RNG。
- epoch probe 明确使用 teacher.proj_norm，恢复 memory、alpha、dtype/device、RNG 和诊断内部状态。
- 未执行 pytest、CUDA、推理或训练；待服务器首先进行 GPU1 单样本 0/1/6 step 验证。
