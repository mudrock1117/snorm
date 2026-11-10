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


## 2026-11-06 固定样本加载优化

- 检测诊断不再遍历完整 test loader；改为扫描 dataset 索引元数据后构建单样本 Subset。
- 保留原 dataset __getitem__、测试 transforms、img_meta/target/scale_factor 和 8 元组 collate。
- Fog/Night 按 ID 内容或显式域前缀路由；Fog ID 不再扫描 Night。
- 增加 dataset 初始化、匹配、单图加载、step 推理和保存日志；失败时输出 ID 示例。
- Path(file_name).stem 同时用于索引匹配和变换后 img_meta 二次校验。
- 确认历史 .gitignore 的 datasets/ 误伤 detection/data/datasets；已改为 /datasets/，源码包仍待服务器来源核验后单独归档。
- 未执行数据加载、推理或测试，待服务器 GPU1 单样本验证。


## 2026-11-08 检测器内部与 GT 诊断

- 新增单样本、同类别、一对一 IoU 匹配，输出逐 Step TP/FP/FN、逐类别统计和逐 GT score/IoU 变化。
- 明确单样本 TP/FP/FN 不等于 VOC mAP。
- 新增离线 sweep JSON + ground_truth.json 重分析入口。
- 确认 RPN forward 返回 4 项、BoxHead forward 返回 9 项；可通过临时 hook 采集 proposals、objectness logits、ROI class logits 与特征统计。
- 内部 hook 默认关闭，每 Step 独立保存并立即移除；原始 tensor 保存另设开关。
- 禁止按数组索引跨 Step 比较 proposals/ROI rows，需使用 GT/class/IoU 或空间匹配。
- 未运行服务器 GPU 测试，待固定 Fog 样本验证。


## 2026-11-10 A3 Protected Small-Anchor

- 新增独立 A3 配置，保持 clear-source-only、32/64/128/256/512 anchors 和原 Fog/Night evaluation。
- 复用已审计的 E2→A checkpoint expansion，不改变 anchor ordering、RPN forward、proposal、matcher、NMS 或 evaluator。
- A3 optimizer 白名单仅含 rpn_norm 的 cls_logits/bbox_pred 四个 Parameter；backbone、proj_norm、box head、RPN shared conv 和 rpn_eval 全部冻结。
- 新增 32-anchor 通道以外的梯度在 optimizer step 前清零；step 后恢复旧通道并清零对应 Adam state，防止 weight decay/momentum 漂移。
- A3 跳过 epoch 边界的 SNorm 状态更新，避免冻结的 proj_norm 被 compute_norm_parameters 改写。
- 新增 CPU/GPU audit 工具，检查 checkpoint mapping、白名单、梯度 mask、旧通道/冻结模块零变化以及新增通道发生更新。
- 未启动正式训练；audit 与 GPU 验证待服务器使用 GPU1 执行。


## 2026-11-10 A3 audit RPN API 修复

- 服务器发现 audit 直接调用 `RPN.losses()` 后误按 dict 读取；真实 `RPN.losses()` 返回 `(objectness_loss, box_loss)` tuple。
- 已改为调用真实 `rpn_norm.forward(None, feature, img_metas, targets)`，按正式 ModuleSet mode=5 相同契约解析 `(proposals, loss_dict, logits, bbox_pd)`。
- 新增返回契约测试，明确拒绝把直接 losses tuple 当作 forward loss dict。
- 本地 CPU 纯逻辑 smoke 已通过；真实服务器 CPU/GPU RPN forward audit 仍待用户执行。
