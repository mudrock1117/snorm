# 训练流程

## A / Small-Anchor 与 A3 / Protected Small-Anchor

- 基线 E2 RPN anchors：`(64, 128, 256, 512)`；A/A3：`(32, 64, 128, 256, 512)`。
- A/A3 初始化统一使用 `diagnostics/small_anchor.py` 的 audited expansion。AnchorGenerator 已确认采用 ratio-major、scale-minor 通道顺序。
- 旧 64/128/256/512 通道从 E2 精确复制；新增 cls 通道为 `[0,5,10]`，新增 bbox 通道块为 `[0:4,20:24,40:44]`。
- A3 使用 `SNORM_EXPERIMENT=A3`，训练数据仍仅为 `bdd_daytime_clear_train`；Fog/Night 只进入原 evaluation。
- A3 冻结全部 detector 参数，仅将 `teacher.rpn_norm.cls_logits.{weight,bias}` 与 `bbox_pred.{weight,bias}` 加入 optimizer。
- 每次 backward 后、optimizer step 前调用 `ProtectedSmallAnchorStep.before_step()`：旧 anchor 梯度清零并快照旧通道。
- optimizer step 后调用 `after_step()`：恢复旧通道，并清零这些通道对应的 Adam tensor state，避免 weight decay/momentum 造成漂移。
- A3 epoch 结束时跳过 SNorm 参数裁剪、reset 和 `compute_norm_parameters()`，保持冻结的 proj_norm 状态。
- 原 VOC evaluator、每 1000 step 评估和 best checkpoint 规则未修改。

验证状态：代码与通道映射已静态审计；`tools/check_a3_protected_small_anchor.py` 的真实 CPU/GPU audit 尚待服务器执行。
