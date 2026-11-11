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


## A4 / Protected-RPN Adaptation

- A4 保留 A3 的 32-anchor output 通道保护：仅 cls `[0,5,10]` 与 bbox `[0:4,20:24,40:44]` 可更新，旧 output 通道在 step 后恢复并清理对应 Adam state。
- 在此基础上，仅额外开放实际存在的 `teacher.rpn_norm.conv.{weight,bias}`；shared conv 不做通道 mask。
- optimizer 严格分为两个参数组：新增 output 通道所在四个 Parameter 使用 `5e-5`，shared conv 两个 Parameter 使用 `5e-6`。
- backbone_0/1/2/3、proj_norm、box_head_norm、rpn_eval 均冻结；A4 与 A3 一样跳过 epoch 边界 SNorm 状态更新。
- 独立配置使用 clear-source-only 训练、`(32,64,128,256,512)` anchors 与独立目录 `s0_dbias_color_A4_protected_rpn/E2`。
- `tools/check_a4_protected_rpn.py` 走真实 RPN training forward/loss contract，并检查 optimizer、旧/新通道、shared conv 与所有冻结模块的参数变化。

验证状态：A4 保护器 CPU smoke 已通过；真实 checkpoint expansion、RPN loss 与 GPU audit 尚待服务器 GPU1 验证，未启动正式训练。
