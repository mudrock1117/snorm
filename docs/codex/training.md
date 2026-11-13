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
- optimizer 严格分为两个参数组：新增 output 通道所在四个 Parameter 使用 `5e-5`，shared conv 两个 Parameter 使用 `5e-6`；A4 的 cosine 调度对两组应用相同缩放因子，使 0.1× 比例全程保持。
- backbone_0/1/2/3、proj_norm、box_head_norm、rpn_eval 均冻结；A4 与 A3 一样跳过 epoch 边界 SNorm 状态更新。
- 独立配置使用 clear-source-only 训练、`(32,64,128,256,512)` anchors 与独立目录 `s0_dbias_color_A4_protected_rpn/E2`。
- `tools/check_a4_protected_rpn.py` 走真实 RPN training forward/loss contract，并检查 optimizer、旧/新通道、shared conv 与所有冻结模块的参数变化。

验证状态：A4 保护器 CPU smoke 已通过；真实 checkpoint expansion、RPN loss 与 GPU audit 尚待服务器 GPU1 验证，未启动正式训练。


## B1 / Protected Small-Anchor + Selective-AHDR

- B1 被硬性固定从 `/home/ubuntu/hdx/snorm_E2_repro/debug/city_2_foggy/s0_dbias_color_E2_zdh/E2/best.pth` 初始化；禁止接续 A3/A4/E2-cont。
- 真实旧推理在每一步执行 `output = 0.6 * raw_render + 0.4 * save_im0`；`save_im0` 是全局 `user_norm_v2` 后经 `norm_back` 的稳定基准，最终 step6 的 raw AHDR 输出是恢复端点。
- 新 gate 输入为 `[F0, Fr-F0]`，结构为 `Conv3x3(6→8)+ReLU+Conv1x1(8→1)+sigmoid`，共 449 参数。
- 输出为 `F0 + alpha_max*M*(Fr-F0)`。默认 `alpha_max=1.2`、最终卷积零初始化、`M=0.5`，因此初始有效系数严格为 0.6，与旧路径数学等价。
- B1 独立 `mode_training=13`：clear source 生成 stop-gradient clean reference，项目已有 synthetic degraded source 进入六步 AHDR、gate 和真实 RPN/ROI detection loss。
- 辅助损失仅使用 source：SmoothL1 consistency、source-GT object-mask consistency、相对初始 gate 0.5 的轻量正则；默认权重分别为 0.01、0.01、0.001。
- optimizer 只包含 A3 四个 RPN output Parameter（仅新通道有效，LR 5e-5）与 gate 四个 Parameter（LR 1e-5）；shared RPN conv、render_net、SNorm、backbone、ROI 与 rpn_eval 冻结。
- B1 继承 A3 epoch 边界保护：跳过 frozen proj_norm 的 clip/reset/`compute_norm_parameters()`，只执行 gate 与新 anchor 的 optimizer/scheduler 更新。
- `SELECTIVE_AHDR.ENABLED=False` 时不实例化 gate，旧配置、checkpoint state_dict 和 forward 行为不变。

验证状态：Python 语法检查和聚焦 CPU smoke 已通过；真实数据、checkpoint、GPU audit 与训练尚未执行。


## B1 三卡 DDP（2026-11-12）

- `SOLVER.BATCH_SIZE` 直接传给每个 rank 的 BatchSampler，因此 DDP 配置值 4 表示 per-GPU 4、三卡 global batch 12。
- 19318 张训练集经三卡 DistributedSampler 后每 rank 6440 个索引，batch 4 时每 epoch 1610 个 optimizer global steps。
- B1 在 DDP 包装前固定 requires_grad whitelist；backward 返回后梯度已经 all-reduce，再执行 old-anchor mask → snapshot → optimizer.step → restore/state clear。
- mode13 必须包含 clear-source GT 的四项 detection loss：rpn/rcnn cls/reg；Fog/Night 不参与训练。
- LR 不随 global batch 放大：new-anchor 5e-5，Gate 1e-5。


### B1 DDP batch 配置修正（2026-11-12）

- 首版 ddp3 YAML 把换行误写为字面量 `\\n`，导致 `BATCH_SIZE: 4` 位于注释内并回退到默认值 1；真实首次 DDP audit 因此执行 batch1。
- YAML 已修复为真实独立配置行。正式训练入口同时断言 cfg batch、BatchSampler batch 一致且 DDP per-rank batch 必须为4。
- protected-update audit 默认显式使用轻量 execution batch1，并与 formal per-rank batch4 分开报告；`--config-only` 使用真实 batch4 DataLoader 验证 global12/1610 steps。


## B2 / Object-Aware Selective-AHDR（2026-11-13）

- B2 基于 B1 固定 E2 checkpoint、六步 AHDR、449 参数 Gate、`alpha_max=1.2` 和初始系数 0.6，不改变检测/AHDR forward 定义。
- 独立 `mode_training=14` 保留 clear-source GT 的 RPN/RCNN cls/reg 四项检测损失；B1 consistency/object-consistency 不进入 B2 total loss。
- source GT 按项目 `img_shape=(W,H)` 裁剪，并按 padded batch canvas `(H,W)` 投影到 Gate 网格；box 扩张 1.15，tiny box 至少覆盖一个 cell。
- 新增 object floor、small-object floor、background、TV 和 global-anchor 五项 Gate loss，权重依次为 0.05/0.10/0.02/0.001/0.01。
- optimizer 与 B1 完全相同：四个 RPN output Parameter（仅 32-anchor 通道可更新，LR 5e-5）和四个 Gate Parameter（LR 1e-5）；其余 detector、AHDR/SNorm、ROI、shared RPN conv 与 eval heads 冻结。
- 三卡正式配置为每 rank batch4、global batch12；Fog/Night 仅在既有 VOC evaluation 节点使用。

验证状态：代码已静态审计并提供单卡/DDP audit；真实 checkpoint、数据、GPU audit、VOC evaluation 和正式训练均待服务器验证，未宣称性能提升。
