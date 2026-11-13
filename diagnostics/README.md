# SNorm/AHDR diagnostics

该目录提供默认关闭、显式调用、只读的诊断基础设施。当前版本不修改 SNorm/AHDR 前向路径，也不自动注册 hook。

## 已覆盖指标

- AHDR step 1～6 的 reference、guide、output、residual；
- mean、std、L2、最大绝对值、相对高频能量；
- 相邻/参考特征余弦相似度、变化量和变化率；
- 给定 GT mask 后目标区域与背景区域响应；
- 模块梯度范数、CUDA 显存和显式计时；
- epoch 边界前后 SNorm 参数或 memory 张量快照；
- 可选保存每步原始 PT 张量和统一 JSON。

## 安全约束

- `enabled=False` 时不创建目录、不写文件、不同步 CUDA；
- 所有张量统计均在 `torch.no_grad()` 下对 detached tensor 计算；
- 不切换模型的 train/eval 状态，不改变随机种子；
- 原始张量只在 `save_tensors=True` 时保存；
- 诊断输出必须使用独立实验目录，不能覆盖 checkpoint 或历史日志。

## 建议接入点

1. 在 SNorm eval 六步 AHDR 调用处，每步结束后调用 `record_ahdr_step`。
2. 用检测 GT 生成二值 mask，传入记录器比较目标/背景响应。
3. 在 `compute_norm_parameters()` 前后调用 `record_state`，使用同一固定样本再次推理。
4. backward 后、optimizer step 前调用 `record_gradients`；step 前后另存参数快照可计算更新量。
5. Fog/Night 固定样本分别使用固定输出目录和随机种子。

## 已确认的接入依据

- E1/E2 训练入口的 `mode_training=0` 走 SNorm 训练 `moe_mode=0`，是单次全局归一化路径。
- SNorm 的训练渲染分支 `moe_mode=7` 是单步三输入 AHDR。
- 当前 AHDR 只编码 `x0` 和 `x1`；`x2` 仅决定返回分支，没有进入 encoder。
- `scale_interact` 已实例化，但 AHDR forward 直接将 `fused_feats` 送入 decoder。
- 融合头生成三个通道，forward 只使用并重新归一化前两个通道。
- 因此应先用本诊断确认六步变化是否递减、夜间高频/目标响应是否持续损失，再修改结构。

## 测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_diagnostics.py
```

当前 CPU 隔离测试覆盖：输入不变性、禁用模式无副作用、六步记录与区域指标。


## 真实六步接入

`SNorm_0322_valid_refine.py` 的 eval full 路径已接入原有 recorder。默认关闭：

```bash
export SNORM_DIAG=1
export SNORM_DIAG_DIR=/path/to/new/diagnostics
export SNORM_DIAG_RUN_ID=fog_night_fixed_v1
export SNORM_DIAG_SAMPLE_IDS=fog_001,night_001
export SNORM_DIAG_SAVE_TENSORS=0
export SNORM_DIAG_SAVE_PNG=1
```

每个样本按 `run_id/sample_id/call_id` 隔离。JSON/CSV 包含 step delta 与 reference delta；PNG 明确按 raw input `[-1,1]` 或渲染张量 `[0,1]` 转换。旧 `SNORM_DEBUG_SAVE` 不受影响。

## 固定样本检测诊断

样本文件每行写一个 file name、file stem 或 dataset id：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/run_snorm_detection_diagnostics.py \
  --config-file configs/dbias/city_2_foggy/domain_det_s0_moe_bn_color.yaml \
  --checkpoint /external/checkpoints/E2.pth \
  --sample-ids diagnostics/fixed_samples.txt \
  --output-dir /new/experiment/diagnostics \
  --steps 0,1,2,3,4,5,6
```

该入口独立于 VOC evaluation，并为每个 step 额外调用一次检测器。正常评估次数和协议不变。RPN proposals 与 ROI logits 尚未由公共 eval 返回暴露，JSON 中明确标为 unavailable。

## epoch 边界探针

```bash
CUDA_VISIBLE_DEVICES=1 python tools/probe_snorm_epoch_boundary.py \
  --config-file configs/dbias/city_2_foggy/domain_det_s0_moe_bn_color.yaml \
  --checkpoint /external/checkpoints/E2.pth \
  --sample-ids diagnostics/fixed_samples.txt \
  --output-dir /new/experiment/epoch_probe
```

探针调用 `compute_norm_parameters()` 前后使用同一固定输入，并在退出前恢复 memory、alpha 参数。当前代码已确认 `memory_index=0` 会使用零 memory 覆盖 alpha；本工具只报告该现象，不改变训练实现。


## 服务器兼容性与首次运行顺序

当前配置 `domain_det_s0_moe_bn_color.yaml` 的 `DATASETS.TESTS` 为：

1. `bdd_daytime_fog_test`
2. `bdd_night_sun_test`

`build_data_loaders(..., is_train=False)` 为每个数据集返回独立 loader，测试 batch 固定为 1；每个 batch 是仓库 collate_fn 产生的 8 元组。诊断入口会验证 batch、`img_metas[0]['img_info']`、`scale_factor`、targets 和 resize 后 boxes，异常时立即停止。GT 和预测框均使用 resize 后检测输入坐标，并覆盖到同尺寸诊断 PNG。

### GPU1 单样本最小诊断

先只在文件中保留一个确实存在的 Fog 文件 stem：

```bash
printf '%s\n' 'REPLACE_WITH_ONE_FOG_FILE_STEM' > diagnostics/fixed_samples.txt
```

然后运行：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/run_snorm_detection_diagnostics.py \
  --config-file configs/dbias/city_2_foggy/domain_det_s0_moe_bn_color.yaml \
  --checkpoint /absolute/path/to/checkpoint.pth \
  --sample-ids diagnostics/fixed_samples.txt \
  --output-dir /absolute/path/to/new_diag_run \
  --steps 0,1,6
```

确认单样本成功后，再把 Fog/Night 固定样本加入列表并运行 `--steps 0,1,2,3,4,5,6`。首次运行不要加 `--save-tensors`，避免 PT 输出占用大量磁盘。

预期路径：

```text
/new_diag_run/<dataset>/<sample>/
  <sample>_step_sweep.json
  overlays/original_gt.png
  overlays/step00_gt_pred.png
  overlays/step01_gt_pred.png
  overlays/step06_gt_pred.png
  ahdr/steps_<sample>/<sample>_steps_<N>/call_000000/
    metrics.json
    steps.csv
    step00_raw_input.png
    step00_save_im0.png
    stepNN_*.png
```

### GPU1 epoch 边界最小探针

```bash
CUDA_VISIBLE_DEVICES=1 python tools/probe_snorm_epoch_boundary.py \
  --config-file configs/dbias/city_2_foggy/domain_det_s0_moe_bn_color.yaml \
  --checkpoint /absolute/path/to/checkpoint.pth \
  --sample-ids diagnostics/fixed_samples.txt \
  --output-dir /absolute/path/to/new_epoch_probe
```

入口明确定位训练脚本使用的 `model.teacher.proj_norm`，并要求其类型为 SNorm。探针默认恢复 memory、alpha、dtype/device 及 Python/NumPy/PyTorch/CUDA RNG 状态。

### checkpoint 兼容

支持：

- checkpoint 根对象就是 state-dict；
- `model`；
- `state_dict`；
- `model_state_dict`；
- `net/network/weights`；
- 全量 `module.` 前缀。

入口总是打印 state 来源、missing keys 和 unexpected keys。backbone、RPN、ROI head 或 SNorm 关键参数缺失时默认停止。仅用于分析不完整权重时才可显式添加 `--allow-critical-missing`；此时结果不能作为正式对比。

### 常见故障

- `sample IDs not found`：使用数据集 XML/文件中的真实 file name、stem 或 id；先确认 Fog/Night 根目录与配置一致。
- `8-item batch`：服务器代码的数据集/collate_fn 与当前分支不一致，先核对 checkout commit。
- `batch size 1`：不要改诊断 loader 的测试 batch；FasterRCNN eval 返回是一项 detection list。
- `GT coordinates exceed resized tensor`：transforms 或 targets 坐标未同步，不应忽略后继续画框。
- `critical detector parameters are missing`：checkpoint 身份或封装格式不匹配；先检查输出的 state source 和 key 列表。
- PNG 缺失但 JSON 存在：检查 `SNORM_DIAG_SAVE_PNG`、Pillow 和输出目录写权限。
- CUDA OOM：先用一个样本和 `--steps 0,1,6`，不要启用 `--save-tensors`。
- 旧环境 API 报错：记录 Python、PyTorch、Pillow 版本和完整堆栈，不要升级原训练环境。


## 固定样本快速索引

诊断入口不再遍历完整 test loader。流程为：

1. 使用 `build_datasets` 初始化原 Fog/Night dataset 和原测试 transforms；
2. 只扫描 dataset 已有的 `ids/image_ids/image_set_index/file_names/data_infos` 等元数据；
3. 同时比较完整名称、`Path(name).name` 和 `Path(name).stem`；
4. 对命中索引构造 `torch.utils.data.Subset`；
5. 仅该 Subset 通过原 dataset `__getitem__`、原 transforms 和原 8 元组 `collate_fn` 加载。

ID 中含 `fog` 时不会扫描 Night dataset；含 `night` 时不会扫描 Fog dataset。也可显式写成 `fog:<id>`、`night:<id>` 或 `<dataset_name>:<id>`。

运行时日志依次显示 dataset 初始化、元数据匹配、单图加载、每个 AHDR step 开始/结束和文件保存。匹配失败会给出各已扫描域的可用 ID 示例。

## detection/data/datasets 源码包缺失

根因是历史 `.gitignore` 使用了无前导斜杠的 `datasets/`。该规则会递归匹配任何层级的同名目录，因此不仅忽略根数据集，也错误忽略 `detection/data/datasets/` Python 源码包。仓库没有 `.gitmodules`，所以它也不是可自动恢复的 Git submodule。

当前分支已把规则改为仅忽略根数据目录：

```gitignore
/datasets/
```

这只允许未来跟踪嵌套源码包，并不会凭空恢复缺失文件。安全恢复步骤：

1. 保留服务器当前软链接，先不要删除或覆盖；
2. 对 E2 工作树的 `detection/data/datasets` 执行只读文件清单、Git 状态和 SHA256 记录；
3. 确认其中只有 Python 源码/必要小型元数据，不含真实图像、标注副本、缓存或权重；
4. 与历史 E2 实际运行版本核对 imports 和文件内容；
5. 经确认后，从 E2 工作树复制到独立临时目录，审查 diff，再显式 `git add detection/data/datasets` 提交；
6. 提交后在新 checkout 验证模块可导入，再移除服务器临时软链接。

在完成来源核验前，不应把软链接目标或未经审查的整个目录直接上传 GitHub。


## 单样本 GT 匹配

诊断入口现在保存 resize 后检测坐标系的 `ground_truth.json`，并自动输出：

- `gt_step_analysis.json`：每步 TP/FP/FN、逐类别统计、预测匹配明细、未匹配 GT；
- `gt_step_changes.csv`：同一 GT 在不同 Step 的 detected/score/IoU；
- `gt_step_summary.txt`：便于快速阅读的变化摘要。

匹配规则：预测按 score 降序，仅与相同类别的尚未匹配 GT 计算 IoU；IoU 达到阈值后进行一对一匹配。默认 score 阈值为 0，IoU 阈值为 0.5，可通过下列参数修改：

```bash
--match-score-threshold 0.5 --match-iou-threshold 0.5
```

这些 TP/FP/FN 只描述一个 resize 后固定样本，不是完整数据集 VOC mAP，不能与历史 Fog/Night mAP 混用。

对已经保存了 `ground_truth.json` 的旧 sweep 可离线重算：

```bash
python tools/analyze_snorm_step_gt.py \
  --sweep-json /path/sample_step_sweep.json \
  --ground-truth-json /path/ground_truth.json \
  --output-dir /path/offline_gt_analysis \
  --iou-threshold 0.5 \
  --score-threshold 0.5
```

## 可选 RPN/ROI 内部采集

默认关闭。启用只读临时 hook：

```bash
--capture-internals
```

每个 Step 独立生成：

```text
internals/step00_internal.json
internals/step01_internal.json
internals/step06_internal.json
```

字段：

- `rpn.proposal_counts`：NMS 后 proposal 数；
- `rpn.proposals`：每张图 proposals 的 shape/dtype/min/max/mean/std；
- `rpn.objectness_logits`：RPN anchor objectness logits 统计；
- `rpn.decoded_bbox_map`：RPN 解码框图统计；
- `roi.class_logits`：进入 ROI 后处理前的类别 logits；
- `roi.box_regression`：ROI box regression 输出统计；
- `roi.roi_features/box_features`：ROI pooling 和 predictor 特征统计；
- `module/field`：字段所属真实模块和 forward 输出位置。

如确需原始 CPU tensor，额外指定：

```bash
--capture-internals --save-internal-tensors
```

这会生成 `stepNN_internal.pt`，可能占用较大内存和磁盘。proposals 数量和 ROI 行在每个 Step 都会因 RPN/NMS 独立变化，不允许按数组索引跨 Step 对齐。跨 Step 对比应使用 GT/class/IoU 或空间匹配。

hook 仅注册到 `teacher.rpn_norm` 与 `teacher.box_head_norm`，forward 结束立即移除，不修改返回值。若服务器实际模块输出结构不是 RPN 4 项或 BoxHead 9 项，JSON 会记录 `rpn_error/roi_error`，而不是伪造数据。


## Fog30 + Night30 批量诊断

固定样本不是连续取前 30 张。选择器按整个数据集索引范围分层，每层用固定种子抽取一个样本，保存实际 sample ID 清单与 manifest，确保重复运行使用同一集合：

```bash
python tools/select_snorm_diagnostic_samples.py \
  --config-file configs/dbias/city_2_foggy/domain_det_s0_moe_bn_color.yaml \
  --output-ids /home/ubuntu/hdx/diag_fog30_night30/sample_ids.txt \
  --output-manifest /home/ubuntu/hdx/diag_fog30_night30/sample_manifest.json \
  --count-per-domain 30 \
  --seed 20261108
```

然后在 GPU1 上运行 Step 0/1/6。必须启用内部采集以取得 RPN proposals，但默认不保存大型原始 tensor：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/run_snorm_detection_diagnostics.py \
  --config-file configs/dbias/city_2_foggy/domain_det_s0_moe_bn_color.yaml \
  --checkpoint /home/ubuntu/hdx/snorm_E2_repro/debug/city_2_foggy/s0_dbias_color_E2_zdh/E2/best.pth \
  --sample-ids /home/ubuntu/hdx/diag_fog30_night30/sample_ids.txt \
  --output-dir /home/ubuntu/hdx/diag_fog30_night30/reports \
  --steps 0,1,6 \
  --match-score-threshold 0.5 \
  --match-iou-threshold 0.5 \
  --rpn-fn-iou-threshold 0.5 \
  --capture-internals
```

RPN coverage 对每个 GT 独立计算其与当前 Step 全部 proposals 的最大 IoU，报告 Recall@0.3/0.5/0.7；最终 FN 按 `--rpn-fn-iou-threshold` 分成 `rpn_uncovered` 与 `rpn_covered_detection_failure`。不同 Step 的 proposal/ROI 数组不会按索引对齐。small/medium/large 使用 resize 后检测坐标中的面积阈值 32²/96²。

生成汇总：

```bash
python tools/summarize_snorm_batch.py \
  --report-root /home/ubuntu/hdx/diag_fog30_night30/reports \
  --output-dir /home/ubuntu/hdx/diag_fog30_night30/summary

cat /home/ubuntu/hdx/diag_fog30_night30/summary/batch_summary.txt
column -s, -t /home/ubuntu/hdx/diag_fog30_night30/summary/batch_summary.csv | less -S
```

输出包括 Fog、Night 与 overall 的 GT/TP/FP/FN、Precision/Recall、RPN Recall、尺度 Recall 和两类 FN 数量，并在 JSON 中保存实际 sample IDs。所有结果只是固定样本诊断汇总，**不是 VOC mAP**。抽样器用索引分层保证覆盖测试集范围，不能在推理前保证 GT 数量/尺度严格平衡；实际分布由汇总中的 GT 与尺度计数核验。


## 最终离线 root-cause 报告

已有 Fog30/Night30 的 `gt_step_analysis.json` 和 RPN coverage 时，不需要重新运行 detector：

```bash
python tools/analyze_snorm_root_cause.py \
  --report-root /home/ubuntu/hdx/diag_fog30_night30/reports \
  --output-dir /home/ubuntu/hdx/diag_fog30_night30/root_cause
```

该命令只读取现有 JSON，输出 `diagnostic_root_cause.txt/json/csv`。FP 使用互斥优先级：duplicate > classification > localization > background。Step0/Step6 FP 身份通过同类别且预测框 IoU>=0.5 的贪心空间匹配建立，不使用 proposal、ROI 或预测数组索引。运行时强制检查 `TP+FN=GT`、四类 FP 之和等于 FP、FN 的尺度/阶段之和等于 FN。结果是固定样本诊断，不是 VOC mAP。


## B1 三卡 DDP audit

三张物理 GPU 通过环境变量映射为进程内 logical GPU 0/1/2：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 SNORM_EXPERIMENT=B1 \
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29631 \
  tools/check_b1_ddp.py \
  --config-file configs/dbias/city_2_foggy/domain_det_s0_moe_bn_color_B1_selective_ahdr_ddp3.yaml \
  --checkpoint /home/ubuntu/hdx/snorm_E2_repro/debug/city_2_foggy/s0_dbias_color_E2_zdh/E2/best.pth
```

audit 每个 rank 执行一个真实 mode13 forward/backward/optimizer step，rank0 汇总 sampler、Gate/anchor gradient、protected delta 和冻结模块 delta。该工具不会启动正式训练。


### B1 DDP batch 检查

Protected-update audit 默认使用明确的轻量执行 batch 1；其输出分别标记 formal batch 与 audit execution batch。无需 backward 的正式 DataLoader 检查：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 SNORM_EXPERIMENT=B1 \
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29633 \
  tools/check_b1_ddp.py --config-only \
  --config-file configs/dbias/city_2_foggy/domain_det_s0_moe_bn_color_B1_selective_ahdr_ddp3.yaml \
  --checkpoint /home/ubuntu/hdx/snorm_E2_repro/debug/city_2_foggy/s0_dbias_color_E2_zdh/E2/best.pth
```

预期 formal per-rank/global batch 为 4/12，sampler samples/rank 为 6440，steps/epoch 为 1610。若需用 batch4 重跑完整 protected audit，增加 `--execution-batch-size 4`。


## B2 Object-Aware Selective-AHDR audit

B2 使用独立分支、配置和工作目录；训练仍仅使用 clear source GT，Fog/Night 只进入 VOC evaluation。先在 GPU1 执行单卡数学、mask、梯度与冻结审计：

```bash
CUDA_VISIBLE_DEVICES=1 SNORM_EXPERIMENT=B2 \
python tools/check_b2_object_aware.py \
  --config-file configs/dbias/city_2_foggy/domain_det_s0_moe_bn_color_B2_object_aware_ahdr_ddp3.yaml \
  --checkpoint /home/ubuntu/hdx/snorm_E2_repro/debug/city_2_foggy/s0_dbias_color_E2_zdh/E2/best.pth
```

再检查三卡正式 DataLoader 语义（per-rank 4、global 12、19318 张时 1610 steps/epoch）：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 SNORM_EXPERIMENT=B2 \
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29641 \
  tools/check_b2_ddp.py --config-only \
  --config-file configs/dbias/city_2_foggy/domain_det_s0_moe_bn_color_B2_object_aware_ahdr_ddp3.yaml \
  --checkpoint /home/ubuntu/hdx/snorm_E2_repro/debug/city_2_foggy/s0_dbias_color_E2_zdh/E2/best.pth
```

最后执行轻量真实 DDP protected-update audit（默认每 rank 执行 batch 1，不代表正式 batch）：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 SNORM_EXPERIMENT=B2 \
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29642 \
  tools/check_b2_ddp.py \
  --config-file configs/dbias/city_2_foggy/domain_det_s0_moe_bn_color_B2_object_aware_ahdr_ddp3.yaml \
  --checkpoint /home/ubuntu/hdx/snorm_E2_repro/debug/city_2_foggy/s0_dbias_color_E2_zdh/E2/best.pth
```

所有 audit 均通过前不启动正式训练。B2 输出目录为
`./debug/city_2_foggy/s0_dbias_color_B2_object_aware_ahdr_ddp3/E2`。


## C1 Source-Adaptive Diversification

可视化真实 clear-source 样本（不加载 Fog/Night）：

```bash
CUDA_VISIBLE_DEVICES=1 SNORM_EXPERIMENT=C1 python tools/check_c1_augmentation.py \
  --config-file configs/dbias/city_2_foggy/domain_det_s0_moe_bn_color_C1_source_diversification_ddp3.yaml \
  --output-dir diagnostic_runs/c1_source_diversification
```

单卡真实 protected-update audit：

```bash
CUDA_VISIBLE_DEVICES=1 SNORM_EXPERIMENT=C1 python tools/check_c1_training.py \
  --config-file configs/dbias/city_2_foggy/domain_det_s0_moe_bn_color_C1_source_diversification_ddp3.yaml \
  --checkpoint /home/ubuntu/hdx/snorm_E2_repro/debug/city_2_foggy/s0_dbias_color_E2_zdh/E2/best.pth
```

三卡先加 `--config-only` 检查 batch4/rank、global12、1610 steps，再去掉该参数执行轻量 protected-update audit。
