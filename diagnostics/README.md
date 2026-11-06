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
