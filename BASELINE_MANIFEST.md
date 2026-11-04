# SNorm/AHDR GitHub 基线清单

## 冻结范围

- 源仓库：`mudrock1117/snorm`
- 源分支：`snorm_optimization_v2`
- 固定 commit：`a98e4fb069d382740ff0a81734f432f5285dcce2`
- 只读恢复分支：`backup/snorm_baseline_20261104`
- 开发分支起点：`feature/ahdr_structural_v3`
- 冻结日期：2026-11-04

源分支、备份分支和开发分支创建时均已通过 GitHub refs API 核验为上述完整 SHA。由于当前连接器不支持写 tag，本基线以备份分支和完整 commit SHA 双重标识。

## 核心代码与配置

- `train_single_c2f_bn_color_0112_render_ahdr.py`：E1/E2 训练入口。
- `detection/modeling/ModuleSet.py`：检测器、SNorm 与训练阶段装配。
- `detection/modeling/norm/R202509x/SNorm_0322_valid_refine.py`：实际导入的 SNorm、统计状态和 AHDR 推理链路。
- `detection/modeling/HDR/OURS/OHDR.py`：AHDR 渲染网络。
- `detection/engine/eval.py`：Fog/Night 评估入口。
- `configs/dbias/city_2_foggy/domain_det_s0_moe_bn_color.yaml`：已识别的训练配置。
- `EXPERIMENT_SUMMARY.md`：仓库内历史实验记录。

## 已知实验结果

| 权重/实验 | Fog | Night | Sum | 身份说明 |
|---|---:|---:|---:|---|
| 历史 E2 | 0.4106 | 0.4346 | 0.8452 | 历史 E2 参考结果 |
| zdh_original | 0.4122 | 0.4232 | 0.8354 | **不是 E2 权重** |
| initial_best / B6 | 0.4071 | 0.4183 | 0.8254 | 当前已知 B6 结果 |

以上数值来自项目背景与已有记录；当前环境没有服务器、数据集和 GPU checkpoint，尚未执行统一协议复测，不能用于宣称新代码带来提升。

## 未备份与未核实事项

本次冻结仅代表 GitHub 远端在固定 commit 的内容，不包含：

- 服务器工作树中的未提交修改；
- checkpoint、优化器状态、SNorm 内部 memory 或训练中间状态；
- datasets、实验日志、缓存和模型权重；
- CUDA/PyTorch/依赖版本及关键运行环境变量；
- 历史 E2、zdh_original、initial_best/B6 checkpoint 的文件 SHA256 与初始化来源；
- epoch 边界精度断崖的根因和各权重评估协议的一致性。

恢复完整实验仍需外部 checkpoint、数据集、原始环境与评估配置；不得把本 GitHub 基线描述为服务器完整备份或完全可复现实验快照。

## 恢复方法

推荐使用不可歧义的完整 commit：

```bash
git fetch origin
git switch --detach a98e4fb069d382740ff0a81734f432f5285dcce2
```

或从备份分支建立本地恢复分支：

```bash
git fetch origin backup/snorm_baseline_20261104
git switch -c restore/snorm_baseline_20261104 origin/backup/snorm_baseline_20261104
git rev-parse HEAD
```

最后一条命令必须输出 `a98e4fb069d382740ff0a81734f432f5285dcce2`。备份分支不得改写、删除或强制推送。
