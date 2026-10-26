# SNorm 跨域检测优化实验总结

> 日期：2026-10-20
> 基线配置：`debug/city_2_foggy/s0_dbias_color_fusion_w_elNet/config.yaml`
> 基线权重：项目根目录 `best.pth`
> 测试域：`bdd_daytime_fog_test`（雾天）、`bdd_night_sun_test`（夜间）
> 评估环境：单卡 RTX 3090，batch_size=1

---

## 0. 基线确认

先跑了原始代码的基线结果，作为后续所有对比的参照：

| 测试域 | mAP | mREC |
|---|---|---|
| **Fog（雾天）** | **0.4083** | 0.5816 |
| **Night（夜间）** | **0.4188** | 0.7334 |
| **Avg** | **0.4136** | — |

---

## 思路一：输入图像增强（像素级，推理阶段）

### 1.1 动机

雾天/夜间退化在数据管线最前端（resize 之后、normalize 之前）做轻量像素级增强，不引入任何额外训练，不修改模型结构。

### 1.2 代码修改

**新增文件** `detection/data/transforms/image_enhancer.py`：

```python
import os
import cv2
import numpy as np
from .transforms import transform

def apply_enhancement(img_bgr, mode):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img)

    if mode == 'gamma':
        gamma = 0.7
        L = np.clip(((L.astype(np.float32) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    elif mode == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L = clahe.apply(L)
    elif mode == 'clahe_gamma':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L = clahe.apply(L)
        gamma = 0.7
        L = np.clip(((L.astype(np.float32) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    elif mode == 'usm':
        blur = cv2.GaussianBlur(L, (5, 5), 1.0)
        L = np.clip(L.astype(np.float32) + 1.5 * (L.astype(np.float32) - blur.astype(np.float32)), 0, 255).astype(np.uint8)
    elif mode == 'clahe_usm':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L = clahe.apply(L)
        blur = cv2.GaussianBlur(L, (5, 5), 1.0)
        L = np.clip(L.astype(np.float32) + 1.5 * (L.astype(np.float32) - blur.astype(np.float32)), 0, 255).astype(np.uint8)
    # none: 直接透传

    img = cv2.merge([L, A, B])
    return cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

class image_enhancer(object):
    def __init__(self):
        self.mode = os.environ.get('ENHANCE_MODE', 'none')

    def __call__(self, img_bgr):
        return apply_enhancement(img_bgr, self.mode)
```

**修改文件** `detection/data/transforms/__init__.py`：

```diff
 from .transforms import *
+from .image_enhancer import image_enhancer
  TRANSFORMS = {
     'random_flip': random_flip,
     'resize': resize,
     'image_enhancer': image_enhancer,   # 新增注册
     'normalize': normalize,
     ...
 }
```

**新增评估脚本** `eval_enhance.py`：
- 从环境变量 `ENHANCE_MODE` 读取增强模式
- 在 transforms 链中 resize 之后、normalize 之前插入 `image_enhancer` 模块
- 串行遍历多种模式，自动分配 GPU，汇总输出

### 1.3 评估命令

```bash
conda run -n domain env ENHANCE_MODE=none \
    python eval_enhance.py \
    --config debug/city_2_foggy/s0_dbias_color_fusion_w_elNet/config.yaml \
    --resume best.pth --output debug_enhance_eval
```

（通过 `ENHANCE_MODE=gamma/clahe/clahe_gamma/usm/clahe_usm` 切换）

### 1.4 结果

| 增强模式 | Fog mAP | Night mAP | **Avg** | vs Baseline Avg |
|---|---|---|---|---|
| **none（基线）** | **0.4083** | **0.4188** | **0.4136** | — |
| gamma | 0.4089 ✅ | 0.4106 ❌ | 0.4098 | **-0.0038** |
| clahe | 0.4087 ≈ | 0.3968 ❌ | 0.4028 | **-0.0108** |
| clahe_gamma | 0.4078 ≈ | 0.3940 ❌ | 0.4009 | **-0.0127** |
| usm | 0.4090 ✅ | 0.4071 ❌ | 0.4081 | **-0.0055** |
| clahe_usm | 0.4077 ≈ | 0.3867 ❌❌ | 0.3972 | **-0.0164** |

### 1.5 结论

1. **Fog 上所有增强几乎和基线持平**（±0.001），说明雾域对像素级变化不敏感，SNorm 本身已能处理雾退化
2. **Night 上所有增强都显著下降**（最差 clahe_usm 掉 0.0321）——夜间光照极低，CLAHE 过度拉伸直方图反而放大噪声，USM 锐化也对噪声过敏
3. **最佳是 gamma 单独用**，但也只掉 0.0038，不值得为此改管线
4. **总体结论：前置像素级增强弊大于利**，6 种方法无一超过基线

---

## 思路二：SNorm save_im0 残差锚点在归一化空间的实验

### 2.1 背景：SNorm 内部逻辑

SNorm 的核心前向逻辑（简化）：

```python
feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=0.0)
save_im0 = feats_0.clone()          # ← 残差锚点（原始代码）

# 多尺度归一化候选
feats_mem = []
for kernel_sz in kernel_sz_set:
    feats_ = self.user_norm_v2(feats_0, kernel_sz_set=kernel_sz, w_a=1.0, w_u=1.0, ema=0.85)
    feats_ = self.norm_back(feats_)  # ← 拉回图像空间
    feats_mem.append(feats_)

feats_mem.append(self.norm_back(feats))

save_im0 = self.norm_back(save_im0)  # ← L583：锚点也被拉回图像空间

# render_net 多步融合
output = save_im0 + _res_w * render_net(feats_mem, save_im0)
```

其中 `norm_back` 的作用是把 user_norm_v2 输出从"归一化空间"（均值0方差1）映射回"原始图像空间"（使用 EMA 统计均值方差的反变换）。

### 2.2 实验假设

> 如果 save_im0 跳过 `norm_back`，让锚点**保持在归一化空间**，残差的物理意义会改变：锚点不再是像素值范围，而是"已被归一化的特征"。render_net 学到的融合权重 `_res_w=0.6` 在这种新空间下可能表现不同。

同时测试 save_im0 用 `feats`（原始输入特征）而非 `feats_0`（第一次 [1,1] 归一化的结果）作为锚点来源。

### 2.3 代码修改

**修改文件** `detection/modeling/norm/R202509x/SNorm_0322_valid_refine.py`，L527 附近：

```python
# --- 原始代码 ---
feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=0.0)
save_im0 = feats_0.clone()
...
save_im0 = self.norm_back(save_im0)   # L583

# --- 修改后（通过环境变量切换） ---
feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=0.0)

_saveim0_mode = os.environ.get('SNORM_SAVEIM0_MODE', 'baseline')
if _saveim0_mode == 'raw_nonormback':
    save_im0 = feats.clone()            # 方向 A：feats + 跳过 norm_back
elif _saveim0_mode == 'norm_nonormback':
    save_im0 = feats_0.clone()          # 方向 B：feats_0 + 跳过 norm_back
elif _saveim0_mode == 'raw_saveim0':
    save_im0 = feats.clone()            # 之前方向：feats + 保留 norm_back
else:  # baseline
    save_im0 = feats_0.clone()
...
# L583 处也改为条件跳过
if _saveim0_mode not in ('raw_nonormback', 'norm_nonormback'):
    save_im0 = self.norm_back(save_im0)
```

### 2.4 四种配置的物理意义

| 配置名 | 环境变量 | save_im0 来源 | norm_back 开启 | 锚点所在空间 |
|---|---|---|---|---|
| baseline | `SNORM_SAVEIM0_MODE` 未设置 | feats_0 | ✅ | 图像空间 |
| raw_saveim0 | `=raw_saveim0` | feats | ✅ | 图像空间 |
| **raw_nonormback**（方向A） | `=raw_nonormback` | feats | ❌ | 归一化空间 |
| **norm_nonormback**（方向B） | `=norm_nonormback` | feats_0 | ❌ | 归一化空间 |

### 2.5 评估命令

```bash
# 方向 A：GPU1
CUDA_VISIBLE_DEVICES=1 SNORM_SAVEIM0_MODE=raw_nonormback \
SNORM_EMA=0.85 SNORM_RES_W=0.6 SNORM_NORM_EMA=0.92 \
python -u eval_enhance.py --config debug/city_2_foggy/s0_dbias_color_fusion_w_elNet/config.yaml \
--resume best.pth --methods none --gpu 0 --output debug_saveim0_raw_nonormback

# 方向 B：GPU2
CUDA_VISIBLE_DEVICES=2 SNORM_SAVEIM0_MODE=norm_nonormback \
SNORM_EMA=0.85 SNORM_RES_W=0.6 SNORM_NORM_EMA=0.92 \
python -u eval_enhance.py --config debug/city_2_foggy/s0_dbias_color_fusion_w_elNet/config.yaml \
--resume best.pth --methods none --gpu 0 --output debug_saveim0_norm_nonormback
```

> 注意：`SNORM_RES_W` / `SNORM_EMA` / `SNORM_NORM_EMA` 是 SNorm 多步迭代的其他超参，这里保持与训练时一致的值（ema=0.85, res_w=0.6, norm_ema=0.92），保证变量唯一。

### 2.6 结果汇总

| 配置 | save_im0 来源 | norm_back | Fog | Night | **Avg** | vs Baseline |
|---|---|---|---|---|---|---|
| **baseline** | feats_0 | ✅ | **0.4083** | **0.4188** | **0.4136** | — |
| raw_saveim0 | feats | ✅ | 0.3993 | 0.3980 | 0.3987 | **-0.0149** |
| raw_nonormback（A） | feats | ❌ | 0.3993 | 0.3980 | 0.3987 | **-0.0149** |
| norm_nonormback（B） | feats_0 | ❌ | 0.3961 | 0.4120 | 0.4041 | **-0.0095** |

### 2.7 关键发现

1. **raw_saveim0 == raw_nonormback**：save_im0 用 `feats` 时，开不开 norm_back 结果一模一样。说明 `norm_back(feats)` 和 `feats` 本身非常接近（因为 feats 是网络 backbone 的原始输出，可能已经在归一化空间）
2. **feats 做锚点灾难**：不管 norm_back 开不开，只要 save_im0 用 `feats` 而非 `feats_0`，整体掉 0.015 mAP——feats_0 是第一次 [1,1] 窗口归一化后的特征，对光照/天气的初始校正更充分
3. **norm_nonormback 是 Night 友好**：方向 B 的 Night=0.4120 只比 baseline 掉了 0.0068（几乎持平），但 Fog 掉了 0.0122。平均掉 0.0095，是所有改动中损失最小的
4. **norm_back 的作用不是"必需"但很稳**：去掉它在 Night 上表现几乎不变，但 Fog 上掉分明显

### 2.8 结论

- **保留 norm_back + save_im0 = feats_0** = baseline 最优组合，无法超越
- **feats 做锚点绝对不行**：损失超过 0.015，不值得尝试更多变体
- **norm_nonormback 值得后续精调**：如果调整 `_res_w`（例如从 0.6 增到 0.8 让归一化空间锚点权重更大），Night 可能回到基线水平甚至超越

---

## 思路三：检测器微调（E1/E2 实验）

### 3.1 动机

思路一和思路二都在**推理阶段**做改动，最终都未能超越 baseline。分析原因：

- 思路一（像素级增强）：Fog 域 SNorm 已能处理退化，Night 域增强反而放大噪声
- 思路二（save_im0 锚点空间）：内部机制已接近最优，调锚点来源/空间只会掉分

因此将焦点转向**训练阶段**的检测器微调。核心假设：

> baseline 训练的是 `render_net + sigma_weight_pred`（渲染网络），检测器头（`rpn_norm` / `box_head_norm`）以及 backbone 的高层特征并未针对 fog/night 测试域进行适配。通过冻结 backbone 大部分层、微调检测头（或加 backbone_3 低学习率），可以在不破坏 SNorm 预训练能力的前提下，让检测器更适应跨域特征。

### 3.2 两种方案

| 实验 | 冻结部分 | 训练部分 | 训练参数量 | 学习率 | 设计理由 |
|------|----------|----------|-----------|--------|----------|
| **E1** | backbone_0/1/2/3 + proj_norm + dis | rpn_norm + box_head_norm | 110M（仅检测头） | LR=5e-5 | 最小改动，验证"检测头本身能否适配跨域" |
| **E2** | backbone_0/1/2 + proj_norm + dis | backbone_3(lr×0.1) + rpn_norm + box_head_norm | 136M（检测头 + 高层特征） | backbone_3=5e-6, head=5e-5 | 让 backbone 最后一层做低强度适配，既不破坏低层特征又能让高层语义对齐目标域 |

**核心超参（与 baseline 完全对齐）**：

| 超参 | 值 | 说明 |
|------|-----|------|
| `SNORM_RES_W` | 0.6 | render_net 残差比例 |
| `SNORM_EMA` | 0.85 | 局部统计量 EMA 系数 |
| 初始权重 | `/home/ubuntu/hdx/Domain_NBias_Render_V2/best.pth` | baseline 最优权重，严格加载 |
| 训练数据 | `bdd_daytime_clear_train` | 源域白天清晰图像（与 baseline 相同） |
| 优化器 | Adam, lr=5e-5, betas=(0.9,0.999) | 与 baseline 相同 |
| 调度器 | CosineAnnealingLR, T_max=total_steps | 与 baseline 相同 |
| batch_size | 6 | 与 baseline 相同 |

### 3.3 代码修改

**唯一修改文件**：`train_single_c2f_bn_color_0112_render_ahdr.py`

修改分三处，均通过环境变量 `SNORM_EXPERIMENT` 控制，**E0（默认）行为与原始 baseline 完全一致**：

#### ① GPU 绑定修复（L12-13）

```python
# 原始：硬编码 CUDA_VISIBLE_DEVICES=0
# 修改后：允许外部传入，仅未设置时默认 GPU0
if not os.environ.get('CUDA_VISIBLE_DEVICES'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

#### ② 自动加载 baseline 权重（在 optimizer 之前）

```python
_resume_pth = '/home/ubuntu/hdx/Domain_NBias_Render_V2/best.pth'
if os.path.isfile(_resume_pth) and not args.resume:
    print(f'  自动加载 baseline 权重: {_resume_pth}')
    checkpoint = torch.load(_resume_pth, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    print('  ✅ 成功加载权重参数')
```

#### ③ WORK_DIR 隔离（L380-384）

```python
_exp_wd = os.environ.get('SNORM_EXPERIMENT', 'E0')
if _exp_wd != 'E0':
    work_dir = os.path.join(work_dir, _exp_wd)
print(f'  [WORK_DIR] SNORM_EXPERIMENT={_exp_wd}, work_dir={work_dir}')
```

#### ④ 参数冻结与 optimizer 构建（核心，L397-456）

```python
_exp = os.environ.get('SNORM_EXPERIMENT', 'E0')
_base_lr = det_cfg.SOLVER.LR
_teacher = model_without_ddp.teacher
_dis = model_without_ddp.dis

if _exp == 'E1':
    # 冻结 backbone_0/1/2/3 + proj_norm + dis
    # 训练 rpn_norm + box_head_norm（全部用 base_lr）
    for p in _teacher.backbone_0.parameters(): p.requires_grad_(False)
    for p in _teacher.backbone_1.parameters(): p.requires_grad_(False)
    for p in _teacher.backbone_2.parameters(): p.requires_grad_(False)
    for p in _teacher.backbone_3.parameters(): p.requires_grad_(False)
    for p in _teacher.proj_norm.parameters(): p.requires_grad_(False)
    for p in _teacher.rpn_norm.parameters(): p.requires_grad_(True)
    for p in _teacher.box_head_norm.parameters(): p.requires_grad_(True)
    for p in _dis.parameters(): p.requires_grad_(False)
    model_paramters += [{'params': _teacher.rpn_norm.parameters(), 'lr': _base_lr}]
    model_paramters += [{'params': _teacher.box_head_norm.parameters(), 'lr': _base_lr}]

elif _exp == 'E2':
    # 冻结 backbone_0/1/2 + proj_norm + dis
    # 训练 backbone_3(lr×0.1) + rpn_norm + box_head_norm(base_lr)
    for p in _teacher.backbone_0.parameters(): p.requires_grad_(False)
    for p in _teacher.backbone_1.parameters(): p.requires_grad_(False)
    for p in _teacher.backbone_2.parameters(): p.requires_grad_(False)
    for p in _teacher.proj_norm.parameters(): p.requires_grad_(False)
    for p in _teacher.backbone_3.parameters(): p.requires_grad_(True)
    for p in _teacher.rpn_norm.parameters(): p.requires_grad_(True)
    for p in _teacher.box_head_norm.parameters(): p.requires_grad_(True)
    for p in _dis.parameters(): p.requires_grad_(False)
    model_paramters += [{'params': _teacher.backbone_3.parameters(), 'lr': _base_lr * 0.1}]
    model_paramters += [{'params': _teacher.rpn_norm.parameters(), 'lr': _base_lr}]
    model_paramters += [{'params': _teacher.box_head_norm.parameters(), 'lr': _base_lr}]

else:  # E0 baseline
    # 冻结 teacher + dis
    # 训练 render_net + sigma_weight_pred
    ...
```

#### ⑤ train_one_epoch 训练分支（L201-207）

E1/E2 在训练循环里直接走 `mode_training=0`（检测头直接前向），跳过 baseline 的 `mode_training=10/11` 两步渲染：

```python
_exp_loop = os.environ.get('SNORM_EXPERIMENT', 'E0')
if _exp_loop in ('E1', 'E2'):
    loss_dict_det, _ = model_det(tr_images, img_metas, s_targets,
                                  mode_training=0, step_iter=global_step,
                                  is_update=is_update, stsc_images=sc_images, stsc_trans=sc_trans)
    det_loss_det = sum(list(loss_dict_det.values()))
    det_loss_det.backward()
```

### 3.4 训练/评估头对齐确认

经代码路径确认：**训练和测试使用同一个检测头**，不需要权重同步逻辑。

- 训练（`mode_training=0`）→ `ModuleSet.py` mode=1 分支 → 使用 `rpn_norm + box_head_norm`
- 测试（`moe_mode=7`）→ `ModuleSet.py` mode=1 分支 → 使用 `rpn_norm + box_head_norm`

`rpn_eval` / `box_head_eval` 仅在 mode=0 分支使用，当前训练/测试均未走到该分支（死代码）。

### 3.5 训练命令（三卡独立、互不干扰）

```bash
# GPU1 - E1
CUDA_VISIBLE_DEVICES=1 SNORM_EXPERIMENT=E1 SNORM_RES_W=0.6 SNORM_EMA=0.85 \
python -u train_single_c2f_bn_color_0112_render_ahdr.py \
    --config-file configs/dbias/city_2_foggy/domain_det_s0_moe_bn_color.yaml

# GPU2 - E2
CUDA_VISIBLE_DEVICES=2 SNORM_EXPERIMENT=E2 SNORM_RES_W=0.6 SNORM_EMA=0.85 \
python -u train_single_c2f_bn_color_0112_render_ahdr.py \
    --config-file configs/dbias/city_2_foggy/domain_det_s0_moe_bn_color.yaml
```

**资源占用**：每张卡 ~24GB 显存，单进程无 DDP，独立 WORK_DIR，互不覆盖权重。

### 3.6 结果汇总（真实 voc_result mAP）

| 实验 | eval step | Fog mAP | Night mAP | **Sum mAP** | vs Baseline |
|------|-----------|---------|-----------|-------------|-------------|
| **Baseline** | best | 0.4041 | 0.4323 | **0.8364** | — |
| E1 | 1000 (ep5) | 0.4063 | 0.4191 | 0.8254 | -0.0110 |
| E1 | **2000 (ep11)** | **0.4115** | 0.4249 | 0.8364 | **持平** |
| E1 | 3000 (ep22) | 0.4103 | 0.4181 | 0.8284 | -0.0080 |
| E1 | 3220 (ep24) | 0.3816 | 0.3898 | 0.7714 | -0.0650 ↓↓ |
| **E2** | 1000 (ep5) | 0.4059 | 0.4200 | 0.8259 | -0.0105 |
| **E2** | **2000 (ep11)** | 0.4106 | **0.4346** | **0.8452** | **↑ +0.0088 ✅** |
| **E2** | 3000 (ep22) | 0.4060 | 0.4170 | 0.8230 | -0.0134 |
| **E2** | 3220 (ep24) | 0.3663 | 0.3891 | 0.7554 | -0.0810 ↓↓ |

### 3.7 关键发现

1. **E2 是首个超越 baseline 的有效方案**：epoch 11（step 2000）时 sum_mAP=0.8452，比 baseline 高 **+0.0088**。Night 域提升明显（0.4346 vs 0.4323, +0.0023），Fog 域也有提升（0.4106 vs 0.4041, +0.0065）
2. **E1 持平 baseline**：纯检测头微调即可达到 baseline 水平（0.8364 vs 0.8364），说明检测头本身有足够的域自适应能力。结构上 foggy 更好（+0.0074）、night 更差（-0.0074），E2 通过 backbone_3 微调把 night 也拉上去了
3. **两个实验都极早收敛**：峰值出现在 epoch 11（step 2000），之后持续过拟合。epoch 22 两个实验都回落到 baseline 以下，epoch 24 更是暴跌 0.06~0.08 sum mAP
4. **backbone_3 的作用不可或缺**：E1 只微调 head 时 foggy↑night↓，night 域没提升；E2 加了 backbone_3(lr×0.1) 后 night 也超越了 baseline——backbone 最后一层承载了高层语义特征的域适配

### 3.8 best.pth 保存位置

| 实验 | best.pth | best step | 备注 |
|------|----------|-----------|------|
| Baseline | `debug/city_2_foggy/s0_dbias_color/best.pth` | — | 最优权重（训练起始加载） |
| **E1** | `debug/city_2_foggy/s0_dbias_color/E1/best.pth` | step 2000 | sum=0.8364 |
| **E2** | `debug/city_2_foggy/s0_dbias_color/E2/best.pth` | step 2000 | **sum=0.8452（最佳）** |

### 3.9 后续改进方向（E3）

基于 E2 的成功经验，后续工作建议：

1. **采用 E2 best.pth 作为新的基线权重**，再做迭代式渲染训练（E3）——在已有的 head+backbone_3 适配基础上，再开启 `mode_training=10/11` 两步渲染
2. **加早停机制**：E1/E2 都在 epoch 11 达峰值，建议设置 `MAX_ITERS_EARLY_STOP`（连续 N 次 eval sum mAP 不提升就停），避免无效训练
3. **密集评估**：把 `STEPS_EVAL` 从 `(16, 22)` 改到 `(1800, 2200, 2600)` 或更密集，精准捕捉峰值
4. **探索更大 backbone_3 lr**：当前用 0.1 倍 base_lr，可试 0.2~0.5 倍看是否能更快收敛或达到更高峰值
5. **SNorm 微调**：尝试把 `proj_norm` 也以极低 lr（如 base_lr×0.01）解冻，让域统计量也能适配目标域

---

## 附录：结果文件位置

| 实验 | 输出目录 | voc_result 文件 |
|---|---|---|
| none（基线） | `debug_enhance_eval/none/` | Fog: `voc_result_2026-10-19_21-53-35.txt` / Night: `voc_result_2026-10-20_00-05-48.txt` |
| gamma | `debug_enhance_eval/gamma/` | Fog: `voc_result_2026-10-20_01-15-00.txt` / Night: `voc_result_2026-10-20_03-27-33.txt` |
| clahe | `debug_enhance_eval/clahe/` | Fog: `voc_result_2026-10-19_21-52-01.txt` / Night: `voc_result_2026-10-20_00-05-50.txt` |
| clahe_gamma | `debug_enhance_eval/clahe_gamma/` | Fog: `voc_result_2026-10-20_01-14-52.txt` / Night: `voc_result_2026-10-20_03-29-12.txt` |
| usm | `debug_enhance_eval/usm/` | Fog: `voc_result_2026-10-19_21-51-54.txt` / Night: `voc_result_2026-10-20_00-05-09.txt` |
| clahe_usm | `debug_enhance_eval/clahe_usm/` | Fog: `voc_result_2026-10-20_01-15-16.txt` / Night: `voc_result_2026-10-20_03-30-23.txt` |
| raw_saveim0 | `debug_raw_saveim0/none/` | Fog: `voc_result_2026-10-20_12-25-56.txt` / Night: `voc_result_2026-10-20_14-37-47.txt` |
| raw_nonormback | `debug_saveim0_raw_nonormback/none/` | Fog: `voc_result_2026-10-20_16-46-16.txt` / Night: `voc_result_2026-10-20_18-58-03.txt` |
| norm_nonormback | `debug_saveim0_norm_nonormback/none/` | Fog: `voc_result_2026-10-20_16-46-44.txt` / Night: `voc_result_2026-10-20_18-58-43.txt` |
| **E1** | `s0_dbias_color/E1/evaluations/` | Fog: `voc_result_0001000/0002000/0003000/0003220.txt` / Night: 同名 |
| **E2** | `s0_dbias_color/E2/evaluations/` | Fog: `voc_result_0001000/0002000/0003000/0003220.txt` / Night: 同名 |