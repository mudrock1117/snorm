# Guided-AHDR v1 结构设计（待诊断验证）

## 状态

本文件只定义第一版实验机制和切换接口，尚未修改 AHDR 网络。原因是当前仍需由服务器诊断确认 Night 样本的逐步高频、目标响应和检测变化；不在证据缺失时同时堆叠多个机制。

## 已确认的代码问题

1. `AHDR.forward(x0, x1, x2)` 只编码 `x0/x1`，`x2` 仅决定返回分支。
2. `scale_interact` 已实例化但 forward 未调用。
3. `fusion_f` 输出三通道，实际只取前两通道并重新归一化。
4. 六步中第二输入是上一步输出，但网络本身没有显式递进状态；每一步又与 `save_im0` 固定回注。
5. 第六步 reference 来源可配置，历史结果显示原始信息回注值得单独诊断，不能直接移除。

## 单一实验机制：共享编码的 guide-preserving state update

目标不是提高视觉亮度，而是让 guide 参与每个尺度的恢复，同时保留旧 AHDR 完整路径。

对尺度 (s)：

1. 使用现有共享 encoder（不新增 guide encoder 参数）得到
   (f_0^s=E_s(x_0), f_1^s=E_s(x_1), f_g^s=E_s(x_2))。
2. 保持现有二输入融合得到候选 (c^s=F_s(f_0^s,f_1^s))。
3. 使用轻量 1×1 门控预测逐位置更新量：
   (z^s=\sigma(W_s[c^s,f_g^s,c^s-f_g^s]))。
4. 生成保真状态：
   (h^s=f_g^s+z^s\odot(c^s-f_g^s))。
   当门控接近 0 时保留 guide 语义/高频；接近 1 时接受候选恢复。
5. 将 ([h^1,h^2,h^3]) 送入已有 `scale_interact`，再进入 decoder。
6. 图像级输出继续使用残差形式，并保留 SNorm 外层 `save_im0` 回注。

该机制只有每尺度一个小型 1×1 门控；guide encoder 与原 encoder 共享权重。预计新增参数远小于复制完整 encoder，具体参数/FLOPs 必须由服务器脚本实测后记录。

## 开关与兼容性

计划接口：

- 默认：`SNORM_AHDR_VARIANT=legacy`，完全沿用旧 `AHDR`。
- 实验：`SNORM_AHDR_VARIANT=guided_v1`。
- 原 E2 配置不修改；新增独立 YAML 或启动脚本设置实验开关。
- checkpoint 加载使用 `strict=False`：
  - 旧参数按原 key 加载；
  - 新门控最后一层 bias 初始化为负值，使初始状态偏向 guide；
  - 新增参数在加载日志中明确列为 missing keys；
  - legacy 模式不调用新参数，作为直接回退路径。

## 首轮验证假设

仅当诊断满足下列至少一项时实现：

- Night 小目标区域的 target/background ratio 随 step 持续下降；
- 高频能量下降与检测 score/召回同步恶化；
- 相邻 step change_rate 很小，证明多步退化为重复映射；
- step6 原始/guide 回注能够恢复目标响应或检测框。

## 消融顺序

1. legacy；
2. shared guide encoding，仅记录不融合；
3. guide-preserving gate，不启用 scale interaction；
4. guide-preserving gate + 现有 scale interaction；
5. step1/3/6 检测监督对比。

不同时引入 EMA、Gamma/CLAHE、简单注意力堆叠或额外融合系数。所有精度结论必须通过相同 Fog/Night VOC 协议实测。
