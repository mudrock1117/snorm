# SNorm 评估索引

## B1 三卡阶段评估（2026-11-12）

- 测试集保持 `bdd_daytime_fog_test` 与 `bdd_night_sun_test`，指标保持 VOC mAP。
- 分布式评估由三个 rank 分片推理并 all-gather，只有 rank0 计算 VOC 指标和写输出；所有 rank 同步进入和退出评估，随后恢复 train mode。
- B1 DDP 默认 global optimizer step 500、1000、1500、1650、2000 保存独立 checkpoint 并评估，环境变量 `B1_DDP_EVAL_STEPS` 可显式覆盖。
- 日志分别报告 Fog mAP、Night mAP 和两者 Sum；这些测试域数据不得进入训练 loss、Gate supervision 或自适应统计。
