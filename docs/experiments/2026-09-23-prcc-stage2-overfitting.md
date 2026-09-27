# PRCC Stage-2 过拟合诊断（2026-09-23）

## 1. 结论摘要

- 当前 recipe 在 PRCC 20-ID holdout 上的换衣 mAP 峰值约 **0.87–0.885**，出现在
  Stage-2 第 4 个 epoch，之后持续下降，第 10 个 epoch 跌到 0.76–0.82。
- 下降的原因是**训练集在第 4–5 个 epoch 就被学透**。So400m 视觉塔全量微调，
  而 PRCC 训练集只有约 150 个身份：
  - `triplet_active_fraction` 从 1.0 跌到 0.02 以下，triplet 基本失效；
  - `reid_loss` 停在 0.82–0.83，接近 label smoothing 0.1 的下限。
- 相机均衡 PK 采样 + 灰度增强（p=0.2）**没有提升**：峰值 0.872 对 0.885，后期下降
  更快（第 10 个 epoch 0.760 对 0.823）。两项改动在同一次运行里同时开启，无法区分
  各自的影响。
- Euclidean triplet 在默认配置下接近失效（第 5 个 epoch `triplet_active_fraction`
  已经只有 0.0025）。换成 cosine triplet 后该项在前 3 个 epoch 仍有信号，
  这个改动应保留。
- Fused 检索相对 image-only 在 PRCC 上几乎没有增益（差值不超过 0.0015）。这与
  `DESIGN.md` §9 的预期一致：PRCC 的 gallery 和 query 各自只有一个 camera。

## 2. 实验设置

所有运行都在 AutoDL RTX 4090 48GB 上完成，共用以下设置：

- 数据与验证：PRCC RGB，`validation_holdout_ids=20`，`validation_holdout_seed=0`。
  指标为留出身份上的 A/C 换衣 mAP，不读取 test split。
- 训练：`seed=0`，Stage-1 共 60 个 epoch，Stage-2 的 epoch 号从 61 开始。
- 其余参数：batch 64（P=16 × K=4），BF16，BNNeck，`retrieval_mode=fused`。

| 运行 | 相对默认 recipe 的改动 | 验证间隔 |
|---|---|---|
| `prcc-baseline-s0` | 无（euclidean triplet，`image_encoder_lr=5e-6`，Stage-2 共 60 个 epoch） | 5 |
| `prcc-cos-lr2e6-e20-s0` | `triplet_metric=cosine image_encoder_lr=2e-6 epochs=20` | 1 |
| `prcc-cos-lr2e6-e20-cb-gray02-s0` | 在上一行基础上加 `camera_balanced_sampling=true grayscale_prob=0.2` | 1 |

三次运行都在 Stage-2 第 10–15 个 epoch 手动停止。

## 3. 结果

### 3.1 Holdout 换衣 mAP（fused）

| Stage-2 epoch | baseline | cos + lr 2e-6 | cos + lr 2e-6 + 相机均衡 + 灰度 |
|---:|---:|---:|---:|
| 1 | – | 0.6689 | 0.6616 |
| 2 | – | 0.8077 | 0.8015 |
| 3 | – | 0.8552 | 0.8409 |
| 4 | – | **0.8847** | **0.8721** |
| 5 | **0.8390** | 0.8778 | 0.8709 |
| 6 | – | 0.8782 | 0.8347 |
| 7 | – | 0.8657 | 0.8053 |
| 8 | – | 0.8652 | 0.7757 |
| 9 | – | 0.8267 | 0.7782 |
| 10 | 0.7345 | 0.8233 | 0.7602 |
| 15 | 0.7008 | – | – |

### 3.2 Rank-1

| Stage-2 epoch | baseline | cos + lr 2e-6 | + 相机均衡 + 灰度 |
|---:|---:|---:|---:|
| 4 | – | 0.8905 | 0.8814 |
| 5 | 0.8552 | 0.8757 | 0.8757 |
| 10 | 0.6910 | 0.8050 | 0.7651 |

### 3.3 `triplet_active_fraction`（epoch 均值）

| Stage-2 epoch | baseline（euclidean） | cos + lr 2e-6 | + 相机均衡 + 灰度 |
|---:|---:|---:|---:|
| 1 | – | 1.0000 | 1.0000 |
| 3 | – | 0.8338 | 0.9328 |
| 4 | – | 0.4341 | 0.5696 |
| 5 | 0.0025 | 0.1856 | 0.2711 |
| 6 | – | 0.0793 | 0.1195 |
| 10 | 0.0003 | 0.0127 | 0.0172 |

mAP 的峰值正好出现在 triplet 从“多数 anchor 有效”变为“少数有效”的那个 epoch。
此后 ID loss 已经到达 label smoothing 的下限，训练在记忆训练身份，泛化开始变差。

### 3.4 Stage-1

| 指标 | cos + lr 2e-6 | + 相机均衡 + 灰度 |
|---|---:|---:|
| Stage-1 epoch 60 loss | 1.1065 | 1.0444 |

相机均衡与灰度让 Stage-1 的 prompt 对齐 loss 更低，但没有传导到 Stage-2 的 mAP。

## 4. 解读

1. **瓶颈是过拟合，不是欠训练。** 三次运行都在 Stage-2 前 4–6 个 epoch 到顶。
   在同一 recipe 上继续加 epoch 或加快学习率不会抬高峰值。
2. **缩短 Stage-2 的作用是让“最后一个 epoch”落在峰值附近，而不是抬高峰值本身。**
   - 最终的 test 运行（`validation_holdout_ids=0`）强制只在最后一个 epoch 评估，
     不能用 `best.pth` 挑 epoch。
   - `epochs=20` 时最终报告的是已经过拟合的模型。
   - 按 `DESIGN.md` §9，最终 epoch 数必须比较不同 `epochs` 设置下各自最终 epoch
     的 holdout mAP 来选。
   - 让 cosine 学习率在峰值附近退火到底，通常还能额外多出 1–2 个点。
3. **单次运行的差异需要谨慎解读。** holdout 只有 20 个身份，0.872 与 0.885 的差距
   可能在种子噪声范围内。结论性的对比至少需要 2 个 seed。

## 5. 下一步（按预期收益排序）

1. **限制视觉塔的可训练部分**：冻结前约 2/3 的 block，或使用 layer-wise LR decay。
   直接针对大模型在小数据上学透太快的问题。
2. **权重 EMA 评估**：维护滑动平均权重用于验证和测试。实现成本低，可以拉平峰值后的
   下降，并能与第 1 项叠加。
3. **换衣对抗损失（CAL 式 clothes-adversarial loss）**：PRCC 的衣服标签可直接由
   `(pid, camera ∈ {A,B} / C)` 得到，针对的正是换衣主指标。改动面比第 1、2 项大。
4. **更强正则**：提高 random erasing 概率、加大 label smoothing 或 weight decay。
   只需改配置，但预期收益较小。
5. **拆开消融**：在更优的 recipe 上分别单独开启相机均衡采样和灰度增强，确认各自是否
   有害。
6. **确定最终 Stage-2 epoch 数**：以上改动定稿后，比较 `epochs ∈ {5, 6, 8}` 各自的
   最终 epoch holdout mAP，再以 `validation_holdout_ids=0` 重训并报告 test。

第 1、2 项已实现：
- 第 1 项：`image_encoder_frozen_layers`、`image_encoder_layer_decay`。
- 第 2 项：`model_ema_decay`；验证时主指标用 EMA 权重，`raw_mAP` 记录训练权重。

第 3 项已实现：`clothes_adversarial_weight`（见 §8）。

## 6. 追加实验：冻结前 18 层 + EMA（`prcc-cos-lr2e6-e20-fz18-ema997-s0`）

在 `prcc-cos-lr2e6-e20-s0` 的配置上加 `image_encoder_frozen_layers=18 model_ema_decay=0.997`
（EMA 带均匀平均 warmup），于 Stage-2 第 13 个 epoch 手动停止。

| Stage-2 epoch | EMA mAP | raw_mAP | `triplet_active_fraction` | 全量微调 raw（对照） |
|---:|---:|---:|---:|---:|
| 2 | 0.5410 | 0.6111 | 1.0000 | 0.8077 |
| 4 | 0.6713 | 0.7191 | 0.9299 | **0.8847** |
| 6 | 0.7285 | 0.7426 | 0.5708 | 0.8782 |
| 8 | 0.7374 | 0.7335 | 0.3493 | 0.8652 |
| 10 | **0.7395** | 0.7349 | 0.2120 | 0.8233 |
| 12 | 0.7358 | 0.7307 | 0.1383 | – |

- **冻结前 18 层有害。** 训练集照样被学透（`reid_loss` 0.855，triplet 有效比例降到 0.14），
  但 holdout 上限从 0.885 降到 0.74。过拟合不是"可训练参数太多"造成的，PRCC 换衣需要底层
  也参与适配。放弃大比例冻结。
- **EMA 符合预期。** mAP 走平后不再下滑；从第 7 个 epoch 起 EMA 比 raw 高 0.001–0.005；
  上升阶段落后 raw 0.05–0.07，这是平均窗口带来的正常滞后。
- 下一轮：全量微调 + `model_ema_decay=0.997`（`prcc-cos-lr2e6-e20-ema997-s0`），单独衡量
  EMA 能否稳住或抬高 0.885 的峰值；之后再试 `image_encoder_layer_decay≈0.85`。

## 7. 全量微调 + EMA（`prcc-cos-lr2e6-e20-ema997-s0`）

配置同 `prcc-cos-lr2e6-e20-s0`，只加 `model_ema_decay=0.997`。第一次启动在 Stage-2 第 3 个
epoch 被关机中断，已删除重跑；第二次在第 4 个 epoch 中途因关机中断，从第 3 个 epoch 的
`last.pth` 续训（续训不恢复数据采样随机状态，训练顺序与不中断时不逐位相同）。

| Stage-2 epoch | EMA mAP | raw_mAP | `triplet_active_fraction` | 对照 raw（无 EMA） |
|---:|---:|---:|---:|---:|
| 1 | 0.5618 | 0.6668 | 1.0000 | 0.6689 |
| 2 | 0.7086 | 0.8065 | 0.9965 | 0.8077 |
| 3 | 0.7990 | 0.8557 | 0.8364 | 0.8552 |
| 4 | 0.8399 | 0.8736 | 0.4463 | **0.8847** |
| 5 | 0.8534 | **0.8867** | 0.1907 | 0.8778 |
| 6 | **0.8612** | 0.8593 | 0.0803 | 0.8782 |
| 7 | 0.8556 | 0.8417 | 0.0382 | 0.8657 |
| 8 | 0.8531 | 0.8230 | 0.0239 | 0.8652 |
| 9 | 0.8444 | 0.8330 | – | 0.8267 |
| 10 | 0.8347 | 0.8096 | – | 0.8233 |
| 11 | 0.8248 | 0.7931 | – | – |

- **EMA 不抬高峰值。** EMA 峰值 0.8612（第 6 个 epoch），低于同一轮 raw 峰值 0.8867（第 5 个
  epoch）。raw 曲线只在 1–2 个 epoch 内处于高点，而 0.997 的平均窗口约 333 步（1.4 个 epoch），
  把上升期较弱的权重也平均进来了。
- **EMA 显著减缓峰值后的下滑。** 第 6→8 个 epoch raw 掉了 0.036，EMA 只掉了 0.008；第 8 个
  epoch EMA 比 raw 高 0.030。
- 对最终的 test 协议（只能报告最后一个 epoch，不能用 `best.pth` 挑）有价值：EMA 让"最后一个
  epoch"对 Stage-2 长度不那么敏感。
- 本轮 raw 曲线与对照基本重合（峰值 0.8867 对 0.8847，早晚一个 epoch），两轮之间的差异在单 seed
  噪声范围内。
- 第 9–11 个 epoch EMA 也开始下滑（0.8444 → 0.8248），只是比 raw 慢；第 11 个 epoch 后手动停止。

## 8. 进行中：LLRD 与 CAL（2026-09-27 启动）

机器换成 RTX 6000D 85GB，数据盘 150G，两轮同时在一张卡上训练。共用配置：
`seed=0 triplet_metric=cosine image_encoder_lr=2e-6 epochs=20 validation_interval=1
model_ema_decay=0.997`。EMA 不影响训练本身，`raw_mAP` 可以直接和 §3、§7 的 raw 对照比较。

| 运行 | 额外改动 | 目的 |
|---|---|---|
| `prcc-cos-lr2e6-e20-lld09-ema997-s0` | `image_encoder_layer_decay=0.9` | 底层学得慢、顶层照常，看能否推迟学透 |
| `prcc-cos-lr2e6-e20-cal1-ema997-s0` | `clothes_adversarial_weight=1.0`（对抗项从 Stage-2 第 2 个 epoch 起） | 直接压制特征里的衣服信息，针对换衣主指标 |

LLRD 中间结果（对照为 §3 的 `prcc-cos-lr2e6-e20-s0` raw）：

| Stage-2 epoch | EMA mAP | raw_mAP | `triplet_active_fraction` | 对照 raw | 对照 `triplet_active_fraction` |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4509 | 0.5335 | 1.0000 | 0.6689 | 1.0000 |
| 2 | 0.5708 | 0.6967 | 0.9999 | 0.8077 | – |
| 3 | 0.6948 | 0.7984 | 0.9848 | 0.8552 | 0.8338 |
| 4 | 0.7814 | 0.8553 | 0.8178 | **0.8847** | 0.4341 |
| 5 | 0.8311 | **0.8866** | 0.5447 | 0.8778 | 0.1856 |

CAL 实现见 `t2c_reid/clothes.py` 与 `DESIGN.md` §6.2.1：衣服标签为 `(pid, A/B 或 C)`，
余弦判别器 scale 16，ε = 0.1，与 Simple-CCReID 官方实现一致（有单元测试逐项对照）。

CAL 第一次启动作废：判别器权重和其他新增参数共用 `new` 组，受 Stage-2 warmup 缩放，第 1 个
epoch 的学习率只有 `1e-4 × 0.2 = 2e-5`（官方为常数 3.5e-4）。119 步后 `clothes_loss` 仍约 5.50
（随机水平），`clothes_accuracy` 为 0，第 2 个 epoch 起的对抗项等于在对抗一个随机判别器。
已停掉，改为独立的 `clothes` 参数组，学习率恒为 `clothes_classifier_lr=3.5e-4`，不受调度影响，
然后从 Stage-1 重跑。

## 9. 产物位置

远端机器：`ssh -p 29610 root@connect.weste.seetacloud.com`（2026-09-27 起；数据盘从旧机器迁移）。

- 日志：`/root/autodl-tmp/logs/<run>.log`。
- 权重：`/root/autodl-tmp/T2C-ReID/checkpoints/prcc-cos-lr2e6-e20-cb-gray02-s0/best.pth`
  （Stage-2 第 4 个 epoch，holdout mAP 0.8721）；`prcc-cos-lr2e6-e20-ema997-s0/best.pth`
  （Stage-2 第 6 个 epoch，EMA mAP 0.8612）与 `last.pth`（第 11 个 epoch）。其他运行的
  checkpoint 已清空。
- 单个 Stage-2 checkpoint 在全量微调时约 8G，开 EMA 后约 9.7G，加上 4.5G 的
  `stage1_last.pth`。旧机器数据盘 50G 一次只放得下一轮运行，扩容到 150G 后可以两轮并行。
