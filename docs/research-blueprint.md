# Train2Central ReID（T2C-ReID）研究与实验蓝图

> 本文档只维护研究假设、实验表、消融方案与机制分析。当前架构、模块边界和实现验收标准以 [DESIGN.md](../DESIGN.md) 为准；可执行命令以 [README.md](../README.md) 为准。

## 1. 项目定位

Train2Central ReID（T2C-ReID）面向监控场景下的 Image-to-Image 行人重识别。检索输入始终是 query 图像和 gallery 图像，不使用自然语言 caption，不进行 Text-to-Image 检索评测，也不使用 CUHK-PEDES 等 Text-ReID 数据集。

模型基座为 SigLIP 2 So400m patch14 双流结构。训练和推理阶段都使用图像编码器与文本编码器：图像流提供主视觉特征，文本流通过 learnable prompt 产生辅助语义特征。训练与推理使用一致的最终检索特征：

```text
f_v_raw = SigLIP2_ImageEncoder(image)
f_v = normalize(FeatureHead(f_v_raw))
f_t = normalize(SigLIP2_TextEncoder(prompt))
f   = normalize(f_v + beta * f_t)
```

主创新是 Camera-aware Cross-modal Training-time Feature Centralization（TFC）：训练期在共享特征空间维护 `(identity, camera)` visual/text 双原型，等权聚合 identity 全局中心，并用可学习相机转移矩阵、多正样本跨相机对比、长尾自适应 momentum 与 class-balanced weighting 降低同一身份在跨摄像头、光照、姿态变化下的特征漂移。推理期保持一次前向，直接用融合特征 `f` 做 cosine 检索。

## 2. 明确不做的内容

T2C-ReID 不引入以下组件或评测设定：

- CCT-Proto、超图推理、原型传播、测试时图搜索。
- Text-to-Image 检索评测。
- 自然语言 caption 监督。
- CUHK-PEDES 或其他 caption 型 Text-ReID 数据集。
- 测试时 per-ID 文本 prompt，因为测试身份未见过，身份 prompt 不可得且会造成泄漏。
- 主报结果中的测试时邻域搜索、测试时优化、memory-bank 检索或 rerank。

## 3. 方法总览

### 3.1 图像流

图像流使用 `google/siglip2-so400m-patch14-384` image encoder。给定行人图像 `x`，图像编码器输出归一化视觉特征：

```text
f_v = normalize(FeatureHead(E_v(x)))
```

`f_v` 是主检索信号，也必须能独立构成 image-only baseline。

### 3.2 文本流

文本流使用 SigLIP 2 text encoder 与 learnable prompt。文本编码器在训练和推理阶段都参与特征生成。

训练阶段使用两类 prompt：

- 每身份 ID prompt：仅用于训练身份，服务于 Stage-1/Stage-2 的监督式 SigLIP 图文对齐。
- 每摄像头 cam prompt：从图像文件名解析 camera ID 后学习。标准 ReID 协议下 camera ID 在测试集可见。

推理阶段只使用对未见身份合法的 prompt：

- 全局共享 prompt。
- 每摄像头 cam prompt。

推理阶段不使用测试身份 per-ID prompt。

### 3.3 Prompt 组合

推荐的 prompt 组合为：

```text
training prompt = global prompt + cam prompt + training ID prompt
inference prompt = global prompt + cam prompt
```

训练 ID prompt 的作用是让训练身份上的图像特征与文本特征先在 SigLIP 2 共享空间中对齐。推理 prompt 刻意保持身份无关，使文本流能泛化到未见身份。

### 3.4 特征融合

主方案采用标量融合：

```text
f = normalize(f_v + beta * f_t)
```

`beta` 可设为固定标量或可学习标量。主报应采用验证后最稳定、最易解释的设置。消融必须包含 `beta = 0`，此时退化为仅图像流，是关键 sanity check。

## 4. 两阶段训练设计

### 4.1 Stage-1：Prompt 预对齐

Stage-1 在训练集身份上学习 identity-aware prompt，使图像特征和 ID prompt 生成的文本特征在 SigLIP 2 共享空间中对齐：

```text
f_v    = normalize(E_v(x_i))
f_t_id = normalize(E_t(P_id(y_i)))
```

这个阶段不是为了创建测试身份 prompt，而是初始化与正则化文本侧，使 Stage-2 从较稳定的图文对齐空间开始训练。

### 4.2 Stage-2：双流 ReID 训练

Stage-2 训练检索模型，包括图像监督、prompt 文本特征、融合特征与 TFC：

```text
f_v_raw = E_v(x_i)
f_v = normalize(FeatureHead(f_v_raw))
f_t = normalize(E_t(P_global + P_cam(c_i) [+ P_id(y_i) for training alignment]))
f   = normalize(f_v + beta * f_t)
```

最终检索分支必须与推理兼容，即可部署的融合分支始终基于 `global prompt + cam prompt`，不依赖测试时身份 prompt。训练身份 prompt 只用于训练期对齐，不进入测试身份检索。

## 5. 损失函数设计

Stage-2 总损失为：

```text
L_total = L_id + L_triplet + alignment_weight * L_siglip + tfc_weight * L_TFC
```

### 5.1 SigLIP 对齐损失

图文对齐使用预训练 SigLIP 2 的 sigmoid pairwise 目标：

```text
S_ij = exp(logit_scale) * cosine(f_v[i], f_t[j]) + logit_bias
L_siglip = -mean_i sum_j log_sigmoid(Y_ij * S_ij)
```

Stage-1 中，同 person ID 的所有 image/text 组合为正样本。Stage-2 中，每张图像与其训练身份 anchor 为正，其余所有训练身份 anchor 为负。`logit_scale` 与 `logit_bias` 保持冻结。对齐目标只能来自训练标签，不能让最终推理依赖不可得的测试身份 prompt。

### 5.2 ReID 损失

`L_reid` 使用标准行人重识别监督：

```text
L_reid = L_id + L_triplet
```

`L_id` 是训练身份分类损失，`L_triplet` 作用于图像侧 BN 前特征。融合特征继续用于最终检索和 TFC；`label_smoothing` 只作用于 `L_id`，不修改 SigLIP binary target。

### 5.3 Camera-aware Cross-modal TFC

TFC 为每个训练身份与相机维护 visual/text EMA prototype，并对该身份已初始化相机做等权聚合：

```text
V_yc <- momentum_y * V_yc + (1 - momentum_y) * mean(f_v | y,c)
T_yc <- momentum_y * T_yc + (1 - momentum_y) * mean(f_t(global+cam+id) | y,c)
V_y  = normalize(mean_c(V_yc))
T_y  = normalize(mean_c(T_yc))
P_yc = normalize(V_yc + beta * T_yc)
P_y  = normalize(V_y  + beta * T_y)
```

最低频 identity 使用更高 momentum；sample-level TFC loss 使用 effective-number class weight。TFC 包括 local/global cosine centralization、visual-to-text center alignment 与跨相机多正样本 InfoNCE。后者把同 identity 异相机 prototype 作为正样本、不同 identity prototype 作为负样本，并由可学习的有方向 `C x C` camera transfer probability 对正相机加权。数据共现 prior 通过 `KL(prior || transfer)` 正则 transfer matrix。

```text
L_TFC = weighted_components(L_local, L_global, L_cross_modal, L_cross_camera)
      + transfer_reg_weight * L_transfer_reg
```

所有 center 与 TFC 数学使用 FP32。training identity prompt 在 Stage-2 仅作为无梯度 text-center teacher；query/gallery retrieval 仍只使用 global + camera prompt。TFC 仅用于训练，推理阶段不使用中心、相机矩阵、最近邻搜索、memory bank 或图传播。

## 6. 推理协议

对每张 query 与 gallery 图像执行同一流程：

1. 按数据集标准从文件名解析 camera ID。
2. 使用 SigLIP 2 image encoder 和 feature head 得到 `f_v`。
3. 组合 `global prompt + cam prompt`。
4. 使用 SigLIP 2 text encoder 得到 `f_t`。
5. 计算 `f = normalize(f_v + beta * f_t)`。
6. 使用 query/gallery 的融合特征 `f` 计算 cosine similarity。

推理路径与 Stage-2 中的最终检索特征路径一致。被移除的只有训练期 ID prompt 与 TFC 损失。

## 7. 数据集与评测协议

### 7.1 主数据集：MSMT17

MSMT17 是主数据集，因为其摄像头、光照、场景变化更强，更适合体现监控场景鲁棒性。

主报指标：

- 无 rerank mAP。
- 无 rerank Rank-1。
- Rank-5、Rank-10 可作为辅助指标，但不替代主指标。

### 7.2 辅数据集：Market-1501

Market-1501 作为辅数据集，按标准 Image-to-Image ReID 协议评测，用于补充说明方法泛化性。

主报指标：

- 无 rerank mAP。
- 无 rerank Rank-1。

### 7.3 结果填写规则

本文档不编造实验数值。所有结果单元格只保留表结构，真实训练或可靠引用核对后再填写。

## 8. 实验表设计

### 8.1 主结果对比表

| 方法 | Backbone | 推理是否使用文本编码器 | Rerank | MSMT17 mAP | MSMT17 Rank-1 | Market mAP | Market Rank-1 |
|---|---|---:|---:|---:|---:|---:|---:|
| AGW | CNN | 否 | 否 | 真实实验或核对引用后填写 | 真实实验或核对引用后填写 | 真实实验或核对引用后填写 | 真实实验或核对引用后填写 |
| TransReID | ViT | 否 | 否 | 真实实验或核对引用后填写 | 真实实验或核对引用后填写 | 真实实验或核对引用后填写 | 真实实验或核对引用后填写 |
| CLIP-ReID | CLIP ViT | 依论文协议核对 | 否 | 真实实验或核对引用后填写 | 真实实验或核对引用后填写 | 真实实验或核对引用后填写 | 真实实验或核对引用后填写 |
| T2C-ReID | SigLIP 2 So400m/14 | 是 | 否 | 真实实验后填写 | 真实实验后填写 | 真实实验后填写 | 真实实验后填写 |

引用近期方法时必须核对协议：无 rerank 不能与 rerank 结果混报，Text-ReID 结果不能放入 Image-ReID 主结果表。

### 8.2 核心消融表

| 变体 | 图像流 | 文本流 | Cam Prompt | TFC | MSMT17 mAP | MSMT17 Rank-1 |
|---|---:|---:|---:|---:|---:|---:|
| Image-only baseline | 是 | 否 | 否 | 否 | 真实实验后填写 | 真实实验后填写 |
| Dual stream without cam prompt | 是 | 是 | 否 | 否 | 真实实验后填写 | 真实实验后填写 |
| Dual stream with cam prompt | 是 | 是 | 是 | 否 | 真实实验后填写 | 真实实验后填写 |
| Dual stream with TFC, no cam prompt | 是 | 是 | 否 | 是 | 真实实验后填写 | 真实实验后填写 |
| Full T2C-ReID | 是 | 是 | 是 | 是 | 真实实验后填写 | 真实实验后填写 |

### 8.3 TFC 消融表

| 变体 | 局部/全局中心 | Text center | Cross-cam InfoNCE | 长尾自适应 | MSMT17 mAP | MSMT17 Rank-1 |
|---|---:|---:|---:|---:|---:|---:|
| Without TFC | 否 | 否 | 否 | 否 | 真实实验后填写 | 真实实验后填写 |
| Global visual center | 仅全局 | 否 | 否 | 否 | 真实实验后填写 | 真实实验后填写 |
| Camera-aware visual TFC | 是 | 否 | 是 | 否 | 真实实验后填写 | 真实实验后填写 |
| Cross-modal TFC | 是 | 是 | 否 | 否 | 真实实验后填写 | 真实实验后填写 |
| Full adaptive TFC | 是 | 是 | 是 | 是 | 真实实验后填写 | 真实实验后填写 |

附加消融分别令 local/global/cross-modal/cross-camera 子项权重为 0，并报告 camera transfer matrix 与有效 cross-camera anchor coverage。

### 8.4 Prompt 设计消融表

| 变体 | 训练 ID Prompt | Global Prompt | Cam Prompt | 推理 ID Prompt | MSMT17 mAP | MSMT17 Rank-1 |
|---|---:|---:|---:|---:|---:|---:|
| Global only | 否 | 是 | 否 | 否 | 真实实验后填写 | 真实实验后填写 |
| Cam only | 否 | 否 | 是 | 否 | 真实实验后填写 | 真实实验后填写 |
| Global + cam | 否 | 是 | 是 | 否 | 真实实验后填写 | 真实实验后填写 |
| Stage-1 ID alignment + global + cam | 是 | 是 | 是 | 否 | 真实实验后填写 | 真实实验后填写 |

该表必须明确：训练 ID prompt 从不作为测试身份 prompt 使用。

### 8.5 融合系数 beta 消融表

| 变体 | 融合规则 | MSMT17 mAP | MSMT17 Rank-1 |
|---|---|---:|---:|
| Image-only | `f = f_v` | 真实实验后填写 | 真实实验后填写 |
| Fixed beta | `f = normalize(f_v + beta * f_t)` | 真实实验后填写 | 真实实验后填写 |
| Learnable scalar beta | `f = normalize(f_v + beta * f_t)` | 真实实验后填写 | 真实实验后填写 |

如果主方案采用可学习 `beta`，需报告训练后的 `beta` 值以增强可解释性。

### 8.6 训练阶段消融表

| 变体 | Stage-1 | Stage-2 | TFC | MSMT17 mAP | MSMT17 Rank-1 |
|---|---:|---:|---:|---:|---:|
| Stage-2 only | 否 | 是 | 否 | 真实实验后填写 | 真实实验后填写 |
| Stage-1 + Stage-2 | 是 | 是 | 否 | 真实实验后填写 | 真实实验后填写 |
| Full T2C-ReID | 是 | 是 | 是 | 真实实验后填写 | 真实实验后填写 |

## 9. 机制分析设计

核心假设是：camera-aware text prompt 为摄像头条件下的外观变化提供轻量语义校正，TFC 则在融合空间压缩身份内离散度。

建议分析：

- 使用 UMAP 或 t-SNE 展示 TFC 前后融合特征簇的紧凑性。
- 统计身份内与身份间 cosine 距离分布。
- 展示跨摄像头检索案例，重点看光照、视角、背景变化下的 query-gallery 匹配。
- 按遮挡、相似服饰、模糊、低分辨率、摄像头风格差异归类失败案例。

这些分析用于解释机制，但不能替代标准 mAP 与 Rank-1 评测。

## 10. 文档边界

本文档不是当前代码的实现规范，也不维护工程任务清单。以下内容由其他文档负责：

- 模型路径、模块边界、训练参数和实现验收标准：[DESIGN.md](../DESIGN.md)。
- 环境安装、训练、评估和实验追踪命令：[README.md](../README.md)。

本文档只保留研究范围、结果表结构、消融方案和机制分析。实验演进仍必须让失败清晰暴露，不添加静默 fallback、假成功路径、mock 指标或隐藏上限来制造可运行假象。

## 11. 最终设计决策

- 任务范围：仅 Image-to-Image 行人重识别。
- 主数据集：MSMT17。
- 辅数据集：Market-1501。
- 基座：SigLIP 2 So400m patch14-384 双流。
- 图文目标：冻结预训练 scale/bias 的监督式 SigLIP sigmoid loss。
- 推理特征：`f = normalize(f_v + beta * f_t)`。
- 推理文本 prompt：global prompt + cam prompt。
- 训练期 prompt：per-ID prompt 用于 Stage-1/identity anchor 对齐，并作为 Stage-2 TFC
  的 no-grad text-center teacher；不进入 query/gallery。
- 主 TFC：camera-aware visual/text 双中心、跨相机 InfoNCE、相机转移矩阵与长尾自适应。
- 主指标：无 rerank mAP 与 Rank-1。
- 结果表：只提供结构，不编造实验数值。
