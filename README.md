# T2C-ReID

Train2Central ReID is a research codebase for Image-to-Image person ReID.
The public project name is `T2C-ReID` and the Python package is `t2c_reid`;
the foundation vision-language model is **SigLIP 2 So400m**.

The implementation uses a two-stage training pipeline:

- `google/siglip2-so400m-patch14-384` image and text towers.
- Market-1501 and MSMT17 person/camera parsing.
- Global, camera, and training-identity learnable prompts in the SigLIP 2 text
  token embedding space.
- Stage-1 supervised SigLIP alignment between image features and identity-aware
  prompt features.
- Stage-2 ReID identity, batch-hard triplet, SigLIP all-identity alignment, and
  camera-aware cross-modal TFC losses with per-identity/per-camera prototypes.
- Fused or image-only no-rerank cosine retrieval.
- Stage-aware Weights & Biases tracking and resumable checkpoints.

The architecture contract is maintained in [DESIGN.md](DESIGN.md). Research
hypotheses and experiment tables are separate in
[docs/research-blueprint.md](docs/research-blueprint.md).

## Retrieval Contract

The image and text towers produce features in the same SigLIP 2 output space:

```text
f_v_raw = SigLIP2_ImageEncoder(image)
f_v     = normalize(FeatureHead(f_v_raw))   # Identity or BNNeck, then L2
f_t     = normalize(SigLIP2_TextEncoder(prompt))
f       = normalize(f_v + beta * f_t)
```

Both fusion inputs are L2-normalized, so `beta` controls their relative
directional contribution independently of feature norms. Triplet distance
computation and hard mining run in FP32 even with BF16/FP16 autocast.
PK sampling requires every training identity to have at least `num_instances`
images; insufficient identities raise an error instead of being silently dropped.

Training identity prompts are never used for query/gallery retrieval:

```text
training prompt  = global + camera + identity
inference prompt = global + camera
```

`--retrieval-mode image_only` returns normalized `FeatureHead(f_v_raw)` without
the text branch. `--retrieval-mode fused` uses the inference prompt above.

## Environment

The project uses `uv` and builds a mandatory Rust extension with `maturin`:

```bash
uv sync
uv run python -c "from t2c_reid.native import NATIVE_VERSION; print(NATIVE_VERSION)"
uv run python -m unittest discover -s tests
```

Install stable Rust `1.85+` before `uv sync`. Windows x86_64 requires the
MSVC Rust target and Visual Studio Build Tools; Linux x86_64 requires a C
linker. The extension is built as the private CPython module
`t2c_reid._native`. There is no automatic Python fallback when that module is
missing or has an incompatible ABI.

Core requirements are declared in `pyproject.toml`:

- Python 3.14+
- Rust 1.85+ and maturin 1.14+
- NumPy 2.4+
- PyTorch 2.13+
- torchvision 0.28+
- Transformers 5.14.1+
- Hydra 1.4 development line (required for Python 3.14)
- Weights & Biases 0.28+

The first real run downloads the selected Hugging Face checkpoint, which is
several GB for So400m.

## Data

Supported datasets:

- `market1501`
- `msmt17`

Market-1501 expects the standard directories:

```text
Market-1501-v15.09.15/
  bounding_box_train/
  query/
  bounding_box_test/
```

MSMT17 expects the standard manifests and `train` / `test` image trees. The
training split combines `list_train.txt` and `list_val.txt`.

## Training

With MSMT17 placed at `data/MSMT17_V1`, start the complete default recipe with:

```bash
uv run train
```

The default run uses Stage-1 for 60 epochs, Stage-2 for 60 epochs, validates
every 5 Stage-2 epochs, and writes to `checkpoints/msmt17-siglip2-tfc`.
Only specify values that differ from the baseline recipe. For example:

```bash
uv run train \
  data_root=D:/datasets/MSMT17_V1 \
  tfc_weight=0.5 \
  alignment_weight=0.25 \
  run_name=msmt17-tfc-weight-sweep
```

Training parameters are composed by Hydra from
`t2c_reid/configs/training/train.yaml`. Dataset-specific paths and run names live in
the `dataset` config group:

```bash
uv run train dataset=market1501
uv run train --cfg job
uv run train --help
```

Hydra overrides use `snake_case=value`; the former argparse-style
`--kebab-case value` options are not supported. Hydra keeps the process working
directory unchanged and writes its resolved config and override provenance
under the ignored `runs/hydra/` tree.

The defaults select MSMT17 at `data/MSMT17_V1`, the fixed model, `392x196`
input, Stage-1 `60`, Stage-2 `60`, batch `64` with `4` instances per identity,
accumulation `1`, eval batch `128`, cosine learning rates with a `5`-epoch
warmup in both stages, BNNeck, automatic precision, and gradient checkpointing.
`job_builder` remains available for test fixtures or external jobs but is not
required for normal T2C-ReID training.

### Why The Batch Is 64 And Accumulation Is 1

`batch_size` is the real triplet and SigLIP mining scope, and gradient
accumulation does not widen it. At `batch_size=8 num_instances=2` the PK
sampler yields `P=4` identities with `K=2` instances, so every batch-hard
triplet anchor has exactly **one** positive and six negatives — the mining
degenerates into "take the only positive". The default is now the standard ReID
`P=16 x K=4`. MSMT17's rarest training identity has 6 images, so `K=4` drops no
identities; on Market-1501, 15 of 751 identities have fewer than 4 images and
are skipped by `IdentityBalancedBatchSampler`.

Since `len(sampler) == len(labels) // batch_size`, a larger batch means
proportionally fewer iterations over the same images per epoch, so epoch cost is
roughly unchanged while GEMM efficiency improves.

### Stage 1

Stage-1 trains identity-aware prompts with the supervised SigLIP objective.
Every image/text pair sharing a person ID is positive; all other pairs in the
PK micro-batch are negative. The image tower is frozen by default.

`stage1_feature_cache=true` is the default. It extracts the frozen
train-split image features once through the eval transform and reuses them for
all Stage-1 epochs. The configuration is rejected when
`freeze_image_encoder_stage1=false`.

### Stage 2

Stage-2 defaults to an unfrozen vision tower and frozen text tower. Each image
is aligned against the camera-agnostic text anchors of **all** training
identities. The total loss is:

```text
L_total = L_id + L_triplet
        + alignment_weight * L_siglip
        + tfc_weight * L_TFC
```

`label_smoothing` applies only to `L_id`. The SigLIP loss keeps its native
binary targets and native row-mean / column-sum reduction.

`alignment_weight` must be read against the frozen pretrained calibration
`t = exp(logit_scale) = 109.89`, `b = -15.93`. At `cos = 0` the positive anchor
alone contributes `-logsigmoid(b) = 15.93` to the loss and a feature gradient of
`t / ||f_v||`, several times the combined ID and triplet gradient, while the
1040 negative anchors are saturated and contribute almost nothing. The default
`0.1` puts alignment at roughly a fifth of the ReID signal; retune it whenever
the training identity count or the feature scale changes.

Camera-aware TFC maintains FP32 visual/text EMA centers for every observed
`(person_id, camera_id)`. Camera-local centers are aggregated with equal camera
weight into global identity centers, then fused with the current Stage-2 beta:

```text
P_local[y,c] = normalize(V_local[y,c] + beta * T_local[y,c])
P_global[y]  = normalize(V_global[y]  + beta * T_global[y])

L_TFC = weighted_mean(L_local, L_global, L_cross_modal, L_cross_camera)
      + transfer_reg_weight * KL(camera_prior || camera_transfer)
```

`L_cross_camera` is multi-positive InfoNCE over initialized prototypes: the
same identity in other cameras is positive and every different identity is
negative. A learned directed `C x C` row-stochastic transfer matrix weights the
available positive cameras. The text center teacher exactly encodes
`global + camera + identity` under `no_grad`; identity prompts remain absent
from query/gallery retrieval.

Long-tail identities use higher EMA momentum between `tfc_momentum` and
`tfc_tail_momentum`. Effective-number class weights are applied to every
sample-level TFC component. Set `tfc_weight=0` to skip the teacher text
forward, all center updates, and all TFC losses.

The pretrained `logit_scale` and `logit_bias` are frozen constants:

```text
logits = exp(logit_scale) * cosine(image, text) + logit_bias
L_siglip = -mean_rows(sum_columns(logsigmoid(sign_target * logits)))
```

### Image Input

The default whole-person input is `392x196`, a 2:1 aspect ratio. With patch14,
this is `28x14 = 392` patches. Images remain BCHW tensors through the dataset
and augmentation pipeline. Hugging Face publishes the selected fixed-resolution
SigLIP 2 checkpoint with `model_type=siglip`; the adapter therefore calls its
official BCHW vision path with positional interpolation enabled. The model's
stride-14 convolution produces the 392 tokens.

The default dimensions are exactly patch-aligned. The fixed Transformers model
also retains its official valid-stride behavior for sizes such as `384x384`;
patch counts use floor division and must not exceed the checkpoint's positional
budget. The checkpoint model type, processor pretraining size, patch settings,
and positional embeddings are validated at startup. The training builder
rejects every other model ID, including NaFlex variants.

### Memory Semantics

`batch_size` is the actual SigLIP pairwise and triplet-mining micro-batch.
Gradient accumulation does not enlarge either mining scope, which is why the
default recipe puts the whole batch in one micro-batch:

```text
micro-batch: 64            (P=16 identities x K=4 instances)
accumulation steps: 1
effective optimizer batch: 64
alignment/triplet scope: 64
```

Approximate Stage-2 peak on the default recipe: about 13 GB static (all
parameters in FP32, plus gradients and AdamW moments for the 428M-parameter
vision tower, the autocast BF16 weight cache, and the TFC prototype buffers) and
about 40 MB of activations per image with gradient checkpointing on — roughly
16 GB at batch 64. Do not set `gradient_checkpointing=false`: without it the
per-image activation cost is over 400 MB.

`precision=auto` resolves as follows:

- CUDA with BF16 support: `bf16`
- other CUDA: `fp16` with `torch.amp.GradScaler`
- CPU: `fp32`

Explicit unsupported low precision fails at startup instead of silently
falling back. FP16 scaler state is saved and restored with the checkpoint.

## Main Hydra Parameters

Run and data defaults:

- `uv run train`
- `dataset=msmt17` (config group: `msmt17|market1501`)
- `data_root=data/MSMT17_V1`
- `stage1_epochs=60`
- `epochs=60` (Stage-2 epochs)
- `validation_interval=5`
- `checkpoint_dir=checkpoints/msmt17-siglip2-tfc`
- `job_builder=t2c_reid.jobs.siglip2_reid:build_training_job`

Backbone and input:

- `siglip2_model_name=google/siglip2-so400m-patch14-384` (the only accepted value)
- `siglip2_checkpoint=null` (optional strict additional state dict)
- `image_height=392`
- `image_width=196`
- `sie_coe=0.0`
- `context_length=4`

Batching and memory:

- `batch_size=64`
- `eval_batch_size=128`
- `num_instances=4`
- `gradient_accumulation_steps=1`
- `num_workers=8`
- `data_backend=rust|python` (default `rust`; Python is a reference backend)
- `prefetch_factor=2`
- `pin_memory=null|true|false` (`null` enables it automatically on CUDA)
- `persistent_workers=null|true|false` (`null` enables it when workers exist)
- `rust_data_threads=2`
- `evaluation_backend=rust|python` (default `rust`)
- `evaluation_chunk_size=256`
- `precision=auto|fp32|bf16|fp16`
- `gradient_checkpointing=true|false`

Optimization:

- `lr=1e-4`
- `image_encoder_lr=5e-6`
- `grad_clip_norm=5.0` (0 disables)
- `alignment_weight=0.1`
- `tfc_weight=1.0`
- `tfc_momentum=0.5` (highest-frequency identity)
- `tfc_tail_momentum=0.9` (lowest-frequency identity)
- `tfc_class_balance_beta=0.9999`
- `tfc_local_weight=1.0`
- `tfc_global_weight=1.0`
- `tfc_cross_modal_weight=0.5`
- `tfc_cross_camera_weight=0.1`
- `tfc_contrast_temperature=0.07`
- `tfc_transfer_reg_weight=0.01`
- `triplet_margin=0.3`
- `triplet_metric=euclidean|cosine`
- `label_smoothing=0.1`
- `stage1_lr_scheduler=none|cosine` (default `cosine`)
- `stage1_warmup_epochs=5`
- `stage2_lr_scheduler=none|cosine` (default `cosine`)
- `stage2_warmup_epochs=5`
- `beta=0.1`
- `beta_warmup_epochs=0`

Freezing and retrieval:

- `freeze_image_encoder_stage1=true|false`
- `freeze_image_encoder_stage2=true|false`
- `freeze_text_encoder=true|false`
- `freeze_prompt_bank_stage2=true|false`
- `reid_head=linear|bnneck` (default `bnneck`)
- `retrieval_mode=fused|image_only`
- `report_rerank=true|false`
- `flip_tta=true|false` (off by default; averaging the mirrored view deviates
  from the protocol of the published baselines this project compares against,
  so it must be an explicit and disclosed choice)

## Checkpoints And Resume

Stage-1 writes `stage1_last.pth`. Stage-2 writes `last.pth` and `best.pth`.
New checkpoints use schema version 3 and include:

- `backbone_family=siglip2`, dataset, Hugging Face model ID, and feature dimension
- training identity/camera counts and a deterministic pid-camera count fingerprint
- Camera-aware TFC version and every momentum, weight, temperature, class-balance,
  beta schedule, and Stage-2 epoch-offset setting
- visual/text local and global prototypes, initialized masks, statistics, and camera transfer logits
- image size, patch size, patch count, maximum patch budget, and vision input format
- tokenizer padding/pooling layout
- resolved precision
- model, optimizer, and FP16 scaler state

Resume a Stage-2 run with the same architecture and precision:

```bash
uv run train resume=checkpoints/msmt17-siglip2-tfc/last.pth
```

This migration intentionally rejects schema 2 Stage-2 checkpoints, OpenAI CLIP
weights, and old T2C-CLIP training checkpoints. The removed config fields
`clip_model_name`, `clip_checkpoint`, and `clip_weight` are rejected by Hydra's
structured schema. Incompatible resume metadata fails before model weights are
loaded.

## Weights & Biases

Enable online tracking:

```bash
uv run wandb login
uv run train \
  enable_wandb=true \
  wandb_project=T2C-ReID \
  run_name=msmt17-siglip2-camera-tfc
```

Training metrics include:

- Stage-1: `loss`, `alignment_loss`, `lr`
- Stage-2: `loss`, `alignment_loss`, `reid_loss`, `triplet_loss`, `tfc_loss`,
  `tfc_local_loss`, `tfc_global_loss`, `tfc_cross_modal_loss`,
  `tfc_cross_camera_loss`, `tfc_transfer_reg_loss`,
  `tfc_cross_camera_coverage`, `lr`
- Validation: `mAP`, `best_mAP`, `rank_1`, `rank_5`, `rank_10`

`stage1_train_step` and `stage2_train_step` count successful optimizer update
windows, not micro-batches. Window metrics are means across their constituent
micro-batches; epoch metrics average all micro-batches.

## Feature Evaluation CLI

Evaluate pre-extracted query/gallery features from `.npz`:

```bash
uv run python -m t2c_reid.cli.evaluate \
  features=path/to/features.npz \
  output=metrics.json \
  ranks=[1,5,10]
```

Evaluation parameters come from
`t2c_reid/configs/evaluation/evaluate.yaml`.
The evaluator applies the standard Image-to-Image ReID protocol and excludes
same-identity, same-camera gallery samples. Rust evaluates exact cosine scores
in query chunks and aggregates deterministic mAP/CMC without retaining the
complete `Q x G` score matrix. Primary metrics remain no-rerank.

Add exact sparse k-reciprocal metrics without replacing the primary result:

```bash
uv run python -m t2c_reid.cli.evaluate \
  features=path/to/features.npz \
  report_rerank=true \
  rerank_k1=20 \
  rerank_k2=6 \
  rerank_lambda=0.3
```

Sparse rerank keeps exact all-sample neighbor search and therefore still has
`O(N^2 D)` compute. After reciprocal edges are known, a second chunked Torch
pass extracts their exact affinity distances; resident affinity and Jaccard
structures remain sparse. Rerank distances within `1e-6` are treated as ties
and ordered by gallery index in both backends.

## Native Data Pipeline

The default training loader sends path/ID records to a batch collator. Rust
reads JPEG/PNG images, converts to RGB, applies flip, randomized ColorJitter,
bilinear resize, padded crop, SigLIP normalization, and normalized-space random
erasing, then transfers an owned contiguous `BCHW float32` allocation to NumPy
and `torch.from_numpy` without copying the element buffer.

A fixed training `seed` is repeatable within the same Rust pipeline version.
The Rust augmentation parameters and operation order match the torchvision
pipeline, but stochastic pixel values and random-number sequences are not
bitwise compatible with the Python backend. Eval resize regression fixtures
differ by at most one 8-bit quantization step after normalization.

## Performance Benchmark

Run the self-contained synthetic benchmark:

```bash
uv run python -m t2c_reid.cli.benchmark_native \
  mode=all \
  runs=5 \
  warmup_runs=1 \
  output=output/native-benchmark.json
```

Benchmark parameters come from
`t2c_reid/configs/benchmark/benchmark.yaml`. Use real training images by adding
`dataset=market1501|msmt17 data_root=PATH`. The benchmark defaults to two Rust
threads per data worker because that
was the first configuration to clear the synthetic throughput gate; production
training uses `rust_data_threads=2` by default and should be
tuned against available CPU cores. The JSON reports median/p95 duration, throughput, sampled RSS, backend
speedup, metric parity, and the acceptance gates: data `1.5x`, primary
evaluation `3x`, rerank `2x`, and rerank RSS ratio `<=0.4`. Benchmark gates are
not automated assertions because timing is hardware-sensitive. The recorded synthetic
Windows run and its real-dataset follow-up command are in
[docs/native-performance.md](docs/native-performance.md).

## Verification

Run the offline suite and inspect the CLI:

```bash
uv run cargo test --manifest-path rust/Cargo.toml --locked
uv run python -m unittest discover -s tests
uv run python -m compileall -q t2c_reid scripts tests
uv run train --help
uv run python -m t2c_reid.cli.evaluate --help
uv run python -m t2c_reid.cli.benchmark_native --help
```

The suite uses tiny randomly initialized fixed Transformers SigLIP models,
focused fakes, and a low-level H-W-C patch-order conformance test. It validates
official text/vision forward equivalence, SIE, native sigmoid losses,
accumulation, mixed precision, checkpoint compatibility, two-stage training,
prompt isolation, retrieval, and W&B behavior without downloading So400m.
