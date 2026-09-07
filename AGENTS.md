# PROJECT KNOWLEDGE BASE

**Generated:** 2026-07-25

## OVERVIEW

Project: **T2C-ReID** (`t2c_reid`)

T2C-ReID is an Image-to-Image person re-identification research codebase. The
project name still contains CLIP, but the only supported foundation model is
`google/siglip2-so400m-patch14-384`. The system preserves a two-stage ReID
pipeline with learnable prompts, camera conditioning, SIE, TFC, optional
BNNeck, fused/image-only retrieval, mAP/CMC evaluation, W&B tracking, mixed
precision, gradient accumulation, and resumable checkpoints.

Stack:

- Python 3.14+; the current locked environment uses Python 3.14.6.
- PyTorch 2.13 and torchvision 0.28, using the CUDA 13.2 wheel index.
- Transformers 5.14.1.
- Hydra 1.4 development line with OmegaConf 2.4 for Python 3.14 support.
- Weights & Biases 0.28.1.
- `uv` for dependency, environment, and command execution.
- Rust 1.85+ with maturin 1.14, PyO3/rust-numpy 0.29, Rayon, and pure-Rust
  JPEG/PNG preprocessing. `uv sync` must build `t2c_reid._native`.
- Standard-library `unittest`; there is no pytest, formatter, linter, or static
  type checker configured in `pyproject.toml`.

## NON-NEGOTIABLE CONTRACT

- The training builder accepts only
  `google/siglip2-so400m-patch14-384`; do not add NaFlex or arbitrary model
  overrides without an explicit design change.
- The Hub checkpoint is a SigLIP 2 checkpoint registered by Transformers as
  fixed `model_type="siglip"` / `SiglipModel`, not `Siglip2Model`. Load it with
  `AutoModel`; its production vision input is BCHW with positional
  interpolation.
- Do not restore OpenAI CLIP backends, `clip_*` model APIs, old CLI flags, or
  compatibility aliases. Old CLIP checkpoints are intentionally incompatible.
- Default ReID input is `(height, width) = (392, 196)`, producing `28 x 14 =
  392` patch14 tokens. The fixed model also supports official valid-stride
  floor behavior such as `384 x 384 -> 27 x 27 = 729` tokens.
- The selected Gemma tokenizer is right padded, exposes `input_ids` only, adds
  no BOS in the prompt adapter, and pools the final position (a PAD token)
  through `text_model.head`. Do not replace this with CLIP-style causal masks,
  maximum-token-ID pooling, or EOS-last assumptions.
- Training identity prompts may be used only for Stage-1 alignment,
  training-identity anchors, and the no-grad Stage-2 TFC text-prototype teacher.
  Query/gallery retrieval may use only global + camera prompts.
- Retrieval applies `FeatureHead` (Identity or BNNeck) to the image feature
  before fusing it with text. Triplet and alignment paths intentionally use
  the documented raw/normalized visual features instead.
- `logit_scale` and `logit_bias` come from the pretrained checkpoint, remain
  frozen, and never enter an optimizer. SigLIP normalization, logits, bias,
  and `logsigmoid` must run in FP32 even inside an outer autocast context.
- Checkpoint schema 3 validates model/image/feature/text-layout/precision plus
  dataset, pid-camera fingerprint, and all TFC configuration before loading
  model state. Schema 2 Stage-2 checkpoints are intentionally incompatible.
  FP16 GradScaler state is auxiliary checkpoint state. Only Stage-2
  checkpoints may be resumed.

## STRUCTURE

```text
T2C-ReID/
|-- t2c_reid/                  Core package
|   |-- siglip2_backbone.py    Fixed SigLIP 2 image/text adapters and validation
|   |-- model.py               T2CReIDModel forward and retrieval paths
|   |-- prompts.py             Global, camera, and train-identity prompt bank
|   |-- anchors.py             Stage-2 identity text-anchor provider/cache
|   |-- losses.py              Native SigLIP and batch-hard triplet losses
|   |-- training.py            Stage-1/Stage-2 loss composition
|   |-- precision.py           Precision policy, autocast, and GradScaler
|   |-- tfc.py                 Camera-aware visual/text prototype bank and TFC losses
|   |-- transforms.py          SigLIP-normalized train/eval image transforms
|   |-- data.py                Market-1501/MSMT17 parsing
|   |-- datasets.py            Python reference dataset and Rust batch collator
|   |-- native.py              Mandatory native ABI/version check
|   |-- evaluation.py          Chunked Rust mAP/CMC and sparse rerank dispatch
|   |-- retrieval.py           Fused/image-only retrieval mode constants
|   |-- configuration.py       Typed Hydra schemas and composition helpers
|   |-- configs/               Training/evaluation/benchmark Hydra configs
|   |-- loops.py               Generic epoch loop and checkpoint writing
|   |-- wandb.py               Stage-aware W&B adapter
|   |-- jobs/siglip2_reid.py   End-to-end training job assembly and runtime
|   `-- cli/evaluate.py        NPZ feature evaluation CLI
|-- scripts/train.py           Generic single/two-stage training entrypoint
|-- rust/                      PyO3 native image/evaluation/rerank crate
|-- tests/                     Offline unittest suite and tiny/fake models
|-- README.md                  Operator-facing setup and command reference
|-- DESIGN.md                  Normative architecture and behavior contract
|-- docs/research-blueprint.md Experiment hypotheses, tables, and reporting plan
|-- pyproject.toml             Python requirements and uv package index
`-- uv.lock                    Reproducible dependency lock
```

`main.py` is a placeholder greeting, not the training or evaluation entrypoint.

## COMMANDS

Run all Python tooling through `uv`; do not use pip or conda for this project.

| Action | Command |
|---|---|
| Install/sync | `uv sync` |
| Native Rust tests | `uv run cargo test --manifest-path rust/Cargo.toml --locked` |
| Full test suite | `uv run python -m unittest discover -s tests` |
| One test module | `uv run python -m unittest tests.test_siglip2_training_components` |
| Syntax/import compile check | `uv run python -m compileall -q t2c_reid scripts tests` |
| Training CLI help | `uv run train --help` |
| Evaluation CLI help | `uv run python -m t2c_reid.cli.evaluate --help` |
| Native benchmark | `uv run python -m t2c_reid.cli.benchmark_native --help` |
| Check diff hygiene | `git diff --check` |

There is no separate application build step. The useful pre-commit checks are
the full unittest suite, compileall, CLI help, and `git diff --check`.

### Training

```bash
uv run train
```

Important defaults target a 32GB single GPU: MSMT17 at `data/MSMT17_V1`,
Stage-1 60 epochs, Stage-2 60 epochs, train batch 64 with 4 instances per
identity (`P=16 x K=4`), gradient accumulation 1, eval batch 128, cosine
learning rates with a 5-epoch warmup in both stages, BNNeck feature head,
label smoothing `0.1`, alignment weight `0.1`, gradient-norm clipping at `5.0`,
precision `auto`, gradient checkpointing enabled, image size `392 x 196`, and
image encoder LR `5e-6`. Override only the parameters needed for an experiment.

`precision=auto` resolves to BF16 on capable CUDA devices, FP16 + GradScaler on
other CUDA devices, and FP32 on CPU. Gradient accumulation does not enlarge the
pairwise SigLIP or triplet-mining scope; both operate on each micro-batch, so
`batch_size` must be the full PK batch and accumulation stays at 1 unless
memory forces otherwise.

Two defaults are calibration-sensitive and must be re-derived, not copied, if
the dataset or feature scale changes:

- `alignment_weight`: the SigLIP anchor term carries the frozen pretrained
  `t = exp(logit_scale) = 109.89` and `b = -15.93`, so at `cos = 0` its positive
  anchor contributes `15.93` to the loss and a feature gradient of `t/||f_v||`.
  The negatives are saturated and contribute almost nothing at initialization.
- `triplet_margin`: batch-hard triplet runs on **unnormalized** `visual_raw`
  with a euclidean metric, so a `0.3` margin is meaningful only while
  `||visual_raw||` is small. Measure the fraction of anchors with a non-zero
  hinge; if it is near zero the term is inert and `triplet_metric=cosine`
  (well scaled for a `[0, 2]` distance) is the right correction.

### Feature Evaluation

```bash
uv run python -m t2c_reid.cli.evaluate \
  features=path/to/features.npz \
  output=metrics.json \
  ranks=[1,5,10]
```

The NPZ must contain `query_features`, `gallery_features`, `query_ids`,
`gallery_ids`, `query_cams`, and `gallery_cams`.

## ARCHITECTURE

Retrieval flow:

```text
f_v_raw = SigLIP2_ImageEncoder(image)
f_v     = normalize(FeatureHead(f_v_raw))   # Identity or BNNeck, then L2
f_t     = normalize(SigLIP2_TextEncoder(global + camera prompt))
f       = normalize(f_v + beta * f_t)
```

Stage-1 freezes the image tower by default and learns identity-aware prompts
with a supervised all-pairs SigLIP sigmoid objective. Every image/text pair
with the same person ID is positive; all other pairs in the PK micro-batch are
negative. The Stage-1 feature cache is legal only while the image tower is
frozen.

Stage-2 unfreezes the image tower by default, keeps the text tower frozen, and
combines identity classification, batch-hard triplet, all-training-identity
SigLIP anchor alignment, and camera-aware cross-modal TFC. TFC maintains FP32
visual/text `(pid, camera)` prototypes, equal-camera global centers, adaptive
per-ID momentum, effective-number class weights, and a learned directed camera
transfer matrix for multi-positive cross-camera InfoNCE. Text anchors are
camera agnostic; the exact global+camera+identity TFC teacher is no-grad and
training-only. Camera retrieval text and identity anchors are cached only when
their prompt and text dependencies are frozen.

Primary evaluation is no-rerank cosine retrieval with standard same-ID,
same-camera gallery exclusion. Torch computes normalized score/distance blocks;
Rust performs deterministic ranking and metric aggregation. Exact sparse
k-reciprocal reranking is optional and must not replace primary mAP/CMC.

The default data backend is Rust. It batch-decodes JPEG/PNG, applies the shared
augmentation configuration, and returns contiguous BCHW FP32 through NumPy to
Torch without an element-buffer copy. The Python backend is explicit reference
only; there is no missing-extension fallback. Rust augmentation is reproducible
within its own version/seed contract but not bitwise identical to torchvision.

## CODING STANDARDS

- Use `from __future__ import annotations` in package modules that need forward
  annotations.
- Use modern Python 3.14 type syntax (`X | None`, built-in generics) and typed
  public/internal boundaries.
- Prefer small frozen dataclasses for immutable configs, batches, metrics, and
  loss breakdowns.
- Keep constants at module scope in uppercase; use `snake_case` for functions
  and variables and `PascalCase` for classes.
- Validate tensor rank, dtype, shape, index range, dimensions, and incompatible
  configuration early. Existing code uses explicit `ValueError`, `TypeError`,
  and `FileNotFoundError` messages rather than assertions in production paths.
- Keep model components injectable. Tests depend on tiny official Transformers
  models and focused fakes instead of downloading So400m weights.
- Preserve device and dtype semantics. Do not move intermediate tensors to CPU
  inside training paths except for final evaluation features/distances.
- Avoid broad refactors in `jobs/siglip2_reid.py`; it is large but owns the
  complete assembly boundary. Put reusable math/model behavior in the smaller
  package modules.
- No formatter or linter is currently authoritative. Match surrounding style,
  keep diffs focused, and rely on tests plus `git diff --check`.

## TESTING GUIDANCE

Tests use standard `unittest` classes and methods named `test_*`. Shared
SigLIP-shaped fakes live in `tests/_siglip2_fakes.py`.

Run focused tests while developing, then the complete suite before finishing:

- Backbone, processor, prompt layout, or SIE changes:
  `tests.test_siglip2_training_components` and `tests.test_siglip2_reid_job`.
- Loss or Stage wiring changes:
  `tests.test_tfc_losses` and `tests.test_training`.
- Precision, accumulation, or checkpoints:
  `tests.test_precision`, `tests.test_training_loop`,
  `tests.test_two_stage_training`, and `tests.test_train_script`.
- Retrieval/evaluation changes:
  `tests.test_evaluation_model` and `tests.test_cli_evaluate`.
- W&B changes: `tests.test_wandb_tracking`.

When testing SigLIP logits under mixed precision, enter an actual autocast
context; low-precision input tensors alone do not prove that eligible matrix
multiplications stay FP32.

## WHERE TO LOOK

- Operational setup and CLI examples: `README.md`.
- Normative architecture decisions and acceptance criteria: `DESIGN.md`.
- Research claims, ablations, and result tables: `docs/research-blueprint.md`.
- Full training assembly and defaults: `t2c_reid/jobs/siglip2_reid.py`.
- Generic training/checkpoint orchestration: `scripts/train.py` and
  `t2c_reid/loops.py`.
- Public package surface: `t2c_reid/__init__.py`.
- Behavioral specification: `tests/`.

## REPOSITORY HYGIENE

- Dataset directories, checkpoints, W&B runs, generated outputs, caches, and
  `.pi-subagents/` are ignored; do not commit them.
- A real So400m load downloads a multi-gigabyte checkpoint. Keep ordinary unit
  tests offline and use tiny initialized models for offline verification.
- Preserve project branding (`T2C-ReID`, `t2c_reid`) even though model-facing
  symbols use SigLIP 2 terminology.
- Update `README.md`, `DESIGN.md`, tests, CLI help, and checkpoint metadata when
  changing a user-visible training contract.
