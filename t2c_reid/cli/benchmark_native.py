"""Benchmark the Rust data and evaluation backends against Python references."""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Self

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader

from t2c_reid.configuration import (
    BenchmarkConfig,
    benchmark_config_from_dict_config,
    compose_benchmark_config,
)
from t2c_reid.data import (
    ReIDSample,
    load_market_split,
    load_msmt17_manifest,
    load_prcc_split,
)
from t2c_reid.datasets import (
    ReIDImageDataset,
    ReIDImageDatasetConfig,
    ReIDMetadataDataset,
    ReIDMetadataDatasetConfig,
    RustReIDBatchCollator,
    build_camera_id_map,
    build_person_id_map,
    collate_reid_batches,
)
from t2c_reid.evaluation import RerankConfig, evaluate_reid, evaluate_reid_with_rerank
from t2c_reid.native import NATIVE_VERSION
from t2c_reid.transforms import Siglip2TrainImageTransform

DATA_SPEEDUP_TARGET = 1.5
EVALUATION_SPEEDUP_TARGET = 3.0
RERANK_SPEEDUP_TARGET = 2.0
RERANK_RSS_RATIO_TARGET = 0.4


class _Processor:
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.5, 0.5, 0.5)


class _RssSampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._baseline = _current_rss_bytes()
        self._peak = self._baseline
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, exc_type, *_args: object) -> None:
        self._stop.set()
        self._thread.join()
        if self._error is not None and exc_type is None:
            raise RuntimeError(
                "RSS sampling failed in the background thread"
            ) from self._error
        self._peak = max(self._peak, _current_rss_bytes())

    @property
    def peak_delta_bytes(self) -> int:
        return max(0, self._peak - self._baseline)

    def _sample(self) -> None:
        try:
            while not self._stop.wait(0.005):
                self._peak = max(self._peak, _current_rss_bytes())
        except Exception as exc:  # noqa: BLE001 - propagate thread failures on exit
            self._error = exc
            self._stop.set()


def main(overrides: Sequence[str] | None = None) -> int | None:
    if overrides is None:
        return _hydra_main()
    return _run_benchmark(compose_benchmark_config(overrides))


@hydra.main(config_path="../configs/benchmark", config_name="benchmark")
def _hydra_main(config: DictConfig) -> int:
    return _run_benchmark(benchmark_config_from_dict_config(config))


def _run_benchmark(config: BenchmarkConfig) -> int:
    if config.rss_worker_backend is not None:
        return _run_rerank_rss_worker(config)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    payload: dict[str, Any] = {
        "settings": {
            "mode": config.mode,
            "runs": config.runs,
            "warmup_runs": config.warmup_runs,
            "seed": config.seed,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "native_version": NATIVE_VERSION,
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "git_diff_sha256": _git_diff_sha256(),
            "dataset": config.dataset,
            "data_source": "real" if config.data_root is not None else "synthetic",
            "data_root": str(config.data_root)
            if config.data_root is not None
            else None,
            "data_samples": config.data_samples,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "rust_data_threads": config.rust_data_threads,
            "image_size": [config.image_height, config.image_width],
            "query_count": config.query_count,
            "gallery_count": config.gallery_count,
            "rerank_query_count": config.rerank_query_count,
            "rerank_gallery_count": config.rerank_gallery_count,
            "feature_dim": config.feature_dim,
            "chunk_size": config.chunk_size,
        }
    }
    data_context = (
        nullcontext(
            _load_real_samples(config.dataset, config.data_root, config.data_samples)
        )
        if config.data_root is not None
        else _synthetic_samples(config.data_samples, config.seed)
    )
    with data_context as samples:
        if config.mode in {"all", "data"}:
            payload["data"] = _benchmark_data(samples, config)
    if config.mode in {"all", "evaluation"}:
        payload["evaluation"] = _benchmark_evaluation(config)
    if config.mode in {"all", "rerank"}:
        payload["rerank"] = _benchmark_rerank(config)

    text = json.dumps(payload, indent=2, sort_keys=True)
    if config.output is None:
        print(text)
    else:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(text + "\n", encoding="utf-8")
    return 0


def _benchmark_data(
    samples: list[ReIDSample], config: BenchmarkConfig
) -> dict[str, Any]:
    transform = Siglip2TrainImageTransform(
        _Processor(), image_size=(config.image_height, config.image_width)
    )
    person_map = build_person_id_map(samples)
    camera_map = build_camera_id_map(samples)
    python_dataset = ReIDImageDataset(
        ReIDImageDatasetConfig(samples, person_map, camera_map, transform)
    )
    rust_dataset = ReIDMetadataDataset(
        ReIDMetadataDatasetConfig(
            samples, person_map, camera_map, transform.native_config
        )
    )
    python_loader = _benchmark_loader(
        python_dataset,
        collate_reid_batches,
        config.batch_size,
        config.num_workers,
    )
    rust_loader = _benchmark_loader(
        rust_dataset,
        RustReIDBatchCollator(transform.native_config, config.rust_data_threads),
        config.batch_size,
        config.num_workers,
    )

    def consume(loader: DataLoader) -> int:
        count = 0
        for batch in loader:
            count += int(batch.images.shape[0])
        return count

    python_result = _measure(
        lambda: consume(python_loader), config.warmup_runs, config.runs, len(samples)
    )
    rust_result = _measure(
        lambda: consume(rust_loader), config.warmup_runs, config.runs, len(samples)
    )
    speedup = python_result["median_seconds"] / rust_result["median_seconds"]
    return {
        "python": python_result,
        "rust": rust_result,
        "speedup": speedup,
        "target": DATA_SPEEDUP_TARGET,
        "passes_target": speedup >= DATA_SPEEDUP_TARGET,
    }


def _benchmark_loader(
    dataset, collate_fn, batch_size: int, num_workers: int
) -> DataLoader:
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(dataset, **options)


def _benchmark_evaluation(config: BenchmarkConfig) -> dict[str, Any]:
    query, gallery, metadata = _feature_fixture(
        config.query_count, config.gallery_count, config.feature_dim, config.seed
    )

    def run(backend: str):
        return evaluate_reid(
            query,
            gallery,
            **metadata,
            backend=backend,
            query_chunk_size=config.chunk_size,
        )

    python_metrics = run("python")
    rust_metrics = run("rust")
    python_result = _measure(
        lambda: run("python"), config.warmup_runs, config.runs, config.query_count
    )
    rust_result = _measure(
        lambda: run("rust"), config.warmup_runs, config.runs, config.query_count
    )
    speedup = python_result["median_seconds"] / rust_result["median_seconds"]
    return {
        "python": python_result,
        "rust": rust_result,
        "speedup": speedup,
        "target": EVALUATION_SPEEDUP_TARGET,
        "passes_target": speedup >= EVALUATION_SPEEDUP_TARGET,
        "metrics_match": python_metrics == rust_metrics,
    }


def _benchmark_rerank(config: BenchmarkConfig) -> dict[str, Any]:
    query, gallery, metadata = _feature_fixture(
        config.rerank_query_count,
        config.rerank_gallery_count,
        config.feature_dim,
        config.seed + 1,
    )
    rerank_config = RerankConfig()

    def run(backend: str):
        return evaluate_reid_with_rerank(
            query,
            gallery,
            **metadata,
            config=rerank_config,
            backend=backend,
            query_chunk_size=config.chunk_size,
        )

    python_metrics = run("python")
    rust_metrics = run("rust")
    python_result = _measure(
        lambda: run("python"), config.warmup_runs, config.runs, len(query)
    )
    rust_result = _measure(
        lambda: run("rust"), config.warmup_runs, config.runs, len(query)
    )
    python_result["peak_rss_delta_bytes"] = _isolated_rerank_rss(config, "python")
    rust_result["peak_rss_delta_bytes"] = _isolated_rerank_rss(config, "rust")
    speedup = python_result["median_seconds"] / rust_result["median_seconds"]
    python_rss = max(1, int(python_result["peak_rss_delta_bytes"]))
    rss_ratio = int(rust_result["peak_rss_delta_bytes"]) / python_rss
    return {
        "python": python_result,
        "rust": rust_result,
        "speedup": speedup,
        "speedup_target": RERANK_SPEEDUP_TARGET,
        "rss_ratio": rss_ratio,
        "rss_ratio_target": RERANK_RSS_RATIO_TARGET,
        "passes_speed_target": speedup >= RERANK_SPEEDUP_TARGET,
        "passes_rss_target": rss_ratio <= RERANK_RSS_RATIO_TARGET,
        "metrics_match": (
            abs(python_metrics.map - rust_metrics.map) <= 1e-6
            and python_metrics.cmc == rust_metrics.cmc
        ),
    }


def _isolated_rerank_rss(config: BenchmarkConfig, backend: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "rss.json"
        command = [
            sys.executable,
            "-m",
            "t2c_reid.cli.benchmark_native",
            "mode=rerank",
            f"rss_worker_backend={backend}",
            f"rerank_query_count={config.rerank_query_count}",
            f"rerank_gallery_count={config.rerank_gallery_count}",
            f"feature_dim={config.feature_dim}",
            f"chunk_size={config.chunk_size}",
            f"seed={config.seed}",
            f"output={output.as_posix()}",
            f"hydra.run.dir={(Path(tmp) / 'hydra').as_posix()}",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(output.read_text(encoding="utf-8"))
        return int(payload["peak_rss_delta_bytes"])


def _run_rerank_rss_worker(config: BenchmarkConfig) -> int:
    query, gallery, metadata = _feature_fixture(
        config.rerank_query_count,
        config.rerank_gallery_count,
        config.feature_dim,
        config.seed + 1,
    )
    gc.collect()
    with _RssSampler() as sampler:
        evaluate_reid_with_rerank(
            query,
            gallery,
            **metadata,
            config=RerankConfig(),
            backend=config.rss_worker_backend,
            query_chunk_size=config.chunk_size,
        )
    payload = {"peak_rss_delta_bytes": sampler.peak_delta_bytes}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if config.output is None:
        print(text)
    else:
        config.output.write_text(text + "\n", encoding="utf-8")
    return 0


def _measure(
    operation: Callable[[], Any], warmup_runs: int, runs: int, item_count: int
) -> dict[str, float | int]:
    if runs < 1 or warmup_runs < 0:
        raise ValueError("runs must be positive and warmup_runs must be non-negative")
    for _ in range(warmup_runs):
        operation()
    durations: list[float] = []
    peak_delta = 0
    for _ in range(runs):
        with _RssSampler() as sampler:
            started = time.perf_counter()
            operation()
            durations.append(time.perf_counter() - started)
        peak_delta = max(peak_delta, sampler.peak_delta_bytes)
    median = statistics.median(durations)
    ordered = sorted(durations)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "median_seconds": median,
        "p95_seconds": p95,
        "items_per_second": item_count / median,
        "peak_rss_delta_bytes": peak_delta,
    }


def _feature_fixture(
    query_count: int,
    gallery_count: int,
    feature_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[int]]]:
    if min(query_count, gallery_count, feature_dim) < 1:
        raise ValueError("feature benchmark dimensions must be positive")
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(query_count, feature_dim, generator=generator)
    gallery = torch.randn(gallery_count, feature_dim, generator=generator)
    identity_count = max(1, min(query_count, gallery_count) // 2)
    query_ids = [index % identity_count for index in range(query_count)]
    gallery_ids = [index % identity_count for index in range(gallery_count)]
    return (
        query,
        gallery,
        {
            "query_ids": query_ids,
            "gallery_ids": gallery_ids,
            "query_cams": [0] * query_count,
            "gallery_cams": [1 + index % 2 for index in range(gallery_count)],
        },
    )


def _synthetic_samples(count: int, seed: int):
    if count < 1:
        raise ValueError("data_samples must be positive")
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    rng = np.random.default_rng(seed)
    samples: list[ReIDSample] = []
    sizes = ((96, 192), (128, 256), (196, 392), (256, 512))
    for index in range(count):
        width, height = sizes[index % len(sizes)]
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        path = root / f"{index:06d}.jpg"
        Image.fromarray(pixels).save(path, quality=90)
        samples.append(
            ReIDSample(
                path, index % max(2, count // 4), index % 4, "synthetic", "train"
            )
        )

    class _Context:
        def __enter__(self):
            return samples

        def __exit__(self, *_args: object) -> None:
            temporary.cleanup()

    return _Context()


def _load_real_samples(dataset: str, root: Path, limit: int) -> list[ReIDSample]:
    if dataset == "market1501":
        samples = load_market_split(root, "train")
    elif dataset == "prcc":
        samples = load_prcc_split(root, "train")
    else:
        samples = [
            *load_msmt17_manifest(root, "train"),
            *load_msmt17_manifest(root, "val"),
        ]
    if not samples:
        raise ValueError("real dataset benchmark found no training samples")
    return samples[:limit]


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout) if result.returncode == 0 else None


def _git_diff_sha256() -> str | None:
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        capture_output=True,
        check=False,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=False,
    )
    if diff.returncode != 0 or untracked.returncode != 0:
        return None
    digest = hashlib.sha256(diff.stdout)
    for raw_path in sorted(path for path in untracked.stdout.split(b"\0") if path):
        path = Path(os.fsdecode(raw_path))
        if not path.is_file():
            continue
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _current_rss_bytes() -> int:
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return int(counters.WorkingSetSize)
        error_code = ctypes.get_last_error()
        raise RuntimeError(
            f"GetProcessMemoryInfo failed with Windows error {error_code}"
        )
    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError("RSS measurement is unavailable on this platform") from exc


if __name__ == "__main__":
    raise SystemExit(main())
