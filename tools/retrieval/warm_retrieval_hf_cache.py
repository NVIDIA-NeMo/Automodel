# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Warm Hugging Face dataset caches for retrieval training configs.

This tool intentionally uses the original map-style retrieval dataset path
(``make_retrieval_dataset``). Run it on CPU nodes before launching GPU training
when corpus ``datasets.load_dataset(...)`` startup is expensive.

Example:

```
python tools/retrieval/warm_retrieval_hf_cache.py \
  --config examples/retrieval/bi_encoder/nemotron_vl_1b/eagle_llama_1b_gmoreira_8_nodes_image.yaml \
  --cache-dir /path/to/shared/hf_cache \
  --touch-samples 128
```
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_RETRIEVAL_DATASET_TARGET = "nemo_automodel.components.datasets.llm.make_retrieval_dataset"


def _patch_missing_torch_distribution_version() -> None:
    original_version = importlib.metadata.version

    def version(distribution_name: str) -> str:
        package_version = original_version(distribution_name)
        if distribution_name == "torch" and package_version is None:
            import torch

            package_version = torch.__version__
        return package_version

    importlib.metadata.version = version


_patch_missing_torch_distribution_version()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required to load --config. Install pyyaml.") from e

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return cfg


def _extract_dataset_config(config_path: Path) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    try:
        dataset_cfg = cfg["dataloader"]["dataset"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"Config does not contain dataloader.dataset: {config_path}") from e
    if not isinstance(dataset_cfg, dict):
        raise ValueError(f"Config dataloader.dataset must be a mapping: {config_path}")

    target = dataset_cfg.get("_target_")
    if target not in {None, _RETRIEVAL_DATASET_TARGET}:
        raise ValueError(
            "warm_retrieval_hf_cache.py only supports the original retrieval dataset target "
            f"{_RETRIEVAL_DATASET_TARGET!r}; got {target!r}. Use the original training config, "
            "not a resolved-dataset override."
        )
    if "data_dir_list" not in dataset_cfg:
        raise ValueError(f"Config dataloader.dataset does not contain data_dir_list: {config_path}")
    return dict(dataset_cfg)


def _dataset_kwargs_from_config(
    dataset_cfg: dict[str, Any],
    dataset_factory: Callable[..., Any],
    *,
    max_train_samples: int | None,
) -> dict[str, Any]:
    kwargs = {k: v for k, v in dataset_cfg.items() if k != "_target_"}
    if max_train_samples is not None:
        kwargs["max_train_samples"] = max_train_samples

    signature = inspect.signature(dataset_factory)
    allowed_keys = set(signature.parameters)
    unknown_keys = sorted(set(kwargs) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Config dataloader.dataset contains unsupported field(s) for {dataset_factory.__name__}: {unknown_keys}"
        )
    return kwargs


def _configure_hf_cache(cache_dir: str | None) -> dict[str, str | None]:
    if cache_dir is None:
        return {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
            "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
        }

    cache_path = str(Path(cache_dir).expanduser())
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    for env_name in ("HF_HOME", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        os.environ[env_name] = cache_path
    return {
        "HF_HOME": cache_path,
        "HF_DATASETS_CACHE": cache_path,
        "HUGGINGFACE_HUB_CACHE": cache_path,
        "TRANSFORMERS_CACHE": cache_path,
    }


def _touch_example(example: dict[str, Any]) -> dict[str, int]:
    doc_text = example.get("doc_text") or []
    doc_image = example.get("doc_image") or []
    stats = {"documents": len(doc_text), "images": 0}
    for image in doc_image:
        if image:
            # Accessing PIL image metadata forces lazy image decoding for HF image features.
            getattr(image, "size", None)
            stats["images"] += 1
    return stats


def warm_retrieval_hf_cache(
    config_path: str,
    *,
    cache_dir: str | None = None,
    touch_samples: int = 0,
    max_train_samples: int | None = None,
    log_every: int = 1000,
    report_path: str | None = None,
    dataset_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Build the original retrieval dataset and optionally touch transformed samples."""
    if touch_samples < 0:
        raise ValueError(f"touch_samples must be >= 0, got {touch_samples}")
    if log_every < 1:
        raise ValueError(f"log_every must be >= 1, got {log_every}")

    cache_env = _configure_hf_cache(cache_dir)
    if dataset_factory is None:
        from nemo_automodel.components.datasets.llm.retrieval_dataset import make_retrieval_dataset

        dataset_factory = make_retrieval_dataset

    dataset_cfg = _extract_dataset_config(Path(config_path))
    dataset_kwargs = _dataset_kwargs_from_config(
        dataset_cfg,
        dataset_factory,
        max_train_samples=max_train_samples,
    )

    start_time = time.perf_counter()
    logger.info("Building retrieval dataset from %s", config_path)
    logger.info("HF cache env: %s", cache_env)
    dataset = dataset_factory(**dataset_kwargs)
    build_time_s = time.perf_counter() - start_time
    dataset_len = len(dataset) if hasattr(dataset, "__len__") else None
    logger.info("Built retrieval dataset with %s examples in %.2fs", dataset_len, build_time_s)

    touched_documents = 0
    touched_images = 0
    touch_time_s = 0.0
    if touch_samples:
        if dataset_len is None:
            sample_count = touch_samples
        else:
            sample_count = min(touch_samples, dataset_len)
        logger.info("Touching %d transformed sample(s)", sample_count)
        touch_start = time.perf_counter()
        for idx in range(sample_count):
            example = dataset[idx]
            stats = _touch_example(example)
            touched_documents += stats["documents"]
            touched_images += stats["images"]
            if (idx + 1) % log_every == 0:
                logger.info("Touched %d/%d sample(s)", idx + 1, sample_count)
        touch_time_s = time.perf_counter() - touch_start
        logger.info(
            "Touched %d sample(s), %d document(s), %d image(s) in %.2fs",
            sample_count,
            touched_documents,
            touched_images,
            touch_time_s,
        )

    report = {
        "config": str(config_path),
        "dataset_length": dataset_len,
        "build_time_s": build_time_s,
        "touch_samples": touch_samples,
        "touched_documents": touched_documents,
        "touched_images": touched_images,
        "touch_time_s": touch_time_s,
        "cache_env": cache_env,
    }
    if report_path is not None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.info("Wrote cache warmup report to %s", report_file)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm HF dataset caches for an AutoModel retrieval config.")
    parser.add_argument("--config", required=True, help="AutoModel YAML config using make_retrieval_dataset")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Shared Hugging Face cache directory. Sets HF_HOME, HF_DATASETS_CACHE, HUGGINGFACE_HUB_CACHE, and "
        "TRANSFORMERS_CACHE before dataset construction.",
    )
    parser.add_argument(
        "--touch-samples",
        type=int,
        default=0,
        help="Number of transformed samples to read after dataset construction. This validates lazy retrieval "
        "transforms and can warm a small amount of filesystem/page cache, but decoded images are not persisted.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional max_train_samples override for quick smoke tests.",
    )
    parser.add_argument("--log-every", type=int, default=1000, help="Progress interval while touching samples")
    parser.add_argument("--report-path", default=None, help="Optional JSON report output path")
    return parser.parse_args()


def main() -> None:
    """Run the CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = _parse_args()
    warm_retrieval_hf_cache(
        args.config,
        cache_dir=args.cache_dir,
        touch_samples=args.touch_samples,
        max_train_samples=args.max_train_samples,
        log_every=args.log_every,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    main()
