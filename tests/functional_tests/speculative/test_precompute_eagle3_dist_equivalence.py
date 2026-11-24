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

"""CLI regression for EAGLE-3 replicated precompute and sparse resume on CPU."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config

from nemo_automodel.components.datasets.llm.eagle3_cache import CachedEagle3Dataset, read_manifest

pytestmark = pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="requires Gloo")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PRECOMPUTE = "nemo_automodel.components.speculative.precompute_eagle3"


def test_cli_configures_logging_when_handler_already_exists(tmp_path: Path) -> None:
    script = """
import logging
logging.warning("handler initialized before CLI")
from nemo_automodel.components.speculative import precompute_eagle3
precompute_eagle3._run = lambda args: logging.getLogger("precompute_test").info("configured info log")
precompute_eagle3.main(["--target-model", "unused", "--input-data", "unused", "--output-dir", "unused"])
"""
    output = _run_cpu([sys.executable, "-c", script], tmp_path)
    assert "configured info log" in output


@pytest.fixture(scope="module")
def tiny_qwen(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Create a local dense Qwen3 checkpoint and seven deterministic chat rows."""
    root = tmp_path_factory.mktemp("eagle3_cpu_target")
    target = root / "model"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            head_dim=8,
            intermediate_size=64,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            pad_token_id=0,
            eos_token_id=2,
            tie_word_embeddings=False,
        )
        AutoModelForCausalLM.from_config(config).save_pretrained(target)

    vocab = ["<pad>", "<unk>", "<eos>", "user", "assistant", "request"] + [f"t{i}" for i in range(58)]
    raw_tokenizer = Tokenizer(models.WordLevel(dict(zip(vocab, range(len(vocab)))), unk_token="<unk>"))
    raw_tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>"
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] + ' ' }}"
        "{% if m['role'] == 'assistant' %}{% generation %}{{ m['content'] + ' <eos> ' }}{% endgeneration %}"
        "{% else %}{{ m['content'] + ' <eos> ' }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    )
    tokenizer.save_pretrained(target)

    data = root / "messages.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"request t{i}"},
                {"role": "assistant", "content": f"t{i} t0 t1 t{i}"},
            ]
        }
        for i in range(7)
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return target, data


def _run_cpu(command: list[str], work: Path, *, expect_error: bool = False) -> str:
    env = os.environ.copy()
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        HF_HOME=str(work / "hf"),
        HF_DATASETS_CACHE=str(work / "datasets"),
        HF_MODULES_CACHE=str(work / "hf" / "modules"),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_PROGRESS_BARS="1",
        PYTHONPATH=os.pathsep.join(filter(None, (str(_REPO_ROOT), env.get("PYTHONPATH")))),
    )
    with subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name == "posix",
    ) as proc:
        try:
            output, _ = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            output, _ = proc.communicate()
            pytest.fail(f"CPU precompute did not exit within 60 seconds.\n{output}")
    if expect_error:
        assert proc.returncode != 0, output
    else:
        assert proc.returncode == 0, output
    return output


def _precompute(
    target: Path,
    data: Path,
    cache: Path,
    *,
    world_size: int = 1,
    topk: int = 4,
    extra: tuple[str, ...] = (),
    expect_error: bool = False,
) -> str:
    command = [sys.executable]
    if world_size > 1:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--local-addr=127.0.0.1",
            f"--nproc-per-node={world_size}",
        ]
    command += [
        "-m",
        _PRECOMPUTE,
        "--target-model",
        str(target),
        "--input-data",
        str(data),
        "--output-dir",
        str(cache),
        "--device",
        "cpu",
        "--dtype",
        "fp32",
        "--seq-length",
        "16",
        "--batch-size",
        "2",
        "--shard-size",
        "2",
        "--draft-vocab-size",
        "16",
        "--target-probs-topk",
        str(topk),
        "--aux-layer-ids",
        "1",
        "3",
        "5",
        *extra,
    ]
    return _run_cpu(command, cache.parent, expect_error=expect_error)


def _assert_equal_cache(reference: Path, result: Path) -> None:
    expected_manifest = read_manifest(str(reference))
    assert read_manifest(str(result)) == expected_manifest
    expected, actual = CachedEagle3Dataset(str(reference)), CachedEagle3Dataset(str(result))
    assert len(actual) == len(expected) == expected_manifest["num_samples"]
    for index in range(len(expected)):
        expected_sample, actual_sample = expected[index], actual[index]
        assert actual_sample.keys() == expected_sample.keys()
        for key, value in expected_sample.items():
            if value.is_floating_point():
                torch.testing.assert_close(actual_sample[key], value, rtol=1e-5, atol=1e-6)
            else:
                assert torch.equal(actual_sample[key], value), (index, key)
    if expected_manifest["target_probs_topk"]:
        for shard in reference.glob("shard-*.safetensors"):
            lhs, rhs = load_file(shard), load_file(result / shard.name)
            assert torch.equal(lhs["target_probs_indices"], rhs["target_probs_indices"])
            torch.testing.assert_close(lhs["target_probs_values"], rhs["target_probs_values"], rtol=1e-5, atol=1e-6)


@pytest.fixture(scope="module", params=[0, 4], ids=["full", "topk"])
def compared_caches(
    request: pytest.FixtureRequest, tiny_qwen: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path, int]:
    root = tmp_path_factory.mktemp("eagle3_precomputed")
    target, data = tiny_qwen
    topk = request.param
    reference, result = root / "single", root / "distributed"
    single_log = _precompute(target, data, reference, topk=topk)
    distributed_log = _precompute(target, data, result, world_size=2, topk=topk)
    # An ordinary DistributedSampler would pad seven rows to eight. Each row here
    # has five supervised tokens, giving a discriminating frequency-count check.
    assert "Counted 35 supervised tokens" in single_log
    assert "Counted 35 supervised tokens" in distributed_log
    return reference, result, topk


def test_cpu_data_parallel_precompute_matches_single_process(compared_caches: tuple[Path, Path, int]) -> None:
    reference, result, _ = compared_caches
    _assert_equal_cache(reference, result)


def _partial_cache(source: Path, destination: Path) -> None:
    """Keep the first committed shard from each of the two original ranks."""
    destination.mkdir()
    for name in (
        "manifest.json",
        "target_embeddings.safetensors",
        "shard-000000.safetensors",
        "shard-000002.safetensors",
    ):
        shutil.copy2(source / name, destination / name)


def _snapshot(path: Path) -> dict[str, tuple[str, int]]:
    return {f.name: (hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_mtime_ns) for f in path.iterdir()}


def test_resume_writes_missing_shards_with_a_different_world_size(
    compared_caches: tuple[Path, Path, int], tiny_qwen: tuple[Path, Path], tmp_path: Path
) -> None:
    reference, result, topk = compared_caches
    incomplete = tmp_path / "interrupted"
    _partial_cache(result, incomplete)
    (incomplete / "shard-000003.safetensors.tmp").write_bytes(b"interrupted temporary write")
    before = _snapshot(incomplete)
    log = _precompute(*tiny_qwen, incomplete, world_size=3, topk=topk, extra=("--resume",))
    assert "Counted 35 supervised tokens" in log
    after = _snapshot(incomplete)
    assert set(after) == set(before) - {"shard-000003.safetensors.tmp"} | {
        "shard-000001.safetensors",
        "shard-000003.safetensors",
    }
    assert all(after[name] == snapshot for name, snapshot in before.items() if not name.endswith(".tmp"))
    _assert_equal_cache(reference, incomplete)


def test_resume_balances_missing_shards(
    compared_caches: tuple[Path, Path, int], tiny_qwen: tuple[Path, Path], tmp_path: Path
) -> None:
    reference, result, topk = compared_caches
    incomplete = tmp_path / "completed_prefix"
    shutil.copytree(result, incomplete)
    # With two ranks, the original rank 0 owns both retained shards.
    for index in (2, 3):
        (incomplete / f"shard-{index:06d}.safetensors").unlink()
    before = _snapshot(incomplete)

    log = _precompute(*tiny_qwen, incomplete, world_size=2, topk=topk, extra=("--resume",))

    assert "Rank 0/2: precomputing samples [4, 6) from shard 2" in log
    assert "Rank 1/2: precomputing samples [6, 7) from shard 3" in log
    after = _snapshot(incomplete)
    assert all(after[name] == snapshot for name, snapshot in before.items())
    _assert_equal_cache(reference, incomplete)


def test_nonzero_rank_shard_validation_fails_before_any_write(
    compared_caches: tuple[Path, Path, int], tiny_qwen: tuple[Path, Path], tmp_path: Path
) -> None:
    _, result, topk = compared_caches
    incomplete = tmp_path / "invalid"
    _partial_cache(result, incomplete)
    bad_shard = incomplete / "shard-000002.safetensors"
    if topk:
        # A readable shard with one missing row tests schema validation too.
        shortened = {key: tensor[:1].clone() for key, tensor in load_file(bad_shard).items()}
        save_file(shortened, bad_shard)
    else:
        bad_shard.write_bytes(b"invalid header")
    before = _snapshot(incomplete)
    log = _precompute(*tiny_qwen, incomplete, world_size=2, topk=topk, extra=("--resume",), expect_error=True)
    assert "rank 1" in log and "shard-000002.safetensors" in log, log
    if topk:
        assert "expected shape" in log, log
    assert _snapshot(incomplete) == before


def test_cpu_data_parallel_precompute_with_an_idle_rank(tiny_qwen: tuple[Path, Path], tmp_path: Path) -> None:
    target, messages = tiny_qwen
    data = tmp_path / "one.jsonl"
    data.write_text(messages.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    cache = tmp_path / "cache"
    log = _precompute(target, data, cache, world_size=2, extra=("--trust-remote-code",))
    assert "Counted 5 supervised tokens" in log
    assert len(CachedEagle3Dataset(str(cache))) == 1
    assert [path.name for path in cache.glob("shard-*.safetensors")] == ["shard-000000.safetensors"]


def test_distributed_cache_drives_an_offline_optimizer_step(
    compared_caches: tuple[Path, Path, int], tiny_qwen: tuple[Path, Path], tmp_path: Path
) -> None:
    _, cache, _ = compared_caches
    target, messages = tiny_qwen
    with (_REPO_ROOT / "examples/speculative/eagle3/llama_eagle3_mvp.yaml").open() as stream:
        config = yaml.safe_load(stream)
    config["dist_env"] = {"backend": "gloo", "timeout_minutes": 1}
    config["checkpoint"].update(enabled=False, checkpoint_dir=str(tmp_path / "checkpoints"))
    config["recipe_args"].update(
        target_model_name_or_path=str(target),
        train_data_path=str(messages),
        cached_target_path=str(cache),
        output_dir=str(tmp_path / "output"),
        seq_length=16,
        draft_vocab_size=16,
        ttt_steps=2,
        num_workers=0,
        micro_batch_size=7,
        num_epochs=1,
    )
    config_file = tmp_path / "train.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    worker = """
import sys
import torch
import yaml
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes.speculative_decoding.train_eagle3 import TrainEagle3Recipe
with open(sys.argv[1]) as stream:
    recipe = TrainEagle3Recipe(ConfigNode(yaml.safe_load(stream)))
recipe.setup()
assert recipe.target_model is None
before = [(param, param.detach().clone()) for param in recipe.draft_model.parameters() if param.requires_grad]
recipe.run_train_validation_loop()
assert recipe.runtime.global_step == 1
assert all(torch.isfinite(param).all() for param, _ in before)
assert any(not torch.equal(param.detach(), initial) for param, initial in before)
"""
    _run_cpu([sys.executable, "-c", worker, str(config_file)], tmp_path)
