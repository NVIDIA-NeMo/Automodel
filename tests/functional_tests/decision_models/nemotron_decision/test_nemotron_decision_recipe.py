# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Two-GPU acceptance of the decision recipe on a tiny Nemotron 3.5 Decision-layout checkpoint.

Real: NeMo model loading (native ``NemotronForDecision``), FSDP2 over two ranks, LoRA or full fine-tuning scope,
activation checkpointing, the training loop, validation selection, native stage save and reload, export to a complete
checkpoint, and an independent Hugging Face reload of that export. The tokenizer is the public one these checkpoints
ship; only the weights are random and tiny.

Native -> stage -> export agreement: the recipe's test probabilities after reloading the selected stage must match
probabilities computed from the exported checkpoint with a stock ``transformers`` ``Ministral3Model`` with causal
attention disabled plus the head at the mask, and their decisions must agree.

``packed``: LoRA with ``data.packed_read`` on multi-question records (``json-sep`` prompts). The reloaded stage's packed
test probabilities must match the stock encoder reading the exported checkpoint with the packed layout (state ->
state; question -> state + itself; positions restart after the state; one mask per question), the same stage read
one question per sequence (``data.packed_read: false``) must match the stock single read, and the exported
``config.json`` must record ``prompt_format``, ``packed_read`` and ``read_layout``.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import yaml

PUBLIC_CODE_REPO = "nvidia/Nemotron-Labs-Diffusion-8B-Base"
PUBLIC_CODE_REVISION = "59ff0ffee284112fc6ccf37493ced41a11031434"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def _tiny_checkpoint(root: Path, seed: int) -> None:
    """Write a tiny fp32 random decision checkpoint with the public tokenizer and the decision contract files."""
    from huggingface_hub import hf_hub_download
    from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

    from nemo_automodel.components.models.nemotron_decision.model import NemotronForDecision
    from tests.unit_tests.models.nemotron_decision.char_tokenizer import write_contract

    root.mkdir(parents=True)
    for name in TOKENIZER_FILES:
        path = hf_hub_download(PUBLIC_CODE_REPO, name, revision=PUBLIC_CODE_REVISION)
        (root / name).write_bytes(Path(path).read_bytes())
    config = Ministral3Config(
        vocab_size=131072,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=16384,
        mask_token_id=100,
        bos_token_id=1,
        eos_token_id=11,
        torch_dtype="float32",
    )
    torch.manual_seed(seed)
    NemotronForDecision(config).save_pretrained(root, safe_serialization=True)
    write_contract(root)


def _records(root: Path) -> None:
    from nemo_automodel.components.models.nemotron_decision.records import convert_simple_line, write_jsonl

    teams = {"billing": "Payments and refunds", "technical": "Bugs and outages", "sales": "Contracts and quotes"}
    lines = []
    for i in range(48):
        team = list(teams)[i % 3]
        lines.append(
            {
                "id": f"c{i}",
                "state": {"ticket": f"ticket {i} about {team}"},
                "question": "Which team?",
                "options": teams,
                "label": team,
                "group": f"g{i}",
                **({"soft_labels": {team: 0.8, "sales": 0.2}} if i % 4 == 0 else {}),
            }
        )
        lines.append(
            {
                "id": f"n{i}",
                "state": {"users": i},
                "question": "Page on-call?",
                "type": "noul",
                "label": i % 2 == 0,
                "group": f"g{i}",
            }
        )
        lines.append(
            {
                "id": f"s{i}",
                "state": {"level": i % 4},
                "question": "Severity?",
                "type": "score",
                "options": ["low", "medium", "high", "critical"],
                "label": i % 4,
                "group": f"g{i}",
            }
        )
    records = [convert_simple_line(line, n) for n, line in enumerate(lines)]
    write_jsonl(root / "train.jsonl", records[:96])
    write_jsonl(root / "validation.jsonl", records[96:120])
    write_jsonl(root / "test.jsonl", records[120:])


def _packed_records(root: Path) -> None:
    """Multi-question records: the three questions of a ticket share its state (one packed read each)."""
    from nemo_automodel.components.models.nemotron_decision.records import write_jsonl

    teams = {"billing": "Payments and refunds", "technical": "Bugs and outages", "sales": "Contracts and quotes"}
    records = []
    for i in range(48):
        team = list(teams)[i % 3]
        record = {
            "id": f"r{i}",
            "state": {"ticket": f"ticket {i} about {team}", "users": i, "level": i % 4},
            "questions": {
                "team": {"type": "choice", "instructions": "Which team?", "criteria": teams},
                "page": {"type": "noul", "instructions": "Page on-call?"},
                "severity": {
                    "type": "score",
                    "instructions": "Severity?",
                    "criteria": ["low", "medium", "high", "critical"],
                },
            },
            "answers": {"team": team, "page": "true" if i % 2 == 0 else "false", "severity": str(i % 4)},
            "meta": {"group": f"g{i}"},
        }
        if i % 4 == 0:
            record["soft_labels"] = {"team": {team: 0.8, "sales": 0.2}}
        records.append(record)
    write_jsonl(root / "train.jsonl", records[:32])
    write_jsonl(root / "validation.jsonl", records[32:40])
    write_jsonl(root / "test.jsonl", records[40:])


def _reference_probabilities(
    export_dir: Path, test_records: Path, device: torch.device, *, prompt_format: str = "json", packed: bool = False
) -> dict[tuple[str, str], list[float]]:
    """Probabilities from the exported checkpoint with a stock bidirectional Ministral 3 encoder and the head.

    ``packed``: each record is one read with the packed layout, built here from the layout rules (not the recipe's
    packer): the state ends at the state-end token, each question follows with its own mask, question positions
    restart after the state, and a boolean mask lets the state see the state and a question the state and itself.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig, PreTrainedTokenizerFast
    from transformers.models.ministral3.modeling_ministral3 import Ministral3Model

    from nemo_automodel.components.models.nemotron_decision.records import encode_question, read_jsonl
    from nemo_automodel.components.models.nemotron_decision.serving_prompt import DecisionContract

    tokenizer = PreTrainedTokenizerFast.from_pretrained(export_dir)
    contract = DecisionContract.from_pretrained(export_dir)
    config = AutoConfig.from_pretrained(export_dir)
    config.is_causal = False
    weights = load_file(str(export_dir / "model.safetensors"))
    encoder = Ministral3Model(config)
    for layer in encoder.layers:
        layer.self_attn.is_causal = False
    encoder.load_state_dict({k.removeprefix("encoder."): v for k, v in weights.items() if k.startswith("encoder.")})
    encoder = encoder.to(device=device, dtype=torch.float32).eval()
    head = weights["diffusion_head.weight"].to(device=device, dtype=torch.float32)
    out = {}
    with torch.no_grad():
        for record in read_jsonl(test_records):
            prompts = {
                question_id: encode_question(
                    tokenizer,
                    contract,
                    record,
                    question_id,
                    mask_token_id=config.mask_token_id,
                    max_tokens=8192,
                    prompt_format=prompt_format,
                )
                for question_id in record["questions"]
            }
            if packed:
                end = tokenizer.convert_tokens_to_ids("<SPECIAL_101>")
                state_length = list(next(iter(prompts.values())).input_ids).index(end) + 1
                ids, positions, groups, slots = [], [], [], {}
                for k, (question_id, prompt) in enumerate(prompts.items()):
                    if not ids:
                        ids += prompt.input_ids[:state_length]
                        positions += range(state_length)
                        groups += [-1] * state_length
                    question = [*prompt.input_ids[state_length:], config.mask_token_id]
                    ids += question
                    positions += range(state_length, state_length + len(question))
                    groups += [k] * len(question)
                    slots[question_id] = len(ids) - 1
                g = torch.tensor(groups, device=device)
                allowed = (g[None, :] == -1) | ((g[:, None] == g[None, :]) & (g[:, None] != -1))
                hidden = encoder(
                    input_ids=torch.tensor([ids], device=device),
                    position_ids=torch.tensor([positions], device=device),
                    attention_mask=allowed[None, None],
                    use_cache=False,
                ).last_hidden_state
                for question_id, prompt in prompts.items():
                    logits = (head[list(prompt.code_token_ids)] @ hidden[0, slots[question_id]]).float()
                    out[(record["id"], question_id)] = logits.softmax(-1).tolist()
                continue
            for question_id, prompt in prompts.items():
                ids = torch.tensor([[*prompt.input_ids, config.mask_token_id]], device=device)
                positions = torch.arange(ids.shape[1], device=device)[None]
                hidden = encoder(input_ids=ids, position_ids=positions, use_cache=False).last_hidden_state
                logits = (head[list(prompt.code_token_ids)] @ hidden[0, -1]).float()
                out[(record["id"], question_id)] = logits.softmax(-1).tolist()
    return out


def _check_selected_stage_and_export(root, run, method, selected, native, repository, single=None) -> None:
    """Rank-0 checks: selected stage, finite losses, export, reference reload agreement and wrong-base rejection."""
    from nemo_automodel.components.checkpoint.utils import read_checkpoint_pointer

    best = read_checkpoint_pointer(run, "LOWEST_VAL")
    assert best is not None and Path(best).name == f"step_{selected['step']:08d}"
    assert method != "full" or selected["step"] > 0, "full fine-tuning selects among trained stages"
    stage = read_checkpoint_pointer(run, "LATEST")
    assert stage is not None and Path(stage).name == "step_00000003"
    if method in ("lora", "packed"):
        from safetensors.torch import load_file

        adapter = load_file(str(Path(stage) / "model" / "adapter_model.safetensors"))
        assert any(k.endswith("lora_B.weight") and v.abs().sum() > 0 for k, v in adapter.items())
    losses = [json.loads(line)["loss"] for line in (run / "training.jsonl").read_text().splitlines()]
    assert len(losses) == 3 and all(torch.isfinite(torch.tensor(losses)))
    out = root / f"export_{method}"
    subprocess.run(
        [
            sys.executable,
            str(repository / "examples/decision/export_nemotron_decision.py"),
            "--base",
            str(root / "base"),
            "--stage",
            str(stage),
            "--output",
            str(out),
        ],
        check=True,
    )
    assert (out / "decision.json").read_text() == (root / "base" / "decision.json").read_text()
    exported = json.loads((out / "config.json").read_text())
    assert exported["architectures"] == ["NemotronForDecision"]
    data = root / ("data_packed" if method == "packed" else "data")
    checks = [(native, False)] + ([(single, False)] if single is not None else [])
    if method == "packed":
        base = json.loads((root / "base" / "config.json").read_text())
        assert exported == {
            **base,
            "prompt_format": "json-sep",
            "packed_read": True,
            "read_layout": exported["read_layout"],
        }
        layout = exported["read_layout"]
        assert (
            layout["packed_read"],
            layout["prompt_format"],
            layout["state_end_token_id"],
            layout["mask_token_id"],
        ) == (
            True,
            "json-sep",
            101,
            100,
        )
        assert layout["attention"] == "state->state; question->state+self; no question<->question"
        checks = [(native, True), (single, False)]
    else:
        assert exported == json.loads((root / "base" / "config.json").read_text())
    for probabilities_by_key, packed in checks:
        reference = _reference_probabilities(
            out,
            data / "test.jsonl",
            torch.device("cuda", 0),
            prompt_format="json-sep" if method == "packed" else "json",
            packed=packed,
        )
        assert reference.keys() == probabilities_by_key.keys()
        for key, probabilities in probabilities_by_key.items():
            torch.testing.assert_close(torch.tensor(probabilities), torch.tensor(reference[key]), atol=2e-5, rtol=1e-4)
            assert max(range(len(probabilities)), key=probabilities.__getitem__) == max(
                range(len(probabilities)), key=reference[key].__getitem__
            )
    wrong = subprocess.run(
        [
            sys.executable,
            str(repository / "examples/decision/export_nemotron_decision.py"),
            "--base",
            str(root / "other"),
            "--stage",
            str(stage),
            "--output",
            str(root / f"wrong_{method}"),
        ],
        capture_output=True,
        text=True,
    )
    assert wrong.returncode != 0 and "fingerprint" in wrong.stderr


def _worker(root: Path, method: str) -> None:
    from nemo_automodel.components.checkpoint.utils import read_checkpoint_pointer
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes.decision.train_nemotron_decision import TrainNemotronDecisionRecipe

    # IEEE fp32 matmuls for the native and reference reads: an fp32 MixedPrecisionPolicy does not disable TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    repository = Path(__file__).resolve().parents[4]
    cfg = yaml.safe_load((repository / "examples/decision/nemotron_decision_support_routing_lora.yaml").read_text())
    cfg["model"]["pretrained_model_name_or_path"] = str(root / "base")
    cfg["model"]["torch_dtype"] = "float32"
    cfg["output_dir"] = str(root / f"run_{method}")
    data = "data_packed" if method == "packed" else "data"
    cfg["data"].update({s: str(root / f"{data}/{s}.jsonl") for s in ("train", "validation", "test")})
    if method == "packed":
        cfg["data"].update(packed_read=True, prompt_format="json-sep")
    cfg["train"].update(global_batch_size=4, max_steps=3, val_every_steps=1)
    cfg["optimizer"]["lr"] = 1.0e-2
    # fp32 compute so native-vs-export agreement tests state, not bf16 rounding (the default policy computes in bf16).
    cfg["distributed"]["mp_policy"] = {
        "_target_": "torch.distributed.fsdp.MixedPrecisionPolicy",
        "param_dtype": "float32",
        "reduce_dtype": "float32",
        "output_dtype": "float32",
    }
    if method == "full":
        cfg["peft"] = None
        cfg["checkpoint"]["save_consolidated"] = "every"
    recipe = TrainNemotronDecisionRecipe(ConfigNode(cfg))
    recipe.setup()
    assert recipe._get_dp_group_size() == 2
    trainable = {name for name, p in recipe.model.named_parameters() if p.requires_grad}
    if method in ("lora", "packed"):
        assert trainable and all("lora_" in name for name in trainable)
    else:
        assert trainable and all(".layers." in name or "encoder.norm" in name for name in trainable)
    assert not any(p.requires_grad for p in recipe.model.get_output_embeddings().parameters())
    assert not any(p.requires_grad for p in recipe.model.get_input_embeddings().parameters())
    optimized = {id(p) for o in recipe.optimizer for g in o.param_groups for p in g["params"]}
    assert optimized == {id(p) for p in recipe.model.parameters() if p.requires_grad}
    recipe.run_train_validation_loop()
    run = root / f"run_{method}"
    selected = json.loads((run / "selected.json").read_text())
    # Native scores of the trained final stage, reloaded in a fresh recipe (save -> reload).
    del recipe
    torch.cuda.empty_cache()
    cfg_eval = {**cfg, "mode": "eval", "init_from": str(run / "LATEST"), "output_dir": str(root / f"eval_{method}")}
    reloaded = TrainNemotronDecisionRecipe(ConfigNode(cfg_eval))
    reloaded.setup()
    reloaded.run_train_validation_loop()
    metrics = json.loads((root / f"eval_{method}" / "metrics_test.json").read_text())
    assert metrics["overall"]["n"] == 24 and metrics["run"]["skipped_questions"] == 0
    predictions = (root / f"eval_{method}" / "predictions_test.jsonl").read_text().splitlines()
    native = {(p["record_id"], p["question_id"]): p["probs"] for p in map(json.loads, predictions)}
    single = None
    if method == "packed":
        # 8 states x 3 questions, one packed read per state; the same stage then reads one question per sequence.
        assert metrics["run"]["packed_read"] is True and metrics["run"]["packed_reads"] == 8
        assert metrics["overall"]["records"] == 8 and "exact_match" in metrics["overall"]
        stage = json.loads((Path(read_checkpoint_pointer(run, "LATEST")) / "stage.json").read_text())
        assert stage["prompt_format"] == "json-sep" and stage["read_layout"]["packed_read"] is True
        del reloaded
        torch.cuda.empty_cache()
        single_cfg = {**cfg_eval, "output_dir": str(root / "eval_packed_single")}
        single_cfg["data"] = {**cfg["data"], "packed_read": False}
        single_recipe = TrainNemotronDecisionRecipe(ConfigNode(single_cfg))
        single_recipe.setup()
        single_recipe.run_train_validation_loop()
        lines = (root / "eval_packed_single" / "predictions_test.jsonl").read_text().splitlines()
        single = {(p["record_id"], p["question_id"]): p["probs"] for p in map(json.loads, lines)}
    failure = None
    if dist.get_rank() == 0:
        try:
            _check_selected_stage_and_export(root, run, method, selected, native, repository, single)
        except BaseException as error:  # report on every rank instead of leaving rank 1 in a barrier
            failure = error
    failed = torch.tensor([failure is not None], device="cuda")
    dist.all_reduce(failed)
    if failure is not None:
        raise failure
    if failed.item():
        raise RuntimeError("rank 0 acceptance checks failed")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--method", choices=["lora", "full", "packed"], required=True)
    args = parser.parse_args()
    _worker(Path(args.root), args.method)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two assigned CUDA GPUs for native FSDP2 setup")
@pytest.mark.parametrize("method", ["lora", "full", "packed"])
def test_nemotron_decision_recipe_two_gpu(tmp_path, method):
    _tiny_checkpoint(tmp_path / "base", seed=0)
    _tiny_checkpoint(tmp_path / "other", seed=1)
    _records(tmp_path / "data")
    _packed_records(tmp_path / "data_packed")
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[4]),
        # Containers can force TF32 in cuBLAS regardless of the torch flags; the parity check needs IEEE fp32.
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        "NVIDIA_TF32_OVERRIDE": "0",
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            __file__,
            "--root",
            str(tmp_path),
            "--method",
            method,
        ],
        check=True,
        env=env,
        timeout=1800,
    )


if __name__ == "__main__":
    _main()
