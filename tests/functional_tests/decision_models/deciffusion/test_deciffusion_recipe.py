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

"""Two-GPU acceptance of the Deciffusion decision recipe on a tiny checkpoint built from the public model code.

Real: NeMo model loading (trust_remote_code), FSDP2 over two ranks, LoRA or full fine-tuning scope, activation
checkpointing, the training loop, validation selection, native stage save and reload, export to a complete
checkpoint, and an independent Hugging Face reload of that export. Only the weights are random and tiny.

Native -> stage -> export agreement: the recipe's test probabilities after reloading the selected stage must match
probabilities computed from the exported checkpoint with plain ``transformers`` and the reference read
(``encoder(use_causal_mask=False)`` + ``diffusion_head`` at the mask), and their decisions must agree.
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
CODE_FILES = (
    "config.json",
    "configuration_nemotron_labs_diffusion.py",
    "modeling_nemotron_labs_diffusion.py",
    "modeling_ministral.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)


def _tiny_checkpoint(root: Path, seed: int) -> None:
    """Download the public code/tokenizer files only and write a tiny fp32 random checkpoint with them."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModel

    root.mkdir(parents=True)
    for name in CODE_FILES:
        path = hf_hub_download(PUBLIC_CODE_REPO, name, revision=PUBLIC_CODE_REVISION)
        (root / name).write_bytes(Path(path).read_bytes())
    config = json.loads((root / "config.json").read_text())
    config.update(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        torch_dtype="float32",
        deciffusion_shared_context=True,
    )
    (root / "config.json").write_text(json.dumps(config, indent=2))
    torch.manual_seed(seed)
    model = AutoModel.from_config(AutoConfig.from_pretrained(root, trust_remote_code=True), trust_remote_code=True)
    model.save_pretrained(root, safe_serialization=True)
    saved = json.loads((root / "config.json").read_text())
    saved["deciffusion_shared_context"] = True  # keep the serving flag the export must remove
    (root / "config.json").write_text(json.dumps(saved, indent=2))


def _rows(root: Path) -> None:
    from nemo_automodel.components.models.deciffusion.rows import convert_simple_row, write_jsonl

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
    rows = [convert_simple_row(line, n) for n, line in enumerate(lines)]
    write_jsonl(root / "train.jsonl", rows[:96])
    write_jsonl(root / "validation.jsonl", rows[96:120])
    write_jsonl(root / "test.jsonl", rows[120:])


def _reference_probabilities(export_dir: Path, test_rows: Path, device: torch.device) -> dict[str, list[float]]:
    """Probabilities from the exported checkpoint with plain transformers and the reference bidirectional read."""
    from transformers import AutoModel, PreTrainedTokenizerFast

    from nemo_automodel.components.models.deciffusion.rows import encode_row, read_jsonl

    tokenizer = PreTrainedTokenizerFast.from_pretrained(export_dir)
    model = AutoModel.from_pretrained(export_dir, trust_remote_code=True, dtype=torch.float32).to(device).eval()
    out = {}
    with torch.no_grad():
        for row in read_jsonl(test_rows):
            example = encode_row(tokenizer, row, max_tokens=8192)
            ids = torch.tensor([[*example.prompt.input_ids, model.mask_token_id]], device=device)
            hidden = model.encoder(
                input_ids=ids, use_cache=False, use_causal_mask=False, is_training=False
            ).last_hidden_state
            logits = model.diffusion_head(hidden[:, -1]).float()[0][list(example.prompt.code_token_ids)]
            out[example.row_id] = logits.softmax(-1).tolist()
    return out


def _check_selected_stage_and_export(root, run, method, selected, native, repository) -> None:
    """Rank-0 checks: selected stage, finite losses, export, reference reload agreement and wrong-base rejection."""
    from nemo_automodel.components.checkpoint.utils import read_checkpoint_pointer

    best = read_checkpoint_pointer(run, "LOWEST_VAL")
    assert best is not None and Path(best).name == f"step_{selected['step']:08d}"
    assert method == "lora" or selected["step"] > 0, "full fine-tuning selects among trained stages"
    stage = read_checkpoint_pointer(run, "LATEST")
    assert stage is not None and Path(stage).name == "step_00000003"
    if method == "lora":
        from safetensors.torch import load_file

        adapter = load_file(str(Path(stage) / "model" / "adapter_model.safetensors"))
        assert any(k.endswith("lora_B.weight") and v.abs().sum() > 0 for k, v in adapter.items())
    losses = [json.loads(line)["loss"] for line in (run / "training.jsonl").read_text().splitlines()]
    assert len(losses) == 3 and all(torch.isfinite(torch.tensor(losses)))
    out = root / f"export_{method}"
    subprocess.run(
        [
            sys.executable,
            str(repository / "examples/decision/export_deciffusion.py"),
            "--base",
            str(root / "base"),
            "--stage",
            str(stage),
            "--output",
            str(out),
        ],
        check=True,
    )
    assert "deciffusion_shared_context" not in json.loads((out / "config.json").read_text())
    reference = _reference_probabilities(out, root / "data/test.jsonl", torch.device("cuda", 0))
    assert reference.keys() == native.keys()
    for key, probabilities in native.items():
        torch.testing.assert_close(torch.tensor(probabilities), torch.tensor(reference[key]), atol=2e-5, rtol=1e-4)
        assert max(range(len(probabilities)), key=probabilities.__getitem__) == max(
            range(len(probabilities)), key=reference[key].__getitem__
        )
    wrong = subprocess.run(
        [
            sys.executable,
            str(repository / "examples/decision/export_deciffusion.py"),
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
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes.decision.train_deciffusion import TrainDeciffusionDecisionRecipe

    repository = Path(__file__).resolve().parents[4]
    cfg = yaml.safe_load((repository / "examples/decision/deciffusion_support_routing_lora.yaml").read_text())
    cfg["model"]["pretrained_model_name_or_path"] = str(root / "base")
    cfg["model"]["torch_dtype"] = "float32"
    cfg["output_dir"] = str(root / f"run_{method}")
    cfg["data"].update({s: str(root / f"data/{s}.jsonl") for s in ("train", "validation", "test")})
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
    recipe = TrainDeciffusionDecisionRecipe(ConfigNode(cfg))
    recipe.setup()
    assert recipe._get_dp_group_size() == 2
    trainable = {name for name, p in recipe.model.named_parameters() if p.requires_grad}
    if method == "lora":
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
    reloaded = TrainDeciffusionDecisionRecipe(ConfigNode(cfg_eval))
    reloaded.setup()
    reloaded.run_train_validation_loop()
    metrics = json.loads((root / f"eval_{method}" / "metrics_test.json").read_text())
    native = {p["id"]: p["probabilities"] for p in metrics["predictions"]}
    failure = None
    if dist.get_rank() == 0:
        try:
            _check_selected_stage_and_export(root, run, method, selected, native, repository)
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
    parser.add_argument("--method", choices=["lora", "full"], required=True)
    args = parser.parse_args()
    _worker(Path(args.root), args.method)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two assigned CUDA GPUs for native FSDP2 setup")
@pytest.mark.parametrize("method", ["lora", "full"])
def test_deciffusion_recipe_two_gpu(tmp_path, method):
    _tiny_checkpoint(tmp_path / "base", seed=0)
    _tiny_checkpoint(tmp_path / "other", seed=1)
    _rows(tmp_path / "data")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[4])}
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
