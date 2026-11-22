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

"""Verify pinned, released-size D1 text decisions and frozen-backbone adaptation.

Downloads occur only when this CLI is invoked. Results contain model revisions
and correctness evidence, without host, accelerator inventory or runtime paths.
The diagnostic and readout artifacts are run outputs, not contribution fixtures.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from nemo_automodel.components.datasets.decision.records import QuestionTarget, question_targets, validate_record
from nemo_automodel.components.models.d1.releases import RELEASE_DTYPES, RELEASES

PROBABILITY_ATOL = 0.02
QUESTION_IDS = {"action", "attack_tactic", "needs_human", "severity", "verdict"}
TIED_ALIAS = ("lm_head.weight", "model.language_model.embed_tokens.weight")
VISION_WRAPPER_PREFIXES = (
    ("model.vision_tower.vision_model.", "model.vision_tower."),
    ("vision.tower.vision_model.", "vision.tower."),
)


def _checkpoint_state_key(serialized_name: str, expected_keys: set[str]) -> str:
    """Resolve only the verified Siglip2 wrapper migration inside the two D1 towers.

    Transformers maps Siglip2VisionModel to its CLIPVisionModel conversion,
    whose PrefixChange removes the former vision_model wrapper. Prefer an
    exact state key when the installed reference retains the old layout.
    """
    if serialized_name in expected_keys:
        return serialized_name
    for old_prefix, new_prefix in VISION_WRAPPER_PREFIXES:
        if serialized_name.startswith(old_prefix):
            migrated = new_prefix + serialized_name[len(old_prefix) :]
            if migrated in expected_keys:
                return migrated
    raise ValueError(f"Unexpected checkpoint tensor: {serialized_name}")


def _canonical_checkpoint_keys(keys: set[str], *, tied: bool) -> set[str]:
    """Expand only the documented D1-3B input/output weight alias.

    SafeTensor serialization may omit either member of the tied pair. All
    other keys remain exact; this does not permit prefix or suffix matching.
    """
    result = set(keys)
    if tied and result.intersection(TIED_ALIAS):
        result.update(TIED_ALIAS)
    return result


def _tie_groups(model: nn.Module) -> list[list[str]]:
    """Return parameter-identity aliases, including duplicate parameter names."""
    groups: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), []).append(name)
    return sorted(sorted(names) for names in groups.values() if len(names) > 1)


def _require_tensor_equal(actual: torch.Tensor, expected: torch.Tensor, *, name: str) -> None:
    """Require exact weight coverage without dtype conversion or broadcasting.

    Args:
        actual: Tensor of arbitrary shape from the native state dictionary.
        expected: Tensor with the same shape, dtype and values from the reference.
        name: Public parameter or buffer name identifying the comparison.
    """
    if actual.is_meta or expected.is_meta:
        raise ValueError(f"Unmaterialized tensor: {name}")
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"Weight shape or dtype differs: {name}")
    if not torch.equal(actual, expected.to(actual.device)):
        raise ValueError(f"Weight values differ: {name}")


def _check_weights(native: nn.Module, publisher: nn.Module, snapshot: Path, *, tied: bool) -> dict:
    """Check every materialized native/publisher weight and serialized tensor.

    Both loaders explicitly request the same per-release storage dtype: BF16
    for D1-3B and FP32 for Omni, whose publisher warns against BF16. Raw
    serialized dtypes are reported, and comparison applies only the requested
    storage cast. Native/publisher dtype equality is checked independently.
    """
    from safetensors import safe_open

    actual, expected = native.state_dict(), publisher.state_dict()
    if set(actual) != set(expected):
        raise ValueError("Native and publisher state-dictionary keys differ")
    for name in actual:
        _require_tensor_equal(actual[name], expected[name], name=name)
    groups = _tie_groups(native)
    if groups != _tie_groups(publisher):
        raise ValueError("Native and publisher parameter identities have different ties")
    if tied and sorted(TIED_ALIAS) not in groups:
        raise ValueError("D1-3B input/output parameter identity is not tied")
    if not tied and groups:
        raise ValueError("Omni unexpectedly contains tied parameter identities")
    files = sorted(snapshot.glob("*.safetensors"))
    if not files:
        raise ValueError("Pinned checkpoint contains no SafeTensor weight files")
    serialized: set[str] = set()
    mapped_keys: set[str] = set()
    renamed_keys = []
    source_dtypes: Counter[str] = Counter()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in serialized:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                serialized.add(name)
                state_name = _checkpoint_state_key(name, set(expected))
                if state_name in mapped_keys:
                    raise ValueError(f"Checkpoint mapping collision: {name} maps to {state_name}")
                mapped_keys.add(state_name)
                if state_name != name:
                    renamed_keys.append({"serialized_key": name, "loaded_key": state_name})
                value = handle.get_tensor(name)
                source_dtypes[str(value.dtype)] += 1
                target = expected[state_name]
                if value.shape != target.shape:
                    raise ValueError(f"Serialized weight shape differs: {name}")
                # Both independent loaders request the same per-release storage dtype.
                _require_tensor_equal(target, value.to(dtype=target.dtype), name=name)
    if _canonical_checkpoint_keys(mapped_keys, tied=tied) != set(expected):
        raise ValueError("Serialized checkpoint does not cover every model state key")
    return {
        "model_state_keys": len(actual),
        "serialized_weight_keys": len(serialized),
        "serialized_key_renames": renamed_keys,
        "serialized_dtypes": dict(source_dtypes),
        "loaded_dtypes": dict(Counter(str(value.dtype) for value in actual.values())),
        "exact_native_publisher_weights": True,
        "complete_serialized_weight_coverage": True,
        "parameter_identity_ties": groups,
    }


def _require_native_class(model: nn.Module, expected_class: type[nn.Module]) -> None:
    """Accept native capability subclasses while rejecting unrelated HF fallback classes."""
    if not isinstance(model, expected_class):
        actual = type(model)
        raise ValueError(
            f"Actual AutoModel loader selected {actual.__module__}.{actual.__qualname__}; "
            f"expected native {expected_class.__module__}.{expected_class.__qualname__}"
        )


def _load_native(model_name: str) -> nn.Module:
    """Exercise actual NeMoAuto pretrained routing, without optional kernel patches."""
    from nemo_automodel import NeMoAutoModelForImageTextToText
    from nemo_automodel.components.models.d1.model import D1OmniModel
    from nemo_automodel.components.models.d1.vl_model import Lfm2VlForConditionalGeneration

    repository, revision, _ = RELEASES[model_name]
    model = (
        NeMoAutoModelForImageTextToText.from_pretrained(
            repository,
            revision=revision,
            code_revision=revision,
            trust_remote_code=True,
            dtype=RELEASE_DTYPES[model_name],
            attn_implementation="sdpa",
            use_liger_kernel=False,
            use_sdpa_patching=False,
        )
        .cuda()
        .eval()
    )
    expected_class = D1OmniModel if model_name == "d1-omni" else Lfm2VlForConditionalGeneration
    _require_native_class(model, expected_class)
    if any(value.is_meta for value in (*model.parameters(), *model.buffers())):
        raise ValueError("Actual AutoModel loader left unmaterialized model tensors")
    return model


def _features(readout, examples: list, device: torch.device) -> list[torch.Tensor]:
    """Extract independently encoded questions through the model-owned contract.

    Returns:
        Detached Tensors of shape [hidden] or [sequence, hidden], one per question.
    """
    values = []
    with torch.no_grad():
        for example in examples:
            ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
            values.append(readout.extract_features(ids, torch.ones_like(ids, dtype=torch.bool)).detach())
    return values


def _alias_probabilities(values: torch.Tensor, groups: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    """Reduce vocabulary diagnostics using the publisher-encoded answer aliases.

    Args:
        values: Tensor of shape [vocab] at the final answer position.
        groups: Vocabulary IDs for each option, in target order.

    Returns:
        Float32 Tensor of shape [options] containing normalized probabilities.
    """
    return torch.stack([values.float()[list(group)].amax() for group in groups]).softmax(-1)


def _diagnose(native, publisher, example, question, *, omni: bool) -> dict:
    """Compare direct-chain readouts only to localize a failed public-API check."""
    ids = torch.tensor([example.input_ids], dtype=torch.long, device="cuda")
    with torch.no_grad():
        if omni:
            markers = torch.tensor([example.marker_positions], device=ids.device)
            direct = (
                native(
                    ids,
                    torch.ones_like(ids, dtype=torch.bool),
                    markers,
                    torch.ones_like(markers, dtype=torch.bool),
                    torch.tensor([example.head_question_type], device=ids.device),
                ).logits[0]
                / example.temperature
            )
            reference = publisher._forward([(None, ids[0].tolist(), list(example.marker_positions), question, True)])[0]
            if question.type == "noul":
                reference = reference[::-1]
            return {"native_direct": direct.softmax(-1).tolist(), "publisher_direct": reference}
        actual = native(ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        expected = publisher(ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()

        return {
            "native_direct": _alias_probabilities(actual, example.alias_token_ids).tolist(),
            "publisher_direct": _alias_probabilities(expected, example.alias_token_ids).tolist(),
            "full_vocabulary_max_abs_error": float((actual - expected).abs().max()),
            "full_vocabulary_cosine": float(nn.functional.cosine_similarity(actual, expected, dim=0)),
        }


def _check_media(native, publisher, tokenizer, prompt, record: dict, examples: list, *, omni: bool) -> dict:
    """Check deterministic media using full released weights and existing tolerances.

    A separately loaded native model retains the same per-release storage
    dtype as the publisher, independently of text-readout preparation.
    """
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor

    image = Image.new("RGB", (64, 64), color=(100, 50, 200))
    checks = []
    if not omni:
        repository, revision, _ = RELEASES["d1-3b"]
        publisher.engine.processor = AutoProcessor.from_pretrained(
            repository, revision=revision, code_revision=revision, trust_remote_code=True
        )
        for example in examples:
            question = prompt.as_question(record["questions"][example.question_id])
            engine = publisher.engine
            prefix = prompt.prefix_text(
                tokenizer, record["state"], engine.bos, engine.state_style, engine.system, engine._image_markup(1)
            )
            text = prefix + prompt.suffix_text(tokenizer, question, engine.lead, engine.option_style)
            inputs = engine._image_inputs(text, [image])
            with torch.no_grad():
                expected = publisher(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                actual = native(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
            cosine = float(nn.functional.cosine_similarity(actual, expected, dim=0))
            if cosine <= 0.9999:
                raise ValueError("Released image full-vocabulary cosine check failed")
            actual_probabilities = _alias_probabilities(actual, example.alias_token_ids)
            expected_probabilities = _alias_probabilities(expected, example.alias_token_ids)
            torch.testing.assert_close(actual_probabilities, expected_probabilities, atol=PROBABILITY_ATOL, rtol=0)
            checks.append(
                {
                    "modality": "image",
                    "question": example.question_id,
                    "full_vocabulary_max_abs_error": float((actual - expected).abs().max()),
                    "full_vocabulary_cosine": cosine,
                    "probability_max_abs_error": float((actual_probabilities - expected_probabilities).abs().max()),
                }
            )
        return {
            "scope": "released-size image full logits and answer aliases",
            "logit_atol": 0.01,
            "logit_rtol": 0.01,
            "minimum_cosine": 0.9999,
            "probability_atol": PROBABILITY_ATOL,
            "checks": checks,
        }
    waveform = np.sin(np.arange(8000, dtype=np.float32) * 0.1) * 0.1
    for modality in ("image", "audio"):
        with torch.no_grad():
            expected_prefix = publisher.vision([image]) if modality == "image" else publisher.audio(waveform)
            prefix = native.encode_images([image]) if modality == "image" else native.encode_audio(waveform)
        torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
        if not torch.isfinite(prefix).all():
            raise ValueError("Released perception prefix is nonfinite")
        configured_limit = (
            publisher.config.image_text_length if modality == "image" else publisher.config.audio_text_length
        )
        limit = min(configured_limit, publisher.config.max_length - prefix.shape[1])
        for example in examples:
            question = prompt.as_question(record["questions"][example.question_id])
            ids, markers = prompt.encode(
                tokenizer, record["state"], question, limit, {"false": "no", "true": "yes"}, modality == "audio"
            )
            input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
            positions = torch.tensor([markers], device="cuda", dtype=torch.long)
            with torch.no_grad():
                expected = publisher._forward([(expected_prefix, ids, markers, question, False)])[0]
                logits = native(
                    input_ids,
                    torch.ones_like(input_ids, dtype=torch.bool),
                    positions,
                    torch.ones_like(positions, dtype=torch.bool),
                    torch.tensor([prompt.QTYPES[question.type]], device="cuda"),
                    prefix,
                    torch.tensor([prefix.shape[1]], device="cuda"),
                ).logits[0]
                actual = logits.softmax(-1)
            if question.type == "noul":
                expected = expected[::-1]
            expected = torch.tensor(expected, device=actual.device)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            checks.append(
                {
                    "modality": modality,
                    "question": example.question_id,
                    "prefix_length": prefix.shape[1],
                    "exact_prefix": True,
                    "probability_max_abs_error": float((actual - expected).abs().max()),
                }
            )
    return {
        "scope": "released-size image/audio prefixes and uncalibrated choice/score/noul probabilities",
        "prefix_atol": 0,
        "prefix_rtol": 0,
        "probability_atol": 2e-6,
        "probability_rtol": 2e-6,
        "checks": checks,
    }


def _train_and_save(
    native, publisher, examples: list, output_dir: Path, *, omni: bool, questions: list[QuestionTarget]
) -> tuple[dict, list]:
    """Run one real supervised head update and prove every backbone tensor stayed frozen."""
    from nemo_automodel.components.loss.decision import DecisionLossConfig
    from nemo_automodel.components.loss.decision_batch import DecisionTargets, QuestionLogits, assemble_decision_batch

    native.requires_grad_(False)
    readout = native.prepare_text_readout(examples)
    head = readout.head.requires_grad_(True)
    features = _features(readout, examples, torch.device("cuda"))
    before = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    optimizer = torch.optim.SGD(head.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    values = readout.score(features, examples, head=head)
    targets = DecisionTargets.from_questions(
        record_ids=[example.record_id for example in examples],
        questions=[[question] for question in questions],
        device=values[0].device,
    )
    batch = assemble_decision_batch(
        logits=[
            [QuestionLogits(example.record_id, example.question_id, example.option_ids, row)]
            for example, row in zip(examples, values, strict=True)
        ],
        targets=targets,
    )
    loss, _ = DecisionLossConfig(mode="sft").build()(batch.logits, batch.targets)
    if not torch.isfinite(loss):
        raise ValueError("Head-only loss is nonfinite")
    loss.backward()
    gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise ValueError("Head gradients are missing or nonfinite")
    if not any(torch.count_nonzero(value) for value in gradients):
        raise ValueError("Every head gradient is zero")
    head_ids = {id(parameter) for parameter in head.parameters()}
    if any(parameter.grad is not None for parameter in native.parameters() if id(parameter) not in head_ids):
        raise ValueError("Frozen backbone received gradients")
    norm = nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    if not torch.isfinite(norm):
        raise ValueError("Head gradient norm is nonfinite")
    optimizer.step()
    after = head.state_dict()
    if not any(not torch.equal(after[name].cpu(), before[name]) for name in after):
        raise ValueError("Optimizer did not change the head")
    reference = publisher.state_dict()
    backbone_keys = []
    for name, value in native.state_dict().items():
        if omni and name.startswith("head."):
            continue
        _require_tensor_equal(value, reference[name], name=name)
        backbone_keys.append(name)
    head.eval()
    with torch.no_grad():
        logits = [value.cpu() for value in readout.score(features, examples, head=head)]
    torch.save({name: value.detach().cpu() for name, value in after.items()}, output_dir / "readout.pt")
    return {
        "loss": float(loss.detach()),
        "head_parameters_with_gradients": len(gradients),
        "head_changed": True,
        "frozen_backbone_keys_unchanged": len(backbone_keys),
        "readout_checkpoint_saved": True,
    }, logits


def _run(args: argparse.Namespace, result: dict) -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from nemo_automodel.components.datasets.decision.records import load_records
    from nemo_automodel.components.models.d1.records import encode_d1_record

    repository, revision, class_name = RELEASES[args.model]
    omni = args.model == "d1-omni"
    snapshot = Path(
        snapshot_download(
            repository, revision=revision, allow_patterns=["*.safetensors", "*.json", "*.py", "*.model", "*.txt"]
        )
    )
    published_class = get_class_from_dynamic_module(class_name, repository, revision=revision, code_revision=revision)
    prompt = importlib.import_module(published_class.__module__.rsplit(".", 1)[0] + ".prompt")
    tokenizer = AutoTokenizer.from_pretrained(
        repository, revision=revision, code_revision=revision, trust_remote_code=True
    )
    if omni:
        # The pinned Omni publisher owns its SDPA execution but does not declare
        # the HF SDPA support flag. Preserve its supported constructor contract.
        publisher = published_class.from_pretrained(repository, revision=revision, dtype=RELEASE_DTYPES[args.model])
    else:
        publisher = published_class.from_pretrained(
            repository, revision=revision, dtype=RELEASE_DTYPES[args.model], attn_implementation="sdpa"
        )
    publisher = publisher.cuda().eval()
    # Override lazy runtime resources, never source code or numerical methods.
    # The publisher otherwise resolves its tokenizer from the mutable default revision.
    if omni:
        publisher.tokenizer = tokenizer
    else:
        runner = importlib.import_module(published_class.__module__.rsplit(".", 1)[0] + ".runner")
        publisher.engine = runner.SystemOne(model=publisher, tokenizer=tokenizer)
    native = _load_native(args.model)
    result["native_class"] = type(native).__name__
    result["weights"] = _check_weights(native, publisher, snapshot, tied=not omni)
    config = AutoConfig.from_pretrained(repository, revision=revision, code_revision=revision, trust_remote_code=True)
    records = load_records(args.data_dir / "validation.jsonl")[:3]
    if len(records) != 3 or any(set(record["questions"]) != QUESTION_IDS for record in records):
        raise ValueError("Acceptance requires three validation records with all five declared questions")
    for record in records:
        validate_record(record)
    examples = [
        example
        for record in records
        for example in encode_d1_record(
            record,
            tokenizer=tokenizer,
            prompt=prompt,
            omni=omni,
            max_length=config.max_length if omni else config.text_config.max_position_embeddings,
            temperatures=config.temperatures if omni else {},
        )
    ]
    if len(examples) != 15:
        raise ValueError("Acceptance must exercise exactly fifteen typed decisions")
    native.requires_grad_(False)
    readout = native.prepare_text_readout(examples)
    features = _features(readout, examples, torch.device("cuda"))
    with torch.no_grad():
        values = readout.score(features, examples, head=readout.head)
    records_by_id = {str(record["id"]): record for record in records}
    checks, failures = [], []
    for example, logits in zip(examples, values, strict=True):
        record = records_by_id[example.record_id]
        qid = example.question_id
        schema = record["questions"][qid]
        question = prompt.as_question(schema)
        if not omni:
            rendered = publisher.engine.render(record["state"], question)
            if tuple(tokenizer.encode(rendered, add_special_tokens=False)) != example.input_ids:
                raise ValueError("Publisher and native single-question token identities differ")
        answer = publisher.system_one(record["state"], {qid: schema})["answers"][qid]
        probabilities = (
            {"true": answer["noul"], "false": 1 - answer["noul"]}
            if schema["type"] == "noul"
            else answer["probabilities"]
        )
        if set(probabilities) != set(example.option_ids):
            raise ValueError("Publisher and native option identities differ")
        expected = torch.tensor([probabilities[key] for key in example.option_ids], device=logits.device)
        actual = logits.softmax(-1)
        maximum = float((actual - expected).abs().max())
        row = {
            "record_id": example.record_id,
            "question": qid,
            "option_ids": list(example.option_ids),
            "max_probability_error": maximum,
        }
        checks.append(row)
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all() or maximum > PROBABILITY_ATOL:
            failures.append(
                {
                    **row,
                    "native_readout": actual.tolist(),
                    "publisher_api": expected.tolist(),
                    **_diagnose(native, publisher, example, question, omni=omni),
                }
            )
    result["probability_checks"] = checks
    if failures:
        (args.output_dir / "probability-diagnostics.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise ValueError(f"Released single-question probability acceptance failed for {len(failures)} decisions")
    if args.include_media:
        media_native = _load_native(args.model)
        result["media"] = _check_media(media_native, publisher, tokenizer, prompt, records[0], examples[:5], omni=omni)
        del media_native
        gc.collect()
        torch.cuda.empty_cache()
    del readout, features, values
    training_examples = examples[:5]
    result["training"], trained_logits = _train_and_save(
        native,
        publisher,
        training_examples,
        args.output_dir,
        omni=omni,
        questions=[
            question_targets(records_by_id[example.record_id], {example.question_id: example.option_ids})[0]
            for example in training_examples
        ],
    )
    del native, publisher
    gc.collect()
    torch.cuda.empty_cache()
    fresh = _load_native(args.model).requires_grad_(False)
    fresh_readout = fresh.prepare_text_readout(examples)
    saved = torch.load(args.output_dir / "readout.pt", map_location="cuda", weights_only=True)
    fresh_readout.head.load_state_dict(saved, strict=True)
    for name, value in fresh_readout.head.state_dict().items():
        _require_tensor_equal(value, saved[name], name=name)
    reloaded_features = _features(fresh_readout, training_examples, torch.device("cuda"))
    with torch.no_grad():
        reloaded = fresh_readout.score(reloaded_features, training_examples, head=fresh_readout.head)
    for actual, expected in zip(reloaded, trained_logits, strict=True):
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-6)
    result["reload"] = {
        "fresh_native_loader": True,
        "fresh_readout": True,
        "exact_head_state": True,
        "logit_atol": 1e-6,
        "logit_rtol": 1e-6,
    }


def main() -> None:
    """Run explicit released-size acceptance; --help performs no downloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=RELEASES, required=True)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-media",
        action="store_true",
        help="Also check full released image/audio paths after text parity passes",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "tutorials/decision_models/fixtures/smoke",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Released-size acceptance requires CUDA")
    torch.manual_seed(7)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repository, revision, _ = RELEASES[args.model]
    result = {
        "model": repository,
        "revision": revision,
        "scope": "released-size text-only decisions; frozen-backbone head update and fresh reload",
        "dtype": str(RELEASE_DTYPES[args.model]).removeprefix("torch."),
        "absolute_probability_tolerance": PROBABILITY_ATOL,
        "validation_records": 3,
        "typed_decisions": 15,
        "status": "running",
    }
    try:
        _run(args, result)
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__)
        raise
    else:
        result["status"] = "passed"
    finally:
        (args.output_dir / "acceptance-result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
