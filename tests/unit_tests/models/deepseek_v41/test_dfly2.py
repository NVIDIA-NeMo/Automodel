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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
)
from nemo_automodel.components.models.deepseek_v41.dfly2 import DeepseekV41DFly2Config, DeepseekV41DFly2Model
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from nemo_automodel.components.speculative.dfly2 import DFly2TargetModel
from nemo_automodel.recipes.speculative_decoding._dspark_warm_start import load_dspark_draft_weights

pytestmark = [pytest.mark.timeout(70), pytest.mark.cpu_only]

# Layers 4 and 5 are ratio-1 layers: 4 publishes the shared 1:1 compressed KV, both select keys.
_PAIRED_LAYERS = (4, 5)


def _text_config(**overrides) -> DeepseekV41TextConfig:
    values = dict(
        vocab_size=64,
        hidden_size=16,
        moe_intermediate_size=16,
        num_hidden_layers=6,
        num_attention_heads=2,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        # Two trailing schedule slots for the draft layers, without compressor or indexer.
        compress_ratios=[0, 0, 2, 2, 1, 1, 0, 0],
        kv_source_layer_ids=[2, 4],
        index_source_layer_ids=[2, 4, 5],
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        candidate_source_layer_id=4,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        sliding_window=4,
        engram_layer_ids=[],
        engram_num_embeddings=[],
        num_nextn_predict_layers=0,
        dspark_block_size=0,
        dspark_noise_token_id=0,
        dtype="float32",
    )
    values.update(overrides)
    return DeepseekV41TextConfig(**values)


def _backend() -> BackendConfig:
    return BackendConfig(attn="eager", linear="torch", rms_norm="torch_fp32", experts="torch", dispatcher="torch")


def _target() -> DeepseekV41ForCausalLM:
    torch.manual_seed(3)
    config = DeepseekV41Config(vision_config=DeepseekV41VisionConfig(num_hidden_layers=0), text_config=_text_config())
    with torch.device("meta"):
        model = DeepseekV41ForCausalLM(config, backend=_backend())
    model.to_empty(device="cpu")
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    return model.eval()


def _draft(**overrides) -> DeepseekV41DFly2Model:
    torch.manual_seed(5)
    values = dict(
        text_config=_text_config(),
        target_layer_ids=_PAIRED_LAYERS,
        block_size=3,
        num_anchors=4,
        mask_token_id=63,
        n_routed_experts=4,
        num_experts_per_tok=2,
    )
    values.update(overrides)
    return DeepseekV41DFly2Config(**values).build(_backend())


def test_capture_reproduces_each_paired_target_attention() -> None:
    """A draft attention started from its paired layer reproduces that layer from the captured keys alone.

    Fed the target layer's own input and the captured key list of every row (without any
    draft key), the draft attention must give exactly the target layer's attention output:
    this pins the window keys, the shared compressed keys, and the index offsets together.
    """
    target = _target()
    draft = _draft()
    draft.initialize_from_target(target)
    input_ids = torch.randint(0, 64, (2, 12))
    attention_mask = torch.ones(2, 12, dtype=torch.long)
    attention_mask[1, 10:] = 0
    seen = {}

    def _record(layer_id):
        def _hook(_module, args, kwargs, output):
            seen[layer_id] = (args[0], output.hidden_states)

        return _hook

    handles = [
        target.model.layers[str(layer_id)].attn.register_forward_hook(_record(layer_id), with_kwargs=True)
        for layer_id in _PAIRED_LAYERS
    ]
    try:
        batch = DFly2TargetModel(target, _PAIRED_LAYERS).generate_batch(input_ids, attention_mask, attention_mask)
    finally:
        for handle in handles:
            handle.remove()

    features = batch.features
    positions = torch.arange(12).expand(2, -1)
    for index, layer_id in enumerate(_PAIRED_LAYERS):
        layer_input, expected = seen[layer_id]
        keys, indices = features.context_keys[index], features.context_indices[index]
        assert keys.shape == (2, 24, 8) and indices.shape == (2, 12, 4 + 2)
        assert int(indices.max()) < 24
        attended = draft.layers[index].attn(layer_input, keys, indices, position_ids=positions)
        valid = attention_mask.bool()
        torch.testing.assert_close(attended[valid], expected[valid], rtol=1e-5, atol=1e-5)
    assert features.last_hidden_states.shape == (2, 12, 16)


def test_window_rows_end_at_the_row_and_skip_padding() -> None:
    target = _target()
    attention_mask = torch.ones(1, 6, dtype=torch.long)
    attention_mask[0, 5] = 0
    features = target.dfly2_target_features(torch.randint(0, 64, (1, 6)), attention_mask=attention_mask, layer_ids=[4])
    window = features.context_indices[0][0, :, :4]
    assert window[0].tolist() == [0, -1, -1, -1]
    assert window[4].tolist() == [1, 2, 3, 4]
    # The padded token is never a window key, and selected keys index the global copy.
    assert (window != 5).all()
    picks = features.context_indices[0][0, :, 4:]
    assert ((picks == -1) | (picks >= 6)).all()


def test_only_ratio_one_layers_can_be_paired() -> None:
    with pytest.raises(ValueError, match="ratio-1"):
        _draft(target_layer_ids=(2, 5))
    with pytest.raises(ValueError, match="ratio-1"):
        _target().dfly2_target_features(
            torch.zeros(1, 4, dtype=torch.long), attention_mask=torch.ones(1, 4), layer_ids=[3]
        )


def test_fresh_draft_starts_as_its_backbone_and_trains_every_head() -> None:
    torch.manual_seed(11)
    draft = _draft()
    draft.initialize_embeddings_and_head(embed_tokens=torch.nn.Embedding(64, 16), lm_head=torch.nn.Linear(16, 64))
    hidden = torch.randn(1, 6, 16)
    previous = torch.randn(1, 6, 16)
    keys = torch.randn(1, 10, 8)
    indices = torch.randint(0, 10, (1, 6, 3), dtype=torch.int32)
    corrected = draft.hidden_correction(
        hidden, previous, keys, indices, position_ids=torch.arange(6).view(1, -1), block_size=3
    )
    torch.testing.assert_close(corrected, hidden, rtol=0, atol=0)
    assert draft.mtp_fusion(torch.randn(1, 2, 16), torch.randn(1, 2, 16)).abs().max() == 0

    sequence = 10
    input_ids = torch.randint(0, 60, (1, sequence))
    loss_mask = torch.ones(1, sequence)
    context_keys = [torch.randn(1, 2 * sequence, 8) for _ in _PAIRED_LAYERS]
    context_indices = [torch.randint(-1, 2 * sequence, (1, sequence, 5), dtype=torch.int32) for _ in _PAIRED_LAYERS]
    outputs = draft(input_ids, loss_mask, context_keys, context_indices, torch.randn(1, sequence, 16))
    assert outputs.draft_logits.shape == (1, 4, 2, 64)
    assert outputs.aligned_target_logits.shape == (1, 4, 2, 64)
    outputs.draft_logits.square().mean().backward()
    # Zero-initialized outputs receive gradient at the first step, so every head can open.
    assert draft.mtp_fusion.gate.grad.abs().sum() > 0
    assert draft.hidden_correction.down_proj.weight.grad.abs().sum() > 0
    assert draft.hidden_correction.attention.wo_b.weight.grad.abs().sum() > 0
    assert draft.layers[0].ffn.experts.gate_and_up_projs.grad is not None
    assert draft.embed_tokens.weight.grad is None and draft.lm_head.weight.grad is None


def test_drafted_slots_label_the_tokens_after_the_anchor() -> None:
    torch.manual_seed(2)
    draft = _draft(num_anchors=1)
    draft.initialize_embeddings_and_head(embed_tokens=torch.nn.Embedding(64, 16), lm_head=torch.nn.Linear(16, 64))
    input_ids = torch.arange(10, 20).view(1, -1)
    loss_mask = torch.zeros(1, 10)
    loss_mask[0, 6:8] = 1
    last_hidden = torch.randn(1, 10, 16)
    context = [torch.randn(1, 20, 8) for _ in _PAIRED_LAYERS]
    indices = [torch.full((1, 10, 2), -1, dtype=torch.int32) for _ in _PAIRED_LAYERS]
    outputs = draft(input_ids, loss_mask, context, indices, last_hidden)
    # The only eligible anchor is 6 (its first label, 7, is supervised): slots predict 7 and 8,
    # and the second is dropped because token 8 is not supervised.
    assert outputs.target_ids.tolist() == [[[17, 18]]]
    assert outputs.eval_mask.tolist() == [[[True, False]]]
    # Each teacher is the target's distribution at the token before its label.
    torch.testing.assert_close(outputs.aligned_target_logits[0, 0], draft.compute_logits(last_hidden[0, 6:8]))


def test_confidence_head_reads_detached_draft_states() -> None:
    draft = _draft(enable_confidence_head=True, confidence_head_stop_gradient=True)
    draft.initialize_embeddings_and_head(embed_tokens=torch.nn.Embedding(64, 16), lm_head=torch.nn.Linear(16, 64))
    sequence = 10
    outputs = draft(
        torch.randint(0, 60, (1, sequence)),
        torch.ones(1, sequence),
        [torch.randn(1, 2 * sequence, 8) for _ in _PAIRED_LAYERS],
        [torch.randint(-1, 2 * sequence, (1, sequence, 4), dtype=torch.int32) for _ in _PAIRED_LAYERS],
        torch.randn(1, sequence, 16),
    )
    assert outputs.confidence_pred.shape == (1, 4, 2) and outputs.confidence_pred.dtype == torch.float32
    outputs.confidence_pred.sum().backward()
    assert draft.confidence_head.confidence_proj.weight.grad is not None
    assert all(parameter.grad is None for parameter in draft.layers.parameters())


def test_released_checkpoint_roundtrip() -> None:
    draft = _draft()
    expected = {key: value.detach().clone() for key, value in draft.state_dict().items()}
    released = draft.state_dict_adapter.to_hf(draft.state_dict())
    # Names of the released AngelSlim/DS-V4.1-Flash-DFly2-B6 layout.
    for key in (
        "model.layers.0.attn.wq_a.weight",
        "model.layers.0.attn.attn_sink",
        "model.layers.0.attn_norm.weight",
        "model.layers.1.ffn_norm.weight",
        "model.layers.0.hc_attn_fn",
        "model.layers.1.hc_ffn_scale",
        "model.layers.0.ffn.gate.bias",
        "model.layers.0.ffn.gate.bias_vl",
        "model.layers.0.ffn.shared_experts.w1.weight",
        "model.layers.0.ffn.experts.3.w2.weight",
        "model.final_norm.weight",
        "model.mtp_fusion.eh_proj.weight",
        "model.mtp_fusion.gate",
        "model.hidden_correction.down_proj.weight",
        "model.hidden_correction.attention.q_in.weight",
        "model.hidden_correction.attention.attn_sink",
        "model.hidden_correction.attention.block_rel_bias",
        "model.hidden_correction.attention.block_gate.weight",
    ):
        assert key in released, key
    assert set(released) == set(draft.state_dict_adapter.get_hf_state_dict_keys(draft.state_dict()))

    restored = draft.state_dict_adapter.from_hf(released)
    assert restored.keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(restored[key], value, rtol=0, atol=0)


def test_warm_start_reads_the_released_layout(tmp_path: Path) -> None:
    source = _draft()
    with torch.no_grad():
        source.mtp_fusion.gate.fill_(0.5)
        source.hidden_correction.attention.block_sink.weight.fill_(-1.0)
    released = source.state_dict_adapter.to_hf(source.state_dict())
    # The release ships neither the shared embedding nor the head.
    released = {
        key: value.contiguous()
        for key, value in released.items()
        if key not in ("model.embed_tokens.weight", "model.lm_head.weight")
    }
    save_file(released, str(tmp_path / "model.safetensors"))

    draft = _draft()
    draft.initialize_embeddings_and_head(embed_tokens=torch.nn.Embedding(64, 16), lm_head=torch.nn.Linear(16, 64))
    load_dspark_draft_weights(
        draft, str(tmp_path), keep_target_embeddings=True, target_layer_ids=list(_PAIRED_LAYERS), mask_token_id=63
    )
    torch.testing.assert_close(draft.mtp_fusion.gate, source.mtp_fusion.gate)
    torch.testing.assert_close(
        draft.hidden_correction.attention.block_sink.weight, source.hidden_correction.attention.block_sink.weight
    )
    torch.testing.assert_close(
        draft.layers[1].ffn.experts.down_projs, source.layers[1].ffn.experts.down_projs, rtol=0, atol=0
    )


def test_exported_config_carries_the_dfly2_settings() -> None:
    draft = _draft()
    exported = json.loads(draft.config.to_json_string(use_diff=False))
    assert exported["architectures"] == ["DeepseekV41DFly2Model"]
    assert exported["dfly_config"]["block_size"] == 3
    assert exported["dfly_config"]["target_kv_layers"] == list(_PAIRED_LAYERS)
    assert exported["dfly_config"]["mask_token_id"] == 63


def test_draft_does_not_inherit_the_targets_sparse_backward() -> None:
    config = DeepseekV41Config(
        vision_config=DeepseekV41VisionConfig(num_hidden_layers=0),
        text_config=_text_config(sparse_attention_backward="deterministic"),
    )
    options = SimpleNamespace(
        target_layer_ids=[4, 5],
        block_size=3,
        num_anchors=2,
        mask_token_id=63,
        enable_confidence_head=False,
        confidence_head_stop_gradient=True,
        attn_backend="eager",
    )
    # The target's TileLang-only deterministic backward would reject the eager draft.
    draft = config.build_dfly2_draft(options)
    assert config.text_config.sparse_attention_backward == "deterministic"
    assert draft.dfly2_config.text_config.sparse_attention_backward == "atomic"


def _fsdp_draft_features(batch: int, sequence: int) -> tuple:
    generator = torch.Generator().manual_seed(23)
    input_ids = torch.randint(0, 60, (batch, sequence), generator=generator)
    context_keys = tuple(torch.randn(batch, 2 * sequence, 8, generator=generator) for _ in _PAIRED_LAYERS)
    context_indices = tuple(
        torch.randint(-1, 2 * sequence, (batch, sequence, 4), generator=generator).int() for _ in _PAIRED_LAYERS
    )
    return input_ids, context_keys, context_indices, torch.randn(batch, sequence, 16, generator=generator)


def _fsdp_trainer():
    from nemo_automodel.components.speculative.dfly2 import DFly2TrainerModule

    # Every valid anchor is taken (num_anchors exceeds the candidates), so the blocks do not
    # depend on how the global batch is split across ranks.
    draft = _draft(num_anchors=16, enable_confidence_head=True, confidence_head_stop_gradient=False)
    torch.manual_seed(29)
    draft.initialize_embeddings_and_head(embed_tokens=torch.nn.Embedding(64, 16), lm_head=torch.nn.Linear(16, 64))
    with torch.no_grad():
        # Open every zero-initialized head so all parameters carry gradient.
        draft.mtp_fusion.gate.fill_(0.5)
        draft.hidden_correction.down_proj.weight.normal_(std=0.1)
        draft.hidden_correction.attention.wo_b.weight.normal_(std=0.1)
        draft.hidden_correction.attention.block_gate_bias.weight.fill_(0.3)
    return DFly2TrainerModule(draft, confidence_head_alpha=1.0)


def _fsdp_training_worker(rank: int, rendezvous: str, reference_path: str, compute_dtype: str) -> None:
    from datetime import timedelta

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor

    from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
    from nemo_automodel.recipes.speculative_decoding.train_dfly2 import TrainDFly2Recipe

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    try:
        trainer = _fsdp_trainer()
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
        param_dtype = getattr(torch, compute_dtype)
        policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32, cast_forward_inputs=False)
        # The recipe's sharding: dtype-uniform units for the draft units, then the root.
        for unit in TrainDFly2Recipe._draft_fsdp_units(trainer.draft_model):
            fully_shard_by_dtype(
                unit,
                mesh=mesh,
                mp_policy=policy,
                offload_policy=None,
                fp32_compute_module_names=tuple(trainer.draft_model._keep_in_fp32_modules_strict),
            )
        fully_shard(trainer, mesh=mesh, mp_policy=policy)

        input_ids, keys, indices, last_hidden = _fsdp_draft_features(2, 10)
        local = slice(rank, rank + 1)
        loss_mask = torch.ones(2, 10)
        loss_mask[0, 7:] = 0  # an uneven split of supervised slots across the ranks
        metrics = trainer(
            input_ids=input_ids[local],
            loss_mask=loss_mask[local],
            context_keys=tuple(key[local] for key in keys),
            context_indices=tuple(index[local] for index in indices),
            target_last_hidden_states=last_hidden[local],
        )
        metrics.loss.backward()
        reference = torch.load(reference_path)
        grads = {}
        for name, parameter in trainer.named_parameters():
            if name not in reference:
                assert parameter.grad is None, name
                continue
            grads[name] = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
        if param_dtype == torch.float32:
            for name, grad in grads.items():
                torch.testing.assert_close(grad, reference[name], rtol=1e-4, atol=1e-6, msg=name)
        else:
            # The recipe's BF16 policy: the FP32 heads must still run (a BF16 weight against their
            # FP32 inputs fails), compute in FP32, and the draft's gradient must stay close overall.
            assert all(torch.isfinite(grad).all() for grad in grads.values())
            head = "draft_model.confidence_head.confidence_proj.weight"
            torch.testing.assert_close(grads[head], reference[head], rtol=0.05, atol=1e-5)
            flat = torch.cat([grad.flatten() for grad in grads.values()])
            expected = torch.cat([reference[name].flatten() for name in grads])
            assert (flat - expected).norm() <= 0.05 * expected.norm()
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(
    20, hard_timeout=70, reason="Two-process gloo FSDP2 forward and backward of the draft against one process."
)
@pytest.mark.parametrize("compute_dtype", ["float32", "bfloat16"])
def test_fsdp_gradient_matches_the_single_process_global_batch(tmp_path: Path, compute_dtype: str) -> None:
    """Sharded like the recipe over two ranks, the draft's gradient equals the one-process gradient.

    The objective divides by the weight sum of all ranks and scales by the world size, so
    FSDP's gradient average reproduces the global weighted mean even though the ranks hold
    different numbers of supervised slots.
    """
    import torch.multiprocessing as mp

    trainer = _fsdp_trainer()
    input_ids, keys, indices, last_hidden = _fsdp_draft_features(2, 10)
    loss_mask = torch.ones(2, 10)
    loss_mask[0, 7:] = 0
    metrics = trainer(
        input_ids=input_ids,
        loss_mask=loss_mask,
        context_keys=keys,
        context_indices=indices,
        target_last_hidden_states=last_hidden,
    )
    metrics.loss.backward()
    reference = {name: parameter.grad for name, parameter in trainer.named_parameters() if parameter.grad is not None}
    assert "draft_model.confidence_head.confidence_proj.weight" in reference
    assert "draft_model.mtp_fusion.gate" in reference
    torch.save(reference, tmp_path / "reference.pt")
    mp.spawn(
        _fsdp_training_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "reference.pt"), compute_dtype),
        nprocs=2,
        join=True,
    )


def test_each_drafted_slot_reads_only_what_exists_when_its_block_is_drafted() -> None:
    """Perturb one input row at a time; slot ``k`` of the block at anchor ``a`` may move only for

    * tokens up to row ``a + k - 1`` (the anchor, then the teacher-forced previous tokens),
    * the target's last hidden state at row ``a - 1`` (the MTP fusion),
    * keys listed by row ``a - 1``'s key list of the same paired layer, and that list itself.

    A label leak, a fusion or key list read from row ``a``, or a draft layer reading another
    paired layer's keys each moves a slot that must not move.
    """
    torch.manual_seed(7)
    sequence, block_size = 10, 3
    # Every candidate anchor is taken, so block j sits at anchor j.
    draft = _draft(num_anchors=sequence, block_size=block_size)
    draft.initialize_embeddings_and_head(embed_tokens=torch.nn.Embedding(64, 16), lm_head=torch.nn.Linear(16, 64))
    with torch.no_grad():
        # Open the zero-initialized heads so every path carries signal.
        draft.mtp_fusion.gate.fill_(1.0)
        draft.hidden_correction.down_proj.weight.normal_(std=0.2)
        attention = draft.hidden_correction.attention
        # Full projections: the embedding half of q_in / kv_in and the gate start at zero.
        for linear in (attention.wo_b, attention.q_in, attention.kv_in, attention.block_gate):
            linear.weight.normal_(std=0.2)
        attention.block_gate_bias.weight.fill_(0.5)
    generator = torch.Generator().manual_seed(11)
    input_ids = torch.randint(0, 60, (1, sequence), generator=generator)
    last_hidden = torch.randn(1, sequence, 16, generator=generator)
    keys = [torch.randn(1, sequence, 8, generator=generator) for _ in _PAIRED_LAYERS]
    # Causal key lists that differ between the two paired layers: row r lists keys at or before r.
    lists = []
    for _ in _PAIRED_LAYERS:
        rows = [torch.randperm(r + 1, generator=generator)[:3] for r in range(sequence)]
        lists.append(torch.stack([torch.nn.functional.pad(row, (0, 3 - len(row)), value=-1) for row in rows])[None])
    loss_mask = torch.ones(1, sequence)

    def logits(ids=input_ids, hidden=last_hidden, context=keys, indices=lists):
        with torch.no_grad():
            return draft(ids, loss_mask, context, [index.int() for index in indices], hidden).draft_logits[0]

    base = logits()
    anchors = range(sequence - 1)
    violations = []

    def check(name, changed, allowed):
        moved = (changed - base).abs().amax(-1) > 1e-6  # [blocks, drafted]
        for anchor in anchors:
            for drafted in range(block_size - 1):
                if moved[anchor, drafted] and not allowed(anchor, drafted + 1):
                    violations.append((name, anchor, drafted + 1))

    missing = []

    def require(name, changed, required):
        moved = (changed - base).abs().amax(-1) > 1e-6
        for anchor in anchors:
            for drafted in range(block_size - 1):
                if required(anchor, drafted + 1) and not moved[anchor, drafted]:
                    missing.append((name, anchor, drafted + 1))

    for row in range(sequence):
        ids = input_ids.clone()
        ids[0, row] = (ids[0, row] + 1) % 60
        changed = logits(ids=ids)
        check(f"token {row}", changed, lambda a, k, r=row: r <= a + k - 1)
        # The token before slot k's label is that slot's teacher-forced previous token.
        require(f"token {row}", changed, lambda a, k, r=row: r == a + k - 1)
        hidden = last_hidden.clone()
        hidden[0, row] += 1.0
        changed = logits(hidden=hidden)
        check(f"last hidden {row}", changed, lambda a, k, r=row: r == a - 1)
        require(f"last hidden {row}", changed, lambda a, k, r=row: r == a - 1)
        for layer in range(len(_PAIRED_LAYERS)):
            context = [key.clone() for key in keys]
            context[layer][0, row] += 1.0
            listed = lambda a, k, r=row, i=layer: a >= 1 and bool((lists[i][0, a - 1] == r).any())  # noqa: E731
            check(f"layer {layer} key {row}", logits(context=context), listed)
            indices = [index.clone() for index in lists]
            indices[layer][0, row] = torch.roll(indices[layer][0, row], 1)
            indices[layer][0, row, 0] = -1
            check(f"layer {layer} key list {row}", logits(indices=indices), lambda a, k, r=row: r == a - 1)
    assert violations == []
    assert missing == []


def test_unused_slots_never_read_key_zero() -> None:
    from nemo_automodel.components.models.deepseek_v41.dfly2 import _listed_latent_attention

    torch.manual_seed(1)
    query = torch.randn(1, 2, 2, 8)
    kv = torch.randn(1, 5, 8)
    # Neither row lists key 0; the second lists nothing at all.
    indices = torch.tensor([[[3, -1, 1, -1], [-1, -1, -1, -1]]], dtype=torch.int32)
    sinks = torch.zeros(2)
    for backend in ("eager", "sdpa"):
        out = _listed_latent_attention(query, kv, indices, sinks, 8**-0.5, backend=backend)
        changed = kv.clone()
        changed[0, 0] += 5.0
        torch.testing.assert_close(
            _listed_latent_attention(query, changed, indices, sinks, 8**-0.5, backend=backend), out
        )
        assert out[0, 1].abs().max() == 0


def test_each_draft_layer_reads_its_own_paired_layer() -> None:
    """With draft layer 1's attention and the correction silenced, only layer 0's keys may matter."""
    torch.manual_seed(3)
    sequence = 8
    draft = _draft(num_anchors=sequence)
    draft.initialize_embeddings_and_head(embed_tokens=torch.nn.Embedding(64, 16), lm_head=torch.nn.Linear(16, 64))
    with torch.no_grad():
        draft.layers[1].attn.wo_b.weight.zero_()
        draft.hidden_correction.attention.wo_b.weight.zero_()
    input_ids = torch.randint(0, 60, (1, sequence))
    keys = [torch.randn(1, sequence, 8) for _ in _PAIRED_LAYERS]
    indices = [torch.arange(sequence).view(1, -1, 1).expand(-1, -1, 2).clone().int() for _ in _PAIRED_LAYERS]
    last_hidden = torch.randn(1, sequence, 16)

    def logits(context):
        with torch.no_grad():
            return draft(input_ids, torch.ones(1, sequence), context, indices, last_hidden).draft_logits

    base = logits(keys)
    torch.testing.assert_close(logits([keys[0], keys[1] + 1.0]), base, rtol=0, atol=0)
    assert not torch.allclose(logits([keys[0] + 1.0, keys[1]]), base)


def _consolidated_save_worker(rank: int, root: str) -> None:
    from datetime import timedelta

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
    from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
    from nemo_automodel.components.speculative.dfly2 import DFly2TrainerModule
    from nemo_automodel.recipes.speculative_decoding.train_dfly2 import TrainDFly2Recipe

    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    try:
        draft = _draft(enable_confidence_head=True)
        torch.manual_seed(13)
        draft.initialize_embeddings_and_head(embed_tokens=torch.nn.Embedding(64, 16), lm_head=torch.nn.Linear(16, 64))
        expected = {key: value.detach().clone() for key, value in draft.state_dict().items()}
        trainer = DFly2TrainerModule(draft)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
        for unit in TrainDFly2Recipe._draft_fsdp_units(draft):
            fully_shard_by_dtype(
                unit,
                mesh=mesh,
                mp_policy=policy,
                offload_policy=None,
                fp32_compute_module_names=tuple(draft._keep_in_fp32_modules_strict),
            )
        fully_shard(trainer, mesh=mesh, mp_policy=policy)
        # As the recipe builds it: native draft keys, no reference checkpoint.
        checkpointer = Checkpointer(
            CheckpointingConfig(
                checkpoint_dir=f"{root}/checkpoints",
                model_save_format="safetensors",
                save_consolidated=True,
                model_cache_dir=f"{root}/cache",
                model_repo_id="test/dfly2",
                model_state_dict_keys=list(expected),
            ),
            dp_rank=rank,
            tp_rank=0,
            pp_rank=0,
        )
        try:
            checkpointer.save_model(draft, f"{root}/saved")
        finally:
            checkpointer.close()
        dist.barrier()
        if rank == 0:
            restored = _draft(enable_confidence_head=True)
            restored.initialize_embeddings_and_head(
                embed_tokens=torch.nn.Embedding(64, 16), lm_head=torch.nn.Linear(16, 64)
            )
            load_dspark_draft_weights(
                restored,
                f"{root}/saved/model/consolidated",
                keep_target_embeddings=True,
                target_layer_ids=list(_PAIRED_LAYERS),
                mask_token_id=63,
            )
            for key, value in restored.state_dict().items():
                if key not in ("embed_tokens.weight", "lm_head.weight"):
                    torch.testing.assert_close(value, expected[key], rtol=0, atol=0, msg=key)
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(
    20, hard_timeout=70, reason="Two-process gloo FSDP2 consolidated save of the draft, then a warm start from it."
)
def test_consolidated_save_of_a_sharded_draft_holds_every_expert(tmp_path: Path) -> None:
    """Each rank exports only the experts of its FSDP shard; the consolidated draft must still hold all of them."""
    import torch.multiprocessing as mp

    mp.spawn(_consolidated_save_worker, args=(str(tmp_path),), nprocs=2, join=True)


def test_attentive_correction_starts_from_the_last_layer_closed() -> None:
    draft = _draft()
    source = draft.layers[-1].attn
    correction = draft.hidden_correction.attention
    # The hidden columns of the query and key inputs take the layer's projections; the
    # previous-embedding columns start at zero, and so does the output.
    torch.testing.assert_close(correction.q_in.weight[:, :16], source.wq_a.weight, rtol=0, atol=0)
    torch.testing.assert_close(correction.kv_in.weight[:, :16], source.wkv.weight, rtol=0, atol=0)
    assert correction.q_in.weight[:, 16:].abs().max() == 0 and correction.kv_in.weight[:, 16:].abs().max() == 0
    torch.testing.assert_close(correction.attn_sink.weight, source.sinks_param.weight, rtol=0, atol=0)
    torch.testing.assert_close(correction.block_sink.weight, source.sinks_param.weight, rtol=0, atol=0)
    assert correction.wo_b.weight.abs().max() == 0 and correction.block_gate_bias.weight.abs().max() == 0


def test_recipe_starts_each_draft_layer_from_its_paired_target_layer() -> None:
    """Eq. (9): attention (sinks included), norms, mHC mixes and the shared expert of draft layer i
    equal the tensors of paired target layer i after the recipe builds the draft."""
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes.speculative_decoding.train_dfly2 import TrainDFly2Recipe
    from nemo_automodel.recipes.speculative_decoding.train_dspark import _DraftArgs

    target = _target()
    recipe = TrainDFly2Recipe.__new__(TrainDFly2Recipe)
    recipe.cfg = ConfigNode({"recipe_args": {"draft_attn_backend": "eager", "confidence_head_stop_gradient": False}})
    recipe.dist_env = SimpleNamespace(world_size=1)
    recipe.target_model = target
    margs = _DraftArgs(
        target_layer_ids=list(_PAIRED_LAYERS),
        block_size=3,
        num_anchors=4,
        mask_token_id=63,
        confidence_head_alpha=1.0,
    )
    draft = recipe._build_model_owned_draft(target.config, margs)
    assert draft.layers[0].attn.backend.attn == "eager"
    assert draft.confidence_head is not None and draft.dfly2_config.confidence_head_stop_gradient is False
    for layer, layer_id in zip(draft.layers, _PAIRED_LAYERS):
        source = target.model.layers[str(layer_id)]
        pairs = [
            (layer.attn, source.attn, ("wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "wo_a", "wo_b", "sinks_param")),
            (layer, source, ("attn_norm", "ffn_norm", "attn_hc", "ffn_hc")),
            (layer.ffn, source.ffn, ("shared_experts",)),
        ]
        for draft_parent, target_parent, names in pairs:
            for name in names:
                theirs = getattr(target_parent, name).state_dict()
                for key, value in getattr(draft_parent, name).state_dict().items():
                    torch.testing.assert_close(value, theirs[key], rtol=0, atol=0, msg=f"{layer_id}.{name}.{key}")
