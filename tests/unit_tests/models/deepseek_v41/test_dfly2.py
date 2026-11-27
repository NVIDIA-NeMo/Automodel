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


def test_confidence_head_reads_detached_backbone_states() -> None:
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
    assert draft.confidence_head.weight.grad is not None
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


def test_attentive_correction_starts_from_the_last_layer() -> None:
    draft = _draft()
    source = draft.layers[-1].attn
    attention = draft.hidden_correction.attention
    torch.testing.assert_close(attention.q_in.weight[:, :16], source.wq_a.weight)
    assert attention.q_in.weight[:, 16:].abs().max() == 0
    torch.testing.assert_close(attention.kv_in.weight[:, :16], source.wkv.weight)
    torch.testing.assert_close(attention.block_sink.weight, source.sinks_param.weight)
    assert attention.wo_b.weight.abs().max() == 0


def test_build_from_target_config_honors_recipe_options() -> None:
    config = DeepseekV41Config(
        vision_config=DeepseekV41VisionConfig(num_hidden_layers=0),
        text_config=_text_config(sparse_attention_backward="deterministic"),
    )
    options = SimpleNamespace(
        target_layer_ids=[4, 5],
        block_size=3,
        num_anchors=2,
        mask_token_id=63,
        enable_confidence_head=True,
        confidence_head_stop_gradient=False,
        attn_backend="eager",
    )
    draft = config.build_dfly2_draft(options)
    assert draft.dfly2_config.target_layer_ids == (4, 5)
    assert draft.confidence_head is not None
    assert len(draft.layers) == 2
    assert draft.layers[0].attn.backend.attn == "eager"
    # The target's TileLang-only backward selector is not inherited by the draft.
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


def _fsdp_training_worker(rank: int, rendezvous: str, reference_path: str) -> None:
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
        policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False)
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
        for name, parameter in trainer.named_parameters():
            if name not in reference:
                assert parameter.grad is None, name
                continue
            grad = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
            torch.testing.assert_close(grad, reference[name], rtol=1e-4, atol=1e-6, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(
    20, hard_timeout=70, reason="Two-process gloo FSDP2 forward and backward of the draft against one process."
)
def test_fsdp_gradient_matches_the_single_process_global_batch(tmp_path: Path) -> None:
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
    assert "draft_model.confidence_head.weight" in reference and "draft_model.mtp_fusion.gate" in reference
    torch.save(reference, tmp_path / "reference.pt")
    mp.spawn(
        _fsdp_training_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path / "reference.pt")), nprocs=2, join=True
    )
