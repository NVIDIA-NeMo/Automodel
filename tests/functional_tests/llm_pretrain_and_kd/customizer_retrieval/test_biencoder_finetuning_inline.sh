#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

set -exo pipefail

export SRC_DIR=${CI_PROJECT_DIR:-/app}
echo "source $SRC_DIR"

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"

# --- Dataset (inline retrieval jsonl: {query,pos_doc,neg_doc}) ---
mkdir -p /app/data/cust-1234/

DATA_SRC="$SRC_DIR/nmp/embedding_testdata/training.jsonl"
if [ ! -f "${DATA_SRC}" ]; then
    echo "Missing embedding testdata file: ${DATA_SRC}" 1>&2
    exit 1
fi
cp "${DATA_SRC}" /app/data/cust-1234/training.jsonl

# --- Base model for biencoder ---
mkdir -p /app/model-cache/llama32_1b

if [ -d /mount/models/llama32_1b ]; then
    cp -r /mount/models/llama32_1b/* /app/model-cache/llama32_1b/
elif [ -d /mount/models/llama32_1b-instruct ]; then
    # fallback: use instruct weights if base is unavailable
    cp -r /mount/models/llama32_1b-instruct/* /app/model-cache/llama32_1b/
else
    # Fallback to download (requires NGC access in the test environment).
    PATH="/app/.venv/bin:$PATH" nmp-customizer-download-model \
        -m ngc://nvidia/nemo/llama-3_2-1b:2.0 \
        -o /app/model-cache/llama32_1b
fi

# --- Output / checkpointing ---
export CKPT_DIR=/workspace/output/biencoder_inline/checkpoints
mkdir -p "${CKPT_DIR}"

CFG_PATH=/tmp/biencoder_inline_smoke.yaml
cat > "${CFG_PATH}" <<'YAML'
seed: 42

dist_env:
  backend: nccl
  timeout_minutes: 5

model:
  _target_: nemo_automodel.components.models.biencoder.NeMoAutoModelBiencoder.from_pretrained
  pretrained_model_name_or_path: /app/model-cache/llama32_1b
  share_encoder: true
  add_linear_pooler: false
  out_dimension: 768
  do_gradient_checkpointing: false
  train_n_passages: 2
  eval_negative_size: 1
  pooling: avg
  l2_normalize: true
  t: 0.02
  use_liger_kernel: false
  use_sdpa_patching: false
  torch_dtype: bfloat16

tokenizer:
  _target_: nemo_automodel._transformers.auto_tokenizer.NeMoAutoTokenizer.from_pretrained
  pretrained_model_name_or_path: /app/model-cache/llama32_1b

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  dataset:
    _target_: nemo_automodel.components.datasets.llm.retrieval_dataset_inline.make_retrieval_dataset
    data_dir_list: /app/data/cust-1234/training.jsonl
    data_type: train
    train_n_passages: 2
    eval_negative_size: 1
    seed: 42
    do_shuffle: false
    max_train_samples: 2
  collate_fn:
    _target_: nemo_automodel.components.datasets.llm.RetrievalBiencoderCollator
    q_max_len: 64
    p_max_len: 64
    query_prefix: ""
    passage_prefix: ""
    padding: longest
    pad_to_multiple_of: 8
  shuffle: false
  num_workers: 0

step_scheduler:
  global_batch_size: 1
  local_batch_size: 1
  ckpt_every_steps: 1
  val_every_steps: 1000000
  num_epochs: 1
  max_steps: 1

optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-5
  weight_decay: 0.0

checkpoint:
  enabled: true
  checkpoint_dir: /workspace/output/biencoder_inline/checkpoints
  model_save_format: safetensors
  save_consolidated: false
YAML

. /opt/venv/bin/activate

# Run the biencoder recipe (uses nemo_automodel/recipes/biencoder/train_biencoder.py via module entrypoint).
/opt/venv/bin/python -m nemo_automodel.recipes.biencoder.train_biencoder --config "${CFG_PATH}"

# Sanity check: training log exists and at least one checkpoint with safetensors was produced.
test -f "${CKPT_DIR}/training.jsonl"

/opt/venv/bin/python - <<'PY'
import glob
import os

ckpt_dir = os.environ["CKPT_DIR"]
paths = glob.glob(os.path.join(ckpt_dir, "epoch_*_step_*", "**", "*.safetensors"), recursive=True)
assert paths, f"No .safetensors checkpoints found under {ckpt_dir}"
print(f"OK: found {len(paths)} safetensors file(s)")
PY

# Compare baseline vs finetuned biencoder checkpoint (pos-neg separation should not degrade).
/opt/venv/bin/python \
    "$SRC_DIR/nmp/services/customizer-legacy/tests/python/training/nemo/e2e/compare_biencoder_models.py" \
    /app/model-cache/llama32_1b \
    "$CKPT_DIR" \
    /app/data/cust-1234/training.jsonl
