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

import re
from functools import partial

import datasets
from torch.optim import Adam
from transformers import AutoTokenizer

from nemo_lm.automodel.components.data.hf_dataset import HFDatasetBuilder
from nemo_lm.automodel.config import (
    AutoModelConfig,
    CheckpointConfig,
    ConfigContainer,
    OptimizerConfig,
    SchedulerConfig,
)
from nemo_lm.automodel.finetune import finetune
from nemo_lm.config.common import (
    LoggerConfig,
    RNGConfig,
    TrainingConfig,
)


# Ref: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hellaswag/utils.py
def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_doc(doc):
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    query = preprocess(doc["activity_label"] + ": " + ctx)
    choices = [preprocess(ending) for ending in doc["endings"]]
    gold = int(doc["label"])
    out_doc = {"query": query, "choices": choices, "gold": gold, "text": query + " " + choices[gold]}
    return out_doc


# Note: I'm training the model causally not through multiclass classification.
def preprocess_dataset(tokenizer, max_length, dataset, seed=42):
    # Format each prompt.
    print("Preprocessing dataset...")
    dataset = dataset.map(process_doc)

    def preprocess_batch(batch, tokenizer, max_length):
        ans = tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
        )
        ans["labels"] = [x[1:] + [-100] for x in ans["input_ids"]]
        ans["loss_mask"] = [[1] * (len(x) - 1) + [0] for x in ans["input_ids"]]
        return ans

    # Apply preprocessing to each batch of the dataset & and remove "conversations" and "text" fields.
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
    ).select_columns(["input_ids", "attention_mask", "labels", "loss_mask"])

    # Shuffle dataset.
    dataset = dataset.shuffle(seed=seed)

    return dataset


if __name__ == "__main__":
    global_batch_size = 256
    micro_batch_size = 4
    seq_length = 2**10

    model_name = "meta-llama/Llama-3.2-1B"
    model_config = AutoModelConfig(
        model_name=model_name,
        load_pretrained_weights=True,
    )

    # Load HellaSwag dataset (train & validation splits)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dset_name = "Rowan/hellaswag"
    dset = datasets.load_dataset(dset_name)
    dset = preprocess_dataset(tokenizer, 7500, dset["train"])
    dset = dset.shuffle(seed=42)

    dataset_config = HFDatasetBuilder(
        path_or_dataset=dset,
        split="train",
        seq_length=seq_length,
        seed=42,
        dataloader_type="cyclic",
        num_workers=0,
        use_dist_sampler=True,
        do_validation=False,
        do_test=False,
    )

    max_steps = 250
    lr = 1e-5
    wd = 0

    # Config Container
    cfg = ConfigContainer(
        model_config=model_config,
        train_config=TrainingConfig(
            train_iters=max_steps,
            eval_interval=1000,
            eval_iters=4,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            exit_signal_handler=True,
        ),
        optimizer_config=OptimizerConfig(
            optimizer_cls=Adam,
            optimizer_kwargs={
                "betas": (0.9, 0.999),
                "eps": 1e-8,
            },
            lr=lr,
            weight_decay=wd,
            min_lr=lr,
        ),
        scheduler_config=SchedulerConfig(
            start_weight_decay=wd,
            end_weight_decay=wd,
            weight_decay_incr_style="constant",
            lr_decay_style="cosine",
            lr_decay_iters=max_steps,
            lr_warmup_iters=int(0.00 * max_steps),
            lr_warmup_init=lr,
            override_opt_param_scheduler=True,
        ),
        dataset_config=dataset_config,
        logger_config=LoggerConfig(
            wandb_project="nemo_automodel_sft_loop",
            wandb_entity="nvidia",
            wandb_exp_name=f"nemolm_automodel_{dset_name.replace('/', '_')}_{model_name}_gbs_{global_batch_size}_seq_len_{seq_length}_lr_{lr}",
            wandb_save_dir="/tmp/nemo_run/wandb",
            tensorboard_dir="/tmp/nemo_run/tensorboard",
            log_timers_to_tensorboard=True,
            log_validation_ppl_to_tensorboard=True,
            tensorboard_log_interval=1,
            timing_log_level=2,
            log_progress=True,
            log_interval=1,
            logging_level="INFO",
            modules_to_filter=["nemo.collections.llm.gpt.data.utils"],
        ),
        checkpoint_config=CheckpointConfig(
            save_interval=10000,
            save=f"/tmp/nemo_run/checkpoints/automodel/{dset_name.replace('/', '_')}_{model_name}_gbs_{global_batch_size}_seq_len_{seq_length}",
            load=f"/tmp/nemo_run/checkpoints/automodel/{dset_name.replace('/', '_')}_{model_name}_gbs_{global_batch_size}_seq_len_{seq_length}",
        ),
        rng_config=RNGConfig(seed=1111),
    )
    finetune(cfg, tokenizer)
