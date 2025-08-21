# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import logging

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import torch
from modelopt.torch.distill.plugins.huggingface import LMLogitsLoss

logger = logging.getLogger(__name__)


def _teacher_factory(cfg_teacher, **kwargs):
    logger.info("Instantiating teacher model ...")
    teacher_model = cfg_teacher.instantiate(**kwargs)
    logger.info("Teacher model instantiated.")

    return teacher_model


def parse_kd_config(kd_config, teacher_kwargs={}):
    kd_loss_fraction = kd_config.get("kd_loss_fraction", 1.0)
    kd_loss_temperature = kd_config.get("kd_loss_temperature", 1.0)
    teacher_config = kd_config.get("teacher_model", None)
    assert teacher_config is not None, "`teacher_model` missing from YAML kd_config section"

    modelopt_cfg = {}
    modelopt_cfg["criterion"] = LMLogitsLoss(temperature=kd_loss_temperature, reduction="none")
    if kd_loss_fraction < 1.0:
        modelopt_cfg["loss_balancer"] = mtd.StaticLossBalancer(kd_loss_weight=kd_loss_fraction)
    else:
        modelopt_cfg["loss_balancer"] = None
    modelopt_cfg["teacher_model"] = (_teacher_factory, (teacher_config,), teacher_kwargs)

    return modelopt_cfg


def kd_reduction_fn(loss: torch.Tensor, labels: torch.Tensor, num_label_tokens: int | None, ignore_index: int = -100) -> torch.Tensor:
    mask = labels.view(-1) != ignore_index
    loss = mask * loss.sum(dim=-1)
    num_label_tokens = num_label_tokens or 1
    return loss.sum() / num_label_tokens


def convert_to_kd_model(model, cfg_kd, teacher_kwargs=None):
    modelopt_cfg = parse_kd_config(cfg_kd, teacher_kwargs=teacher_kwargs)
    model = mtd.convert(model, mode=[("kd_loss", modelopt_cfg)])
    # discard KD state
    mto.ModeloptStateManager(model)._state.pop()
    return model
