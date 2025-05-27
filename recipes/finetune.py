from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from automodel.config.loader import load_yaml_config
from automodel.training.init_utils import initialize_distributed
from automodel.base_recipe import BaseRecipe


# ---------------------------
#  Stateless helper functions
# ---------------------------

def build_model(device, model_wrapper, cfg_model) -> nn.Module:
    model = cfg_model.instantiate()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.weight.requires_grad_(False)
    if model_wrapper is not None:
        model = model_wrapper.parallelize(model)
    # print(model)
    # quit()
    return model.to(device)

def build_optimizer(device, cfg_opt, model) -> Optimizer:
    return cfg_opt.instantiate(params=model.parameters())

def build_loss_fn(device, cfg_loss):
    if callable(cfg_loss):
        return cfg_loss
    else:
        return cfg_loss.instantiate().to(device)

def build_dataloader(device, cfg_ds, cfg_dl) -> DataLoader:
    ds = cfg_ds.instantiate()
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    # , num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False)
    return cfg_dl.instantiate(dataset=ds, sampler=sampler)

def build_distributed(cfg_dist: Dict[str, Any]) -> DistInfo:
    backend = cfg_dist.get("backend", "nccl")
    timeout = cfg_dist.get("timeout_minutes", 1)
    return initialize_distributed(backend=backend, timeout_minutes=timeout)

class StepScheduler:
    """
    Maintains counters and tells the trainer when to step/ckpt.
    SRP: *time-base policy* ONLY.
    """
    def __init__(self,
                 grad_acc_steps: int,
                 ckpt_every_steps: int,
                 epoch_len: Optional[int],
                 start_step: int = 0,
                 start_epoch: int = 0,
                 num_epochs: int = 0):

        self.grad_acc_steps   = grad_acc_steps
        self.ckpt_every_steps = ckpt_every_steps
        self.epoch_len        = epoch_len
        self.step   = start_step
        self.epoch  = start_epoch
        self.num_epochs = num_epochs
        # print('self.grad_acc_steps= '  +str(self.grad_acc_steps))
        # quit()

    def update(self, batch_idx: int) -> Tuple[bool, bool]:
        """Return (is_grad_step, is_ckpt_step) after incrementing step counter."""
        self.step += 1
        is_grad = (self.step % self.grad_acc_steps) == 0
        last_batch = self.epoch_len is not None and batch_idx == self.epoch_len - 1
        is_ckpt = (self.step % self.ckpt_every_steps) == 0 or last_batch
        return is_grad, is_ckpt

    # (optional) persistence
    def state_dict(self):
        return {"step": self.step, "epoch": self.epoch}
    def load_state_dict(self, s):
        self.step, self.epoch = s["step"], s["epoch"]


# ---------------------------------------------------------------------------
#  Trainer class – orchestration only
# ---------------------------------------------------------------------------

class FinetuneRecipeForNextTokenPrediction(BaseRecipe):
    """
    Orchestrates the full training life-cycle.
    wiring + loop; no low-level domain logic.
    """
    def __init__(self, cfg):
        self.cfg = cfg

    # ------------------ build phase ------------------
    def setup(self):
        """ Builds all components needed for training/validation/logging/checkpointing/etc.

        This is the last place where self.cfg should be referenced.

        Raises:
            NotImplemented: Raises if it tries to restore a checkpoint; will be removed.
        """
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        model_wrapper = None
        if 'distributed' in self.cfg:
            model_wrapper = self.cfg.distributed.instantiate(world_size=self.dist_env.world_size)
            print(model_wrapper)
        torch.manual_seed(self.cfg.get("seed", 42) + self.dist_env.rank)

        # Build components
        self.model = build_model(self.dist_env.device, model_wrapper, self.cfg.model)
        # quit()
        self.optimizer = build_optimizer(self.dist_env.device, self.cfg.optimizer, self.model)
        self.loss_fn   = build_loss_fn(self.dist_env.device, self.cfg.loss_fn)
        self.dataloader = build_dataloader(self.dist_env.device, self.cfg.dataset, self.cfg.dataloader)

        # Scheduler
        self.scheduler = StepScheduler(
            num_epochs = self.cfg.get("epochs", 10),
            grad_acc_steps   = self.cfg.get("grad_acc_steps", 10),
            ckpt_every_steps = self.cfg.get("ckpt_every_steps", 100),
            epoch_len        = len(self.dataloader),
        )

        # Optionally resume
        if path := self.cfg.get("restore_from"):
            raise NotImplemented("TODO")

    # ------------------ main loop ------------------
    def run_train_validation_loop(self):
        self.model.train()
        for self.scheduler.epoch in range(self.scheduler.epoch, self.scheduler.num_epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                is_grad, is_ckpt = self.scheduler.update(batch_idx)
                loss = self._run_train_step(batch, is_grad)
                # if self.dist_env.is_main and is_ckpt:
                #     self._save_checkpoint()
                if self.dist_env.is_main and is_grad:
                    print(f"step {self.scheduler.step} | loss {loss.item():.6f}", flush=True)


    # ------------------ helpers ------------------
    def _run_train_step(self, batch, is_grad):
        batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("labels")
        mask   = batch.pop("loss_mask", None)

        out  = self.model(**batch)
        # print(batch)
        # quit()
        loss = self.loss_fn(out.logits.view(-1, out.logits.size(-1)),
                            labels.view(-1), mask=mask)
        loss.backward()

        if is_grad:
            #         if self.cfg.training.get("calculate_per_token_loss", False):
            # world_size = get_world_size_safe()
            # num_tokens_for_grad_scaling = self.total_num_tokens.clone().detach()
            # dist.all_reduce(num_tokens_for_grad_scaling)
                # DDP reduces across ranks, so we need to scale by the world size to inverse it
            scaling_factor = 109 #world_size / num_tokens_for_grad_scaling
            # print('scaling_factor= ' + str(scaling_factor))
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(scaling_factor)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, foreach=True)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.detach()

    def _save_checkpoint(self):
        path = self.cfg.get("ckpt_path", "latest.pt")
        for key in self.__dict__['__state_tracked']:
            torch.save(getattr(self, key).state_dict(),
                path + "_key"
            )
        print(f"[ckpt] saved to {path}", flush=True)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg = load_yaml_config("llama_3_2_1b_hellaswag.yaml")
    trainer = FinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

if __name__ == "__main__":
    main()
