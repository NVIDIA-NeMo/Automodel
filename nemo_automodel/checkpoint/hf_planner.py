

# taken and edited from https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/_hf_planner.py

# mypy: allow-untyped-defs
from dataclasses import dataclass, replace

from torch.distributed.checkpoint._dedup_save_plans import (
    dedup_save_plans_with_fqn_to_index_mapping,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.planner import ReadItem, SavePlan


__all__ = ["HuggingFaceSavePlanner", "HuggingFaceLoadPlanner"]


@dataclass
class _FqnToFileMapping:
    fqn_to_file_index_mapping: dict[str, int]


class HuggingFaceSavePlanner(DefaultSavePlanner):
    """
    A save planner that dedups the save plans based on the fqn to file index mapping.
    """

    def _dedup_save_plans(self, all_plans: list[SavePlan]) -> list[SavePlan]:
        assert len(all_plans) > 0, "all_plans should not be empty"
        assert all_plans[0].storage_data is not None, "storage_data should not be None"
        assert isinstance(all_plans[0].storage_data, _FqnToFileMapping), (
            "storage_data should be of type _FqnToFileMapping"
        )

        fqn_to_index_mapping: dict[str, int] = all_plans[
            0
        ].storage_data.fqn_to_file_index_mapping

        return dedup_save_plans_with_fqn_to_index_mapping(
            all_plans, fqn_to_index_mapping
        )


class HuggingFaceLoadPlanner(DefaultLoadPlanner):
    def __init__(self, allow_tensor_resize: bool = False):
        super().__init__()
        self.allow_tensor_resize = allow_tensor_resize

    def resolve_tensor(self, read_item: ReadItem):
        return self.lookup_tensor(read_item.dest_index)


from torch.distributed.checkpoint.planner import WriteItemType

# def dedup_save_plans_with_fqn_to_index_mapping(all_plans, fqn_to_index):
#     num_ranks = len(all_plans)
#     for rank, plan in enumerate(all_plans):
#         kept = []
#         for wi in plan.items:
#             if wi.type == WriteItemType.SHARD:        # keep every shard
#                 kept.append(wi)
#             else:                                     # dedup replicated items
#                 if (fqn_to_index[wi.index.fqn] - 1) % num_ranks == rank:
#                     kept.append(wi)
#         all_plans[rank] = replace(plan, items=kept)
#     return all_plans