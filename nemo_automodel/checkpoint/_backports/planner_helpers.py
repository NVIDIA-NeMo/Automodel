# taken from https://github.com/pytorch/pytorch/blob/c13e725edd8dd21406c629bf625f2d6c59ceedd1/torch/distributed/checkpoint/planner_helpers.py # pylint: disable=line-too-long

from torch.distributed.checkpoint.planner import SavePlan

def _contains_usable_plan(delta_plans: list[SavePlan]) -> bool:
    """
    Check if any delta plan is usable, indicating the plan has changed.

    Args:
        delta_plans (List[SavePlan]): A list of delta plans to check.
    Returns:
        True if any delta plan is usable, False otherwise.
    """
    return any(delta_plan and delta_plan.usable for delta_plan in delta_plans)