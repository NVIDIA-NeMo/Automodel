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