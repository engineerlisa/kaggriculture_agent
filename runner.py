from agent_framework import (
    Plan,
    assign_tasks as default_assign_tasks,
    build_context,
    execute_assignments,
    generate_candidate_tasks,
)


def run_agent(obs, policy):
    """Run one turn with a swappable decision policy.

    Required policy functions:
      select_crop(ctx)
      select_animal(ctx, crop_to_plant)
      is_terminal_liquidation(ctx)
      rank_task(ctx, worker_index, worker_position, task)
      market_actions(ctx, plan, assignments)
    """
    ctx = build_context(obs)

    crop_to_plant = policy.select_crop(ctx)
    animal_to_add = policy.select_animal(ctx, crop_to_plant)
    plan = Plan(
        crop_to_plant=crop_to_plant,
        animal_to_add=animal_to_add,
        terminal_liquidation=policy.is_terminal_liquidation(ctx),
    )

    tasks = generate_candidate_tasks(ctx, plan)
    custom_assign_tasks = getattr(policy, "assign_tasks", None)

    if custom_assign_tasks is not None:
        assignments = custom_assign_tasks(ctx, tasks)
    else:
        assignments = default_assign_tasks(ctx, tasks, policy.rank_task)
    farmer_action, hand_actions = execute_assignments(ctx, assignments)

    return {
        "farmer": farmer_action,
        "hands": hand_actions,
        "market": policy.market_actions(ctx, plan, assignments),
    }
