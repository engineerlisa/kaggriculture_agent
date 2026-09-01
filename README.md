# Kaggriculture baseline refactor

This is a behavior-preserving structural refactor of `dynamic_all_agent_v1.py`.
The goal is not to make the heuristic smarter. The goal is to make the
heuristic one replaceable policy on top of stable game/assignment plumbing.

## Structure

```text
main.py
runner.py
agent_framework/
    core.py
    tasks.py
policies/
    heuristic_v1.py
behavior_check.py
static_equivalence_check.py
```

### `agent_framework/core.py`

Owns state and hard facts that should not be relearned by a model:

- `Context`, `Task`, and `Plan`
- observation parsing
- geometry and shed access
- worker inventory and task feasibility
- seed/resource bookkeeping helpers
- end-of-day overflow calculation
- basic crop/animal state helpers

### `agent_framework/tasks.py`

Owns the stable worker-action pipeline:

1. generate candidate tasks for the selected plan
2. reject mechanically infeasible worker/task pairs
3. ask the policy to rank remaining pairs
4. enforce one worker per task tile and seed reservation
5. convert assignments into Kaggriculture actions

The framework does **not** decide whether harvesting, collecting fertilizer,
or planting is economically desirable. Those are policy decisions.

### `policies/heuristic_v1.py`

Contains the current baseline's actual opinions:

- crop revenue/value estimates and crop selection
- animal revenue/value estimates and animal selection
- harvest rules and fertilizer collection threshold
- task priority/ranking
- hiring, buying, selling, feed reserve, and land-purchase rules
- terminal-liquidation timing

This file intentionally preserves existing hacks and thresholds so the
structural refactor can be evaluated independently of strategy changes.

### `runner.py`

Connects a policy to the framework. The default is `policies.heuristic_v1`.
A replacement policy only needs these functions:

```python
select_crop(ctx)
select_animal(ctx, crop_to_plant)
is_terminal_liquidation(ctx)
rank_task(ctx, worker_index, worker_position, task)
market_actions(ctx, plan, assignments)
```

That supports several useful experimental surfaces without forcing an
end-to-end learned agent through the framework:

- replace crop/animal selection or valuation with model inference
- replace worker/task ranking with a learned utility model
- replace the market policy independently
- keep the heuristic policy as the adversarial baseline

An end-to-end RL policy can still bypass this architecture entirely.

## Validation

`static_equivalence_check.py` injects a small Kaggriculture API stub and
compares the original and refactored agents over synthetic observations.
It is useful in environments where `kaggle_environments` is not installed.

```bash
python static_equivalence_check.py --original /path/to/dynamic_all_agent_v1.py
```

`behavior_check.py` is the stronger check. Run it in the existing
Kaggriculture development environment to compare complete episodes against the
original agent from both player positions and fixed seeds.

```bash
python behavior_check.py --original /path/to/dynamic_all_agent_v1.py
```

The refactor should not be treated as fully behavior-verified until that real
environment check passes.

## Deliberately deferred behavioral change

The game processes at most 10 market orders per player per turn. The existing
baseline can emit more than 10, after which the environment silently drops the
rest. This refactor preserves that behavior so architecture and strategy are
not changed simultaneously.

The first follow-up change should make market-order priority explicit and cap
orders at 10, then evaluate that change independently.

## Submission

`main.py` remains the competition entry point. A multi-file Kaggle submission
can include `main.py`, `runner.py`, `agent_framework/`, and `policies/` at the
archive root.

## Evaluation pipeline

The project now includes the structured evaluation pipeline under `evaluation_pipeline/` plus a notebook-compatible `metrics.py` facade. See `EVALUATION_README.md` for usage and metric definitions.
