"""Gradient-boosted labor allocator on top of the frozen heuristic strategy.

Only worker/task ranking is learned. Crop/animal selection, task generation,
market behavior, task feasibility, resource reservation, and action execution
remain identical to heuristic_v2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from agent_framework import Plan, build_context, execute_assignments, generate_candidate_tasks
from agent_framework.core import distance, worker_can_do_task
from labor_ml.features import build_pair_records, feature_names
from policies import heuristic_v2 as baseline


NAME = "gbt_labor"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "labor_gbt.json"
HARD_FIRST_TASKS = {"critical_water", "critical_feed", "deposit_product"}

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        model.get_booster().set_param({"nthread": 1})
        _MODEL = model
    return _MODEL


def _eligible(ctx, worker_index, worker_position, task):
    return (
        worker_can_do_task(ctx, worker_index, task)
        and baseline.rank_task(ctx, worker_index, worker_position, task) is not None
    )


def _assign_tasks(ctx, tasks):
    workers = [ctx.me["farmer"], *ctx.me["hands"]]
    pair_records = build_pair_records(
        ctx,
        tasks,
        lambda wi, pos, task: _eligible(ctx, wi, pos, task),
    )

    if not pair_records:
        return [None] * len(workers)

    names = feature_names()
    X = np.asarray(
        [[record.features[name] for name in names] for record in pair_records],
        dtype=np.float32,
    )
    scores = _model().predict_proba(X)[:, 1]

    candidates = []
    for record, score in zip(pair_records, scores):
        task = tasks[record.task_index]
        worker_position = tuple(workers[record.worker_index])
        hard_tier = 0 if task.type in HARD_FIRST_TASKS else 1
        fallback_rank = baseline.rank_task(
            ctx,
            record.worker_index,
            worker_position,
            task,
        )
        candidates.append(
            (
                hard_tier,
                -float(score),
                fallback_rank,
                record.worker_index,
                record.task_index,
            )
        )

    candidates.sort()

    available_seeds = dict(ctx.private["seeds"])
    assignments = [None] * len(workers)
    assigned_tiles = set()

    for _, _, _, worker_index, task_index in candidates:
        if assignments[worker_index] is not None:
            continue

        task = tasks[task_index]
        if task.type != "deposit_product" and (task.x, task.y) in assigned_tiles:
            continue

        if task.type == "plant":
            if available_seeds.get(task.crop, 0) <= 0:
                continue
            available_seeds[task.crop] -= 1

        assignments[worker_index] = task
        if task.type != "deposit_product":
            assigned_tiles.add((task.x, task.y))

    return assignments


def agent(obs):
    ctx = build_context(obs)

    crop_to_plant = baseline.select_crop(ctx)
    animal_to_add = baseline.select_animal(ctx, crop_to_plant)
    plan = Plan(
        crop_to_plant=crop_to_plant,
        animal_to_add=animal_to_add,
        terminal_liquidation=baseline.is_terminal_liquidation(ctx),
    )

    tasks = generate_candidate_tasks(ctx, plan)
    assignments = _assign_tasks(ctx, tasks)
    farmer_action, hand_actions = execute_assignments(ctx, assignments)

    return {
        "farmer": farmer_action,
        "hands": hand_actions,
        "market": baseline.market_actions(ctx, plan, assignments),
    }
