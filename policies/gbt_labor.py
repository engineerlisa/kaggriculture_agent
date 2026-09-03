"""Gradient-boosted labor allocator on top of the frozen heuristic strategy.

The heuristic owns task priority and every non-labor decision. The GBT ranks
worker/task pairings only within a heuristic priority tier. v4 also exposes
same-day assignment continuity to the model so it can learn not to abandon a
trip merely because the relational geometry changed one turn later.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from xgboost import XGBClassifier

from agent_framework.core import worker_can_do_task
from labor_ml.features import (
    build_pair_records,
    feature_names,
    task_family,
    task_identity,
)
from policies import heuristic_v2 as baseline
from runner import run_agent


NAME = "gbt_labor"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "labor_gbt.json"
_MODEL = None
_HISTORY = {
    "player": None,
    "day": None,
    "last_step": None,
    "workers": {},
}

# Reuse the frozen heuristic for every non-labor decision surface.
select_crop = baseline.select_crop
select_animal = baseline.select_animal
is_terminal_liquidation = baseline.is_terminal_liquidation
market_actions = baseline.market_actions


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


def _priority(task):
    return baseline.TASK_PRIORITY[task.type]


def _history_for_turn(ctx):
    """Return prior same-day state, resetting across episodes and day rollover."""
    global _HISTORY
    step = int(ctx.obs.get("step", ctx.day * 24 + ctx.hour))
    player = int(ctx.obs.get("player", 0))

    new_episode = (
        _HISTORY["last_step"] is None
        or _HISTORY["player"] != player
        or step <= _HISTORY["last_step"]
    )
    new_day = _HISTORY["day"] is not None and int(ctx.day) != _HISTORY["day"]

    if new_episode or new_day:
        _HISTORY = {
            "player": player,
            "day": int(ctx.day),
            "last_step": None,
            "workers": {},
        }

    return _HISTORY["workers"]


def _store_history(ctx, workers, assignments):
    global _HISTORY
    previous_workers = _HISTORY.get("workers", {})
    next_workers = {}

    for worker_index, worker_position in enumerate(workers):
        task = assignments[worker_index] if worker_index < len(assignments) else None
        previous = previous_workers.get(worker_index)

        if task is None:
            next_workers[worker_index] = {
                "position": tuple(worker_position),
                "target": None,
                "family": None,
                "target_xy": None,
                "commitment_turns": 0,
            }
            continue

        identity = task_identity(task)
        if previous and previous.get("target") == identity:
            commitment = int(previous.get("commitment_turns", 0)) + 1
        else:
            commitment = 1

        next_workers[worker_index] = {
            "position": tuple(worker_position),
            "target": identity,
            "family": task_family(task),
            "target_xy": (task.x, task.y),
            "commitment_turns": commitment,
        }

    _HISTORY = {
        "player": int(ctx.obs.get("player", 0)),
        "day": int(ctx.day),
        "last_step": int(ctx.obs.get("step", ctx.day * 24 + ctx.hour)),
        "workers": next_workers,
    }


def assign_tasks(ctx, tasks):
    """Rank worker/task pairings within frozen heuristic task-priority tiers."""
    workers = [ctx.me["farmer"], *ctx.me["hands"]]
    worker_history = _history_for_turn(ctx)

    pair_records = build_pair_records(
        ctx,
        tasks,
        lambda wi, pos, task: _eligible(ctx, wi, pos, task),
        task_priority=_priority,
        worker_history=worker_history,
    )

    if not pair_records:
        assignments = [None] * len(workers)
        _store_history(ctx, workers, assignments)
        return assignments

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
        fallback_rank = baseline.rank_task(
            ctx,
            record.worker_index,
            worker_position,
            task,
        )
        candidates.append(
            (
                _priority(task),
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

    _store_history(ctx, workers, assignments)
    return assignments


def agent(obs):
    return run_agent(obs, sys.modules[__name__])
