from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, CROPS

from agent_framework.core import distance, distance_to_shed, worker_inventory


TASK_TYPES = (
    "deposit_product",
    "critical_feed",
    "critical_water",
    "pickup_wheat",
    "harvest_animal",
    "harvest",
    "place_animal",
    "pickup_animal",
    "feed",
    "water",
    "care",
    "collect_fertilizer",
    "build_structure",
    "plant",
    "weed",
)
CROPS_ORDER = tuple(CROPS)
ANIMALS_ORDER = tuple(ANIMALS)


@dataclass(frozen=True)
class PairRecord:
    worker_index: int
    task_index: int
    features: dict[str, float]


def task_family(task) -> str:
    if task.type in {"water", "critical_water"}:
        return "water"
    if task.type in {"feed", "critical_feed"}:
        return "feed"
    return task.type


def task_identity(task) -> tuple:
    """Stable target identity across urgency transitions such as water -> critical_water."""
    return (
        task_family(task),
        int(task.x),
        int(task.y),
        task.crop,
        task.animal,
        task.item,
    )


def _tile_features(ctx, task) -> dict[str, float]:
    tile = task.tile if isinstance(task.tile, dict) else {}
    crop = task.crop
    animal = task.animal

    crop_age = 0
    if crop is not None and "planted_day" in tile:
        crop_age = ctx.day - int(tile["planted_day"])

    animal_age = 0
    if animal is not None and "placed_day" in tile:
        animal_age = ctx.day - int(tile["placed_day"])

    fertilized_until = int(tile.get("fertilized_until_day", -1))

    return {
        "crop_age": float(crop_age),
        "animal_age": float(animal_age),
        "yield_units": float(tile.get("yield_units", 0) or 0),
        "watered_today": float(bool(tile.get("watered_today", False))),
        "consecutive_unwatered": float(tile.get("consecutive_unwatered", 0) or 0),
        "fertilized_days_remaining": float(max(0, fertilized_until - ctx.day + 1)),
        "fed_today": float(bool(tile.get("fed_today", False))),
        "consecutive_unfed": float(tile.get("consecutive_unfed", 0) or 0),
        "cared_today": float(bool(tile.get("cared_today", False))),
        "pending_care_bonus": float(tile.get("pending_care_bonus", 0) or 0),
        "fertilizer_available": float(bool(tile.get("fertilizer_available", False))),
    }


def _history_features(worker_position, task, history) -> dict[str, float]:
    if not history:
        return {
            "has_previous_assignment": 0.0,
            "same_as_previous_target": 0.0,
            "same_tile_as_previous_target": 0.0,
            "same_task_family_as_previous_target": 0.0,
            "previous_commitment_turns": 0.0,
            "progress_since_previous_turn": 0.0,
            "last_move_toward_candidate": 0.0,
            "last_move_away_from_candidate": 0.0,
            "distance_to_previous_target": 20.0,
            "candidate_distance_minus_previous_target": 0.0,
        }

    previous_position = history.get("position")
    previous_target = history.get("target")
    previous_family = history.get("family")
    previous_xy = history.get("target_xy")

    pair_distance = distance(worker_position[0], worker_position[1], task.x, task.y)
    progress = 0.0
    if previous_position is not None:
        previous_candidate_distance = distance(
            previous_position[0], previous_position[1], task.x, task.y
        )
        progress = float(previous_candidate_distance - pair_distance)

    has_previous = previous_target is not None
    same_target = has_previous and task_identity(task) == previous_target
    same_tile = has_previous and previous_xy == (task.x, task.y)
    same_family = has_previous and previous_family == task_family(task)

    if previous_xy is not None:
        previous_target_distance = distance(
            worker_position[0], worker_position[1], previous_xy[0], previous_xy[1]
        )
    else:
        previous_target_distance = 20

    return {
        "has_previous_assignment": float(has_previous),
        "same_as_previous_target": float(same_target),
        "same_tile_as_previous_target": float(same_tile),
        "same_task_family_as_previous_target": float(same_family),
        "previous_commitment_turns": float(history.get("commitment_turns", 0) or 0),
        "progress_since_previous_turn": progress,
        "last_move_toward_candidate": float(progress > 0),
        "last_move_away_from_candidate": float(progress < 0),
        "distance_to_previous_target": float(previous_target_distance),
        "candidate_distance_minus_previous_target": float(pair_distance - previous_target_distance),
    }


def build_pair_records(
    ctx,
    tasks,
    eligible: Callable[[int, tuple[int, int], object], bool],
    emit_pairs: set[tuple[int, int]] | None = None,
    *,
    task_priority: Callable[[object], int] | None = None,
    worker_history: dict[int, dict] | None = None,
) -> list[PairRecord]:
    """Build feature rows for eligible worker/task pairs.

    Worker-side comparative geometry is computed within the candidate task's
    heuristic priority tier, matching deployment where the GBT can rank only
    inside a tier. Temporal features describe the worker's previous assignment
    and movement during the same day.
    """
    workers = [ctx.me["farmer"], *ctx.me["hands"]]
    if not workers or not tasks:
        return []

    if task_priority is None:
        task_priority = lambda task: 0
    worker_history = worker_history or {}

    distances = [
        [distance(wx, wy, task.x, task.y) for task in tasks]
        for wx, wy in workers
    ]

    eligible_pairs: list[tuple[int, int]] = []
    by_worker: dict[int, list[int]] = defaultdict(list)
    by_worker_priority: dict[tuple[int, int], list[int]] = defaultdict(list)
    by_task: dict[int, list[int]] = defaultdict(list)

    for worker_index, worker_position in enumerate(workers):
        position = tuple(worker_position)
        for task_index, task in enumerate(tasks):
            if not eligible(worker_index, position, task):
                continue
            eligible_pairs.append((worker_index, task_index))
            by_worker[worker_index].append(task_index)
            by_worker_priority[(worker_index, task_priority(task))].append(task_index)
            by_task[task_index].append(worker_index)

    if not eligible_pairs:
        return []

    worker_priority_sorted = {}
    for key, task_indices in by_worker_priority.items():
        worker_index, _ = key
        worker_priority_sorted[key] = sorted(
            (distances[worker_index][task_index], task_index)
            for task_index in task_indices
        )

    task_sorted = {}
    for task_index, worker_indices in by_task.items():
        task_sorted[task_index] = sorted(
            (distances[worker_index][task_index], worker_index)
            for worker_index in worker_indices
        )

    task_type_counts = Counter(tasks[task_index].type for _, task_index in eligible_pairs)
    priority_task_sets: dict[int, set[int]] = defaultdict(set)
    for _, task_index in eligible_pairs:
        priority_task_sets[task_priority(tasks[task_index])].add(task_index)

    critical_task_count = sum(
        task.type in {"critical_water", "critical_feed"}
        for task in tasks
    )

    records: list[PairRecord] = []
    for worker_index, task_index in eligible_pairs:
        if emit_pairs is not None and (worker_index, task_index) not in emit_pairs:
            continue

        worker_x, worker_y = workers[worker_index]
        worker_position = (worker_x, worker_y)
        task = tasks[task_index]
        priority = task_priority(task)
        pair_distance = distances[worker_index][task_index]

        sorted_workers = task_sorted[task_index]
        best_worker_distance = sorted_workers[0][0]
        nearest_other_worker_distance = pair_distance + 10
        for d, other_index in sorted_workers:
            if other_index != worker_index:
                nearest_other_worker_distance = d
                break
        num_workers_closer = sum(d < pair_distance for d, _ in sorted_workers)

        # These worker-side alternatives are deliberately restricted to the
        # same heuristic priority tier, because lower/higher tiers cannot beat
        # this task on GBT score at deployment.
        sorted_tasks = worker_priority_sorted[(worker_index, priority)]
        best_task_distance = sorted_tasks[0][0]
        nearest_other_task_distance = pair_distance + 10
        for d, other_task_index in sorted_tasks:
            if other_task_index != task_index:
                nearest_other_task_distance = d
                break

        inventory = worker_inventory(ctx, worker_index)
        total_inventory = sum(inventory.values())
        carried_animal = any(inventory.get(animal, 0) > 0 for animal in ANIMALS)
        hours_remaining = 24 - ctx.hour

        features: dict[str, float] = {
            "distance": float(pair_distance),
            "on_task": float(pair_distance == 0),
            "hours_remaining": float(hours_remaining),
            "travel_action_slack": float(hours_remaining - pair_distance - 1),
            "worker_distance_to_shed": float(distance_to_shed(ctx, worker_x, worker_y)),
            "task_distance_to_shed": float(distance_to_shed(ctx, task.x, task.y)),
            "nearest_other_worker_distance": float(nearest_other_worker_distance),
            "is_nearest_worker": float(pair_distance == best_worker_distance),
            "num_workers_closer": float(num_workers_closer),
            "distance_from_best_worker": float(pair_distance - best_worker_distance),
            "nearest_other_task_distance": float(nearest_other_task_distance),
            "distance_from_worker_best_task": float(pair_distance - best_task_distance),
            "worker_count": float(len(workers)),
            "candidate_task_count": float(len(tasks)),
            "eligible_task_count_for_worker": float(len(by_worker[worker_index])),
            "eligible_worker_count_for_task": float(len(by_task[task_index])),
            "eligible_same_priority_task_count_for_worker": float(len(sorted_tasks)),
            "same_priority_task_count": float(len(priority_task_sets[priority])),
            "same_type_pair_count": float(task_type_counts[task.type]),
            "critical_task_count": float(critical_task_count),
            "task_amount": float(task.amount or 0),
            "worker_wheat": float(inventory.get("WHEAT", 0)),
            "worker_fertilizer": float(inventory.get("FERTILIZER", 0)),
            "worker_inventory_units": float(total_inventory),
            "worker_carries_animal": float(carried_animal),
            **_history_features(worker_position, task, worker_history.get(worker_index)),
            **_tile_features(ctx, task),
        }

        for task_type in TASK_TYPES:
            features[f"task__{task_type}"] = float(task.type == task_type)
        for crop in CROPS_ORDER:
            features[f"crop__{crop}"] = float(task.crop == crop)
        for animal in ANIMALS_ORDER:
            features[f"animal__{animal}"] = float(task.animal == animal)

        records.append(PairRecord(worker_index, task_index, features))

    return records


def feature_names() -> list[str]:
    base = [
        "distance", "on_task", "hours_remaining", "travel_action_slack",
        "worker_distance_to_shed", "task_distance_to_shed",
        "nearest_other_worker_distance", "is_nearest_worker", "num_workers_closer",
        "distance_from_best_worker", "nearest_other_task_distance",
        "distance_from_worker_best_task", "worker_count", "candidate_task_count",
        "eligible_task_count_for_worker", "eligible_worker_count_for_task",
        "eligible_same_priority_task_count_for_worker", "same_priority_task_count",
        "same_type_pair_count", "critical_task_count", "task_amount",
        "worker_wheat", "worker_fertilizer", "worker_inventory_units",
        "worker_carries_animal",
        "has_previous_assignment", "same_as_previous_target",
        "same_tile_as_previous_target", "same_task_family_as_previous_target",
        "previous_commitment_turns", "progress_since_previous_turn",
        "last_move_toward_candidate", "last_move_away_from_candidate",
        "distance_to_previous_target", "candidate_distance_minus_previous_target",
        "crop_age", "animal_age", "yield_units", "watered_today",
        "consecutive_unwatered", "fertilized_days_remaining", "fed_today",
        "consecutive_unfed", "cared_today", "pending_care_bonus",
        "fertilizer_available",
    ]
    return (
        base
        + [f"task__{task_type}" for task_type in TASK_TYPES]
        + [f"crop__{crop}" for crop in CROPS_ORDER]
        + [f"animal__{animal}" for animal in ANIMALS_ORDER]
    )
