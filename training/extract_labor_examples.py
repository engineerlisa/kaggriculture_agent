from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS

from agent_framework import Plan, build_context, generate_candidate_tasks
from agent_framework.core import worker_can_do_task
from labor_ml.features import build_pair_records, feature_names, task_family, task_identity
from policies import heuristic_v2 as baseline


DEFAULT_EXPERTS = {"Whyme Labs", "Yuan800", "Crop Dusta"}
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
SUPPORTED_ENDPOINTS = {
    "PLANT", "WATER", "HARVEST", "BUILD_COOP", "BUILD_PASTURE", "DIG",
    "PICKUP", "PLACE", "FEED", "COLLECT_FERTILIZER", "CARE",
}


def _workers(obs: dict, player: int) -> list[list[int]]:
    farm = obs["farms"][player]
    return [farm["farmer"], *farm["hands"]]


def _worker_actions(action_dict: dict) -> list[list]:
    return [action_dict.get("farmer", ["PASS"]), *action_dict.get("hands", [])]


def _decision_obs(steps, result_index: int, player: int) -> dict:
    return steps[result_index - 1][player]["observation"]


def _precompute_endpoints(steps, player: int):
    """High-confidence intent labels propagated backward along direct movement."""
    intents = {}
    for result_index in range(len(steps) - 1, 0, -1):
        obs = _decision_obs(steps, result_index, player)
        day = int(obs["day"])
        workers = _workers(obs, player)
        actions = _worker_actions(steps[result_index][player].get("action") or {})

        for worker_index, position in enumerate(workers):
            action = actions[worker_index] if worker_index < len(actions) else ["PASS"]
            op = action[0] if action else "PASS"

            if op in SUPPORTED_ENDPOINTS:
                intents[(result_index, worker_index)] = {
                    "result_index": result_index,
                    "x": position[0],
                    "y": position[1],
                    "action": action,
                    "movement_steps": 0,
                }
                continue

            if op not in MOVES or result_index + 1 >= len(steps):
                continue

            next_obs = _decision_obs(steps, result_index + 1, player)
            if int(next_obs["day"]) != day:
                continue
            next_workers = _workers(next_obs, player)
            if worker_index >= len(next_workers):
                continue

            next_intent = intents.get((result_index + 1, worker_index))
            if next_intent is None:
                continue

            target = (next_intent["x"], next_intent["y"])
            current_distance = abs(position[0] - target[0]) + abs(position[1] - target[1])
            next_position = next_workers[worker_index]
            next_distance = abs(next_position[0] - target[0]) + abs(next_position[1] - target[1])
            if current_distance != next_distance + 1:
                continue

            intents[(result_index, worker_index)] = {
                **next_intent,
                "movement_steps": next_intent["movement_steps"] + 1,
            }
    return intents


def _task_matches_endpoint(task, endpoint, endpoint_obs: dict, player: int) -> bool:
    if (task.x, task.y) != (endpoint["x"], endpoint["y"]):
        return False

    action = endpoint["action"]
    op = action[0]
    item = action[1] if len(action) > 1 else None

    if op == "PLANT":
        return task.type == "plant" and task.crop == item
    if op == "WATER":
        return task.type in {"water", "critical_water"}
    if op == "HARVEST":
        tile = endpoint_obs["farms"][player]["tiles"][task.y][task.x]
        if isinstance(tile, dict) and "animal" in tile:
            return task.type == "harvest_animal"
        return task.type == "harvest"
    if op == "DIG":
        return task.type == "weed"
    if op == "FEED":
        return task.type in {"feed", "critical_feed"}
    if op == "CARE":
        return task.type == "care"
    if op == "COLLECT_FERTILIZER":
        return task.type == "collect_fertilizer"
    if op in {"BUILD_COOP", "BUILD_PASTURE"}:
        if task.type != "build_structure" or task.animal is None:
            return False
        expected = "BUILD_COOP" if task.animal == "GOOSE" else "BUILD_PASTURE"
        return op == expected
    if op == "PICKUP":
        if item == "WHEAT":
            return task.type == "pickup_wheat"
        if item in ANIMALS:
            return task.type == "pickup_animal" and task.animal == item
        return False
    if op == "PLACE":
        if item in ANIMALS:
            return task.type == "place_animal" and task.animal == item
        return task.type == "deposit_product" and task.item == item
    return False


def _build_plan(ctx):
    crop = baseline.select_crop(ctx)
    animal = baseline.select_animal(ctx, crop)
    return Plan(
        crop_to_plant=crop,
        animal_to_add=animal,
        terminal_liquidation=baseline.is_terminal_liquidation(ctx),
    )


def _eligible(ctx, worker_index, worker_position, task) -> bool:
    return (
        worker_can_do_task(ctx, worker_index, task)
        and baseline.rank_task(ctx, worker_index, worker_position, task) is not None
    )


def _priority(task) -> int:
    return baseline.TASK_PRIORITY[task.type]


def extract_episode(path: Path, experts: set[str], row_sink=None):
    episode = json.loads(path.read_text())
    steps = episode["steps"]
    names = episode.get("info", {}).get("TeamNames", [])
    episode_id = str(episode.get("info", {}).get("EpisodeId") or episode.get("id") or path.stem)

    rows = [] if row_sink is None else None
    stats = Counter()
    agent_decisions = Counter()

    for player, agent_name in enumerate(names):
        if agent_name not in experts:
            continue

        endpoints = _precompute_endpoints(steps, player)
        history_by_worker: dict[int, dict] = {}
        history_day = None

        for result_index in range(1, len(steps)):
            decision_obs = _decision_obs(steps, result_index, player)
            ctx = build_context(decision_obs)
            if history_day is None or ctx.day != history_day:
                history_by_worker = {}
                history_day = ctx.day

            plan = _build_plan(ctx)
            tasks = generate_candidate_tasks(ctx, plan)
            workers = _workers(decision_obs, player)
            selected_by_worker = {}
            positive_task_by_worker = {}

            for worker_index, worker_position in enumerate(workers):
                stats["worker_decisions"] += 1
                endpoint = endpoints.get((result_index, worker_index))
                if endpoint is None:
                    stats["no_high_confidence_endpoint"] += 1
                    continue

                endpoint_obs = _decision_obs(steps, endpoint["result_index"], player)
                eligible_task_indices = []
                matching_task_indices = []
                for task_index, task in enumerate(tasks):
                    if not _eligible(ctx, worker_index, tuple(worker_position), task):
                        continue
                    eligible_task_indices.append(task_index)
                    if _task_matches_endpoint(task, endpoint, endpoint_obs, player):
                        matching_task_indices.append(task_index)

                if len(matching_task_indices) != 1:
                    if not matching_task_indices:
                        stats["endpoint_not_in_baseline_candidates"] += 1
                    else:
                        stats["ambiguous_candidate_match"] += 1
                    continue

                positive_task_index = matching_task_indices[0]
                positive_task = tasks[positive_task_index]
                positive_priority = _priority(positive_task)

                # v4 learns only the decision surface it controls at deployment:
                # alternatives in the SAME heuristic priority tier.
                negatives = [
                    ti for ti in eligible_task_indices
                    if ti != positive_task_index and _priority(tasks[ti]) == positive_priority
                ]
                if not negatives:
                    stats["no_same_priority_alternative"] += 1
                    continue

                # Up to 16 nearest same-tier hard negatives. Global task priority
                # is intentionally absent from the learned comparison.
                negatives.sort(key=lambda ti: (
                    abs(worker_position[0] - tasks[ti].x) + abs(worker_position[1] - tasks[ti].y),
                    tasks[ti].type,
                    tasks[ti].x,
                    tasks[ti].y,
                ))
                selected = [positive_task_index, *negatives[:16]]
                selected_by_worker[worker_index] = selected
                positive_task_by_worker[worker_index] = (positive_task_index, endpoint)
                agent_decisions[agent_name] += 1
                stats["labeled_decisions"] += 1
                stats["movement_labeled_decisions"] += int(endpoint["movement_steps"] > 0)
                stats[f"priority_{positive_priority}_decisions"] += 1

            emit_pairs = {
                (worker_index, task_index)
                for worker_index, task_indices in selected_by_worker.items()
                for task_index in task_indices
            }

            if emit_pairs:
                pair_records = build_pair_records(
                    ctx,
                    tasks,
                    lambda wi, pos, task: _eligible(ctx, wi, pos, task),
                    emit_pairs=emit_pairs,
                    task_priority=_priority,
                    worker_history=history_by_worker,
                )

                for record in pair_records:
                    worker_index = record.worker_index
                    positive_task_index, endpoint = positive_task_by_worker[worker_index]
                    task = tasks[record.task_index]
                    negative_count = max(1, len(selected_by_worker[worker_index]) - 1)
                    is_positive = int(record.task_index == positive_task_index)
                    row = {
                        "episode_id": episode_id,
                        "agent": agent_name,
                        "player": player,
                        "result_index": result_index,
                        "day": int(ctx.day),
                        "hour": int(ctx.hour),
                        "worker_index": worker_index,
                        "decision_id": f"{episode_id}:{player}:{result_index}:{worker_index}",
                        "target_task_type": tasks[positive_task_index].type,
                        "candidate_task_type": task.type,
                        "candidate_x": task.x,
                        "candidate_y": task.y,
                        "movement_target": int(endpoint["movement_steps"] > 0),
                        "row_weight": 1.0 if is_positive else 1.0 / negative_count,
                        "label": is_positive,
                        **record.features,
                    }
                    if row_sink is None:
                        rows.append(row)
                    else:
                        row_sink(row)

            # Build the temporal state for the NEXT turn. Use only targets that
            # were expressible in our candidate vocabulary; otherwise clear the
            # assignment rather than inventing intent.
            next_history = {}
            for worker_index, worker_position in enumerate(workers):
                previous = history_by_worker.get(worker_index)
                target_info = positive_task_by_worker.get(worker_index)
                if target_info is None:
                    next_history[worker_index] = {
                        "position": tuple(worker_position),
                        "target": None,
                        "family": None,
                        "target_xy": None,
                        "commitment_turns": 0,
                    }
                    continue

                positive_task = tasks[target_info[0]]
                identity = task_identity(positive_task)
                if previous and previous.get("target") == identity:
                    commitment = int(previous.get("commitment_turns", 0)) + 1
                else:
                    commitment = 1
                next_history[worker_index] = {
                    "position": tuple(worker_position),
                    "target": identity,
                    "family": task_family(positive_task),
                    "target_xy": (positive_task.x, positive_task.y),
                    "commitment_turns": commitment,
                }
            history_by_worker = next_history

    return (rows or []), stats, agent_decisions


class _NpzSink:
    def __init__(self, experts: set[str]):
        self.names = feature_names()
        self.experts = sorted(experts)
        self.buffers = {name: [] for name in (
            "X", "y", "weight", "episode", "agent", "group", "movement"
        )}
        self.chunks = {name: [] for name in self.buffers}
        self.group_ids = {}
        self.agent_ids = {name: i for i, name in enumerate(self.experts)}
        self.row_count = 0
        self.chunk_size = 4096

    def __call__(self, row):
        group_key = row["decision_id"]
        group_id = self.group_ids.setdefault(group_key, len(self.group_ids))
        self.buffers["X"].append([row[name] for name in self.names])
        self.buffers["y"].append(row["label"])
        self.buffers["weight"].append(row["row_weight"])
        self.buffers["episode"].append(int(row["episode_id"]))
        self.buffers["agent"].append(self.agent_ids[row["agent"]])
        self.buffers["group"].append(group_id)
        self.buffers["movement"].append(row["movement_target"])
        self.row_count += 1
        if len(self.buffers["y"]) >= self.chunk_size:
            self._flush()

    def _flush(self):
        if not self.buffers["y"]:
            return
        self.chunks["X"].append(np.asarray(self.buffers["X"], dtype=np.float32))
        self.chunks["y"].append(np.asarray(self.buffers["y"], dtype=np.uint8))
        self.chunks["weight"].append(np.asarray(self.buffers["weight"], dtype=np.float32))
        self.chunks["episode"].append(np.asarray(self.buffers["episode"], dtype=np.int64))
        self.chunks["agent"].append(np.asarray(self.buffers["agent"], dtype=np.uint8))
        self.chunks["group"].append(np.asarray(self.buffers["group"], dtype=np.int64))
        self.chunks["movement"].append(np.asarray(self.buffers["movement"], dtype=np.uint8))
        for values in self.buffers.values():
            values.clear()

    def save(self, path: Path):
        self._flush()
        arrays = {
            name: np.concatenate(chunks, axis=0)
            for name, chunks in self.chunks.items()
        }
        np.savez(
            path,
            **arrays,
            feature_names=np.asarray(self.names),
            agent_names=np.asarray(self.experts),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--experts",
        type=str,
        default=",".join(sorted(DEFAULT_EXPERTS)),
        help="Comma-separated agent names to imitate.",
    )
    args = parser.parse_args()
    experts = {name.strip() for name in args.experts.split(",") if name.strip()}

    total_stats = Counter()
    total_agent_decisions = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.suffix == ".npz":
        sink = _NpzSink(experts)
        for path in sorted(args.replay_dir.glob("*.json")):
            _, stats, agent_decisions = extract_episode(path, experts, row_sink=sink)
            total_stats.update(stats)
            total_agent_decisions.update(agent_decisions)
            if stats["worker_decisions"]:
                print(path.name, dict(stats), flush=True)
        if sink.row_count == 0:
            raise RuntimeError("No labeled training rows were extracted.")
        sink.save(args.output)
        row_count = sink.row_count
    else:
        row_count = 0
        writer = None
        opener = gzip.open if args.output.suffix == ".gz" else open
        with opener(args.output, "wt", newline="") as f:
            def write_row(row):
                nonlocal writer, row_count
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                row_count += 1

            for path in sorted(args.replay_dir.glob("*.json")):
                _, stats, agent_decisions = extract_episode(path, experts, row_sink=write_row)
                total_stats.update(stats)
                total_agent_decisions.update(agent_decisions)

    print(f"wrote {row_count:,} pair rows to {args.output}")
    print("totals", dict(total_stats))
    print("labeled decisions by expert", dict(total_agent_decisions))
    print("features", len(feature_names()))


if __name__ == "__main__":
    main()
