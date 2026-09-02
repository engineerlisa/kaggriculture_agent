from __future__ import annotations

import argparse
import csv
import gzip
import json

import numpy as np

from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS

from agent_framework import Plan, build_context, generate_candidate_tasks
from agent_framework.core import worker_can_do_task
from labor_ml.features import build_pair_records, feature_names
from policies import heuristic_v2 as baseline


EXPERTS = {"Whyme Labs", "Yuan800", "Crop Dusta"}
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


def _action_for_worker(steps, result_index: int, player: int, worker_index: int):
    action_dict = steps[result_index][player].get("action") or {}
    actions = _worker_actions(action_dict)
    return actions[worker_index] if worker_index < len(actions) else ["PASS"]


def _decision_obs(steps, result_index: int, player: int) -> dict:
    return steps[result_index - 1][player]["observation"]


def _precompute_endpoints(steps, player: int):
    """High-confidence intent labels for every (result_index, worker_index).

    Backward propagation makes this O(worker-turns), not O(worker-turns *
    remaining episode length). A movement inherits the next turn's endpoint
    only when it moves exactly one Manhattan step closer to that endpoint.
    """
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


def extract_episode(path: Path, row_sink=None):
    episode = json.loads(path.read_text())
    steps = episode["steps"]
    names = episode.get("info", {}).get("TeamNames", [])
    episode_id = str(episode.get("info", {}).get("EpisodeId") or episode.get("id") or path.stem)

    rows = [] if row_sink is None else None
    stats = Counter()
    agent_decisions = Counter()

    for player, agent_name in enumerate(names):
        if agent_name not in EXPERTS:
            continue

        endpoints = _precompute_endpoints(steps, player)

        for result_index in range(1, len(steps)):
            decision_obs = _decision_obs(steps, result_index, player)
            ctx = build_context(decision_obs)
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
                ranked = []
                for task_index, task in enumerate(tasks):
                    if not _eligible(ctx, worker_index, tuple(worker_position), task):
                        continue
                    rank = baseline.rank_task(ctx, worker_index, tuple(worker_position), task)
                    eligible_task_indices.append(task_index)
                    ranked.append((rank, task_index))
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
                negatives = [ti for ti in eligible_task_indices if ti != positive_task_index]

                # Keep hard negatives instead of materializing dozens of nearly
                # equivalent easy rows for every worker decision.
                selected_negatives = []
                seen = set()
                def add_candidates(indices):
                    for ti in indices:
                        if ti == positive_task_index or ti in seen:
                            continue
                        seen.add(ti)
                        selected_negatives.append(ti)
                        if len(selected_negatives) >= 16:
                            return True
                    return False

                add_candidates([ti for _, ti in sorted(ranked)])
                if len(selected_negatives) < 16:
                    same_type = sorted(
                        (abs(worker_position[0] - tasks[ti].x) + abs(worker_position[1] - tasks[ti].y), ti)
                        for ti in negatives
                        if tasks[ti].type == positive_task.type
                    )
                    add_candidates([ti for _, ti in same_type])
                if len(selected_negatives) < 16:
                    add_candidates(negatives)

                selected = [positive_task_index, *selected_negatives[:16]]
                selected_by_worker[worker_index] = selected
                positive_task_by_worker[worker_index] = (positive_task_index, endpoint)
                agent_decisions[agent_name] += 1
                stats["labeled_decisions"] += 1
                stats["movement_labeled_decisions"] += int(endpoint["movement_steps"] > 0)

            emit_pairs = {
                (worker_index, task_index)
                for worker_index, task_indices in selected_by_worker.items()
                for task_index in task_indices
            }
            if not emit_pairs:
                continue

            pair_records = build_pair_records(
                ctx, tasks,
                lambda wi, pos, task: _eligible(ctx, wi, pos, task),
                emit_pairs=emit_pairs,
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

    return (rows or []), stats, agent_decisions


class _NpzSink:
    def __init__(self):
        self.names = feature_names()
        self.buffers = {name: [] for name in (
            "X", "y", "weight", "episode", "agent", "group", "movement"
        )}
        self.chunks = {name: [] for name in self.buffers}
        self.group_ids = {}
        self.agent_ids = {name: i for i, name in enumerate(sorted(EXPERTS))}
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
            agent_names=np.asarray(sorted(EXPERTS)),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    total_stats = Counter()
    total_agent_decisions = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.suffix == ".npz":
        sink = _NpzSink()
        for path in sorted(args.replay_dir.glob("*.json")):
            _, stats, agent_decisions = extract_episode(path, row_sink=sink)
            total_stats.update(stats)
            total_agent_decisions.update(agent_decisions)
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
                _, stats, agent_decisions = extract_episode(path, row_sink=write_row)
                total_stats.update(stats)
                total_agent_decisions.update(agent_decisions)
                print(path.name, dict(stats), flush=True)

    print(f"wrote {row_count:,} pair rows to {args.output}")
    print("totals", dict(total_stats))
    print("labeled decisions by expert", dict(total_agent_decisions))
    print("features", len(feature_names()))


if __name__ == "__main__":
    main()
