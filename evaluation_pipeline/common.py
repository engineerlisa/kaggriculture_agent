from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import json


ANIMALS = {"GOOSE", "COW", "SHEEP"}

CROP_SEED_COSTS = {"WHEAT": 10,
                   "CARROT": 20,
                   "TOMATO": 50,
                   "STRAWBERRY": 100,
                   "MELON": 80}

ANIMAL_PURCHASE_COSTS = {"GOOSE": 300,
                         "COW": 400,
                         "SHEEP": 500}

ANIMAL_PRODUCTION_SCHEDULE = {"GOOSE": {"first_yield_day": 4, "interval": 1, "max_held": 4},
                              "COW": {"first_yield_day": 8, "interval": 2, "max_held": 6},
                              "SHEEP": {"first_yield_day": 6, "interval": 3, "max_held": 6}}

SELLABLE_PRODUCTS = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                     "EGG", "MILK", "WOOL", "FERTILIZER"}


@dataclass(frozen=True)
class EpisodeIdentity:
    run_id: str
    match_id: str
    episode_id: str
    source_file: str
    requested_seed: Optional[int]
    resolved_seed: Optional[int]
    player: int
    agent: str
    opponent: str
    role: str
    opponent_role: str


def obs(step: list[dict[str, Any]], player: int) -> dict[str, Any]:
    return step[player]["observation"]


def farm(observation: dict[str, Any], player: int) -> dict[str, Any]:
    return observation["farms"][player]


def private(observation: dict[str, Any]) -> dict[str, Any]:
    return observation["private"]


def workers(observation: dict[str, Any], player: int) -> list[list[int]]:
    player_farm = farm(observation, player)
    return [player_farm["farmer"], *player_farm["hands"]]


def worker_inventory(observation: dict[str, Any], worker_index: int) -> dict[str, int]:
    inventories = private(observation).get("inventories", [])
    if worker_index < len(inventories):
        return inventories[worker_index]
    return {}


def shed_access_tiles(observation: dict[str, Any], player: int) -> set[tuple[int, int]]:
    half = len(farm(observation, player)["tiles"]) // 2
    return {(half - 1, half - 1), (half, half - 1),
            (half - 1, half), (half, half)}


def tile_at(observation: dict[str, Any], player: int, pos: tuple[int, int]) -> Any:
    x, y = pos
    tiles = farm(observation, player)["tiles"]
    if y < 0 or y >= len(tiles) or x < 0 or x >= len(tiles[y]):
        return None
    return tiles[y][x]


def is_animal_tile(tile: Any) -> bool:
    return isinstance(tile, dict) and "animal" in tile


def is_empty_structure(tile: Any) -> bool:
    return (isinstance(tile, dict)
            and tile.get("kind") in {"COOP", "PASTURE"}
            and "animal" not in tile)


def action_parts(action: Any) -> tuple[str, Optional[str], int]:
    if not action:
        return "PASS", None, 0

    op = action[0]
    item = action[1] if len(action) > 1 and isinstance(action[1], str) else None
    amount = 1

    if len(action) > 2 and isinstance(action[2], (int, float)):
        amount = int(action[2])

    return op, item, amount


def base_row(identity: EpisodeIdentity) -> dict[str, Any]:
    return {"run_id": identity.run_id,
            "match_id": identity.match_id,
            "episode_id": identity.episode_id,
            "source_file": identity.source_file,
            "requested_seed": identity.requested_seed,
            "resolved_seed": identity.resolved_seed,
            "player": identity.player,
            "agent": identity.agent,
            "opponent": identity.opponent,
            "role": identity.role,
            "opponent_role": identity.opponent_role}


def event(identity: EpisodeIdentity,
          *,
          step: int,
          day: int,
          hour: int,
          event: str,
          item: Optional[str] = None,
          x: Optional[int] = None,
          y: Optional[int] = None,
          amount: Optional[float] = None,
          details: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    row = base_row(identity)
    row.update({"step": step,
                "day": day,
                "hour": hour,
                "event": event,
                "item": item,
                "x": x,
                "y": y,
                "amount": amount,
                "details": json.dumps(details, sort_keys=True) if details else None})
    return row


def hire_cost(starting_hires: int, count: int, multiplier: int = 1) -> int:
    def fib_cost(n: int) -> int:
        a, b = 1, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    return sum(multiplier * fib_cost(starting_hires + i) for i in range(count))
