from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import (ANIMALS, CROP_SEED_COSTS, EpisodeIdentity, SELLABLE_PRODUCTS,
                     base_row, farm, is_animal_tile, obs, private)


@dataclass
class StateSamples:
    land_occupancy_samples: list[float]
    shed_samples: list[int]
    crop_tile_turns: int
    animal_tile_turns: int
    weed_tile_turns: int
    structure_tile_turns: int
    shed_full_turns: int
    daily_rows: list[dict[str, Any]]


def occupied_land_counts(observation: dict[str, Any], player: int) -> dict[str, int]:
    counts = {"unlocked": 0, "occupied": 0, "crop": 0, "animal": 0, "weed": 0, "structure": 0}

    for row in farm(observation, player)["tiles"]:
        for tile in row:
            if tile == "LOCKED":
                continue

            counts["unlocked"] += 1
            if tile is None:
                continue

            counts["occupied"] += 1
            if isinstance(tile, dict):
                if tile.get("kind") == "PLANT":
                    counts["crop"] += 1
                elif tile.get("kind") == "WEED":
                    counts["weed"] += 1
                elif "animal" in tile:
                    counts["animal"] += 1
                elif tile.get("kind") in {"COOP", "PASTURE"}:
                    counts["structure"] += 1

    return counts


def private_nonseed_units(observation: dict[str, Any]) -> int:
    private_state = private(observation)
    shed = sum(private_state.get("shed", {}).values())
    carried = sum(sum(inv.values()) for inv in private_state.get("inventories", []))
    return shed + carried


def private_sellable_units(observation: dict[str, Any]) -> int:
    private_state = private(observation)
    total = sum(private_state.get("shed", {}).get(item, 0) for item in SELLABLE_PRODUCTS)

    for inventory in private_state.get("inventories", []):
        total += sum(inventory.get(item, 0) for item in SELLABLE_PRODUCTS)

    return total


def private_sellable_value_at_current_prices(observation: dict[str, Any]) -> float:
    private_state = private(observation)
    prices = observation.get("market", {}).get("prices", {})
    total = 0.0

    for item in SELLABLE_PRODUCTS:
        quantity = private_state.get("shed", {}).get(item, 0)
        quantity += sum(inv.get(item, 0) for inv in private_state.get("inventories", []))
        total += quantity * prices.get(item, 0)

    return total


def owned_animals(observation: dict[str, Any], player: int) -> int:
    private_state = private(observation)
    total = sum(private_state.get("shed", {}).get(animal, 0) for animal in ANIMALS)
    total += sum(sum(inv.get(animal, 0) for animal in ANIMALS)
                 for inv in private_state.get("inventories", []))

    for row in farm(observation, player)["tiles"]:
        for tile in row:
            if is_animal_tile(tile):
                total += 1

    return total


def daily_state_row(observation: dict[str, Any], identity: EpisodeIdentity) -> dict[str, Any]:
    player_farm = farm(observation, identity.player)
    private_state = private(observation)
    land = occupied_land_counts(observation, identity.player)

    crop_counts = {crop: 0 for crop in CROP_SEED_COSTS}
    animal_counts = {animal: 0 for animal in ANIMALS}

    for tile_row in player_farm["tiles"]:
        for tile in tile_row:
            if not isinstance(tile, dict):
                continue

            if tile.get("kind") == "PLANT" and tile.get("crop") in crop_counts:
                crop_counts[tile["crop"]] += 1

            if tile.get("animal") in animal_counts:
                animal_counts[tile["animal"]] += 1

    row = base_row(identity)
    row.update({"day": int(observation.get("day", 0)),
                "hour": int(observation.get("hour", 0)),
                "cash": float(player_farm.get("money", 0)),
                "unlocked_tiles": land["unlocked"],
                "occupied_tiles": land["occupied"],
                "land_occupancy": land["occupied"] / land["unlocked"] if land["unlocked"] else 0.0,
                "crop_tiles": land["crop"],
                "animal_tiles": land["animal"],
                "weed_tiles": land["weed"],
                "empty_structure_tiles": land["structure"],
                "shed_units": sum(private_state.get("shed", {}).values()),
                "seed_units": sum(private_state.get("seeds", {}).values()),
                "nonseed_inventory_units": private_nonseed_units(observation),
                "shops_unlocked": len(observation.get("town", {}).get("unlocked_shops", [])),
                **{f"{crop.lower()}_plants": count for crop, count in crop_counts.items()},
                **{f"{animal.lower()}_count": count for animal, count in animal_counts.items()},
                **{f"price_{item.lower()}": price
                   for item, price in observation.get("market", {}).get("prices", {}).items()},
                **{f"market_inventory_{item.lower()}": quantity
                   for item, quantity in observation.get("market", {}).get("inventory", {}).items()}})

    return row


def collect_state_samples(steps: list[list[dict[str, Any]]],
                          player: int,
                          identity: EpisodeIdentity,
                          shed_capacity: int) -> StateSamples:
    land_occupancy_samples: list[float] = []
    shed_samples: list[int] = []
    daily_rows: list[dict[str, Any]] = []
    crop_tile_turns = 0
    animal_tile_turns = 0
    weed_tile_turns = 0
    structure_tile_turns = 0
    shed_full_turns = 0
    seen_daily_state_days: set[int] = set()

    for step in steps:
        observation = obs(step, player)
        observed_day = int(observation.get("day", 0))
        observed_hour = int(observation.get("hour", 0))

        if observed_hour == 0 and observed_day not in seen_daily_state_days:
            daily_rows.append(daily_state_row(observation, identity))
            seen_daily_state_days.add(observed_day)

        land = occupied_land_counts(observation, player)
        if land["unlocked"]:
            land_occupancy_samples.append(land["occupied"] / land["unlocked"])

        crop_tile_turns += land["crop"]
        animal_tile_turns += land["animal"]
        weed_tile_turns += land["weed"]
        structure_tile_turns += land["structure"]

        shed_units = sum(private(observation).get("shed", {}).values())
        shed_samples.append(shed_units)
        if shed_units >= shed_capacity:
            shed_full_turns += 1

    return StateSamples(land_occupancy_samples=land_occupancy_samples,
                        shed_samples=shed_samples,
                        crop_tile_turns=crop_tile_turns,
                        animal_tile_turns=animal_tile_turns,
                        weed_tile_turns=weed_tile_turns,
                        structure_tile_turns=structure_tile_turns,
                        shed_full_turns=shed_full_turns,
                        daily_rows=daily_rows)
