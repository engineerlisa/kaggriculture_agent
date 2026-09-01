from __future__ import annotations

from typing import Any, Optional

from .common import (ANIMALS, action_parts, farm, is_animal_tile, is_empty_structure,
                     private, shed_access_tiles, tile_at, worker_inventory, workers)


MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
PRODUCTIVE_ACTIONS = {"PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP",
                      "BUILD_PASTURE", "DIG", "FEED", "COLLECT_FERTILIZER", "CARE"}
LOGISTICS_ACTIONS = {"PICKUP", "PLACE", "DROP"}


def action_category(op: str) -> str:
    if op == "PASS":
        return "idle"
    if op in MOVES:
        return "movement"
    if op in PRODUCTIVE_ACTIONS:
        return "productive"
    if op in LOGISTICS_ACTIONS:
        return "logistics"
    return "other"


def effective_worker_actions(prev_obs: dict[str, Any],
                             curr_obs: dict[str, Any],
                             player: int,
                             action_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Infer whether each submitted worker action had an effect.

    The result is intentionally conservative. `success=None` means the JSON is
    not sufficient to prove success without reimplementing more of the game
    engine. Common farm actions are resolved exactly from the pre-action state.
    """
    del curr_obs  # Reserved for transition checks that may be added later.

    worker_positions = workers(prev_obs, player)
    actions = [action_dict.get("farmer", ["PASS"]), *action_dict.get("hands", [])]
    board_size = len(farm(prev_obs, player)["tiles"])
    shed_tiles = shed_access_tiles(prev_obs, player)

    # Kaggriculture invalidates every PLANT of a crop if the turn submits more
    # plant actions for that crop than the seed inventory can cover.
    plant_counts: dict[str, int] = {}
    for action in actions:
        op, item, _ = action_parts(action)
        if op == "PLANT" and item:
            plant_counts[item] = plant_counts.get(item, 0) + 1

    available_seeds = private(prev_obs).get("seeds", {})
    claimed_effect_positions: dict[str, set[tuple[int, int]]] = {
        "PLANT": set(), "WATER": set(), "HARVEST": set(), "FERTILIZE": set(),
        "BUILD_COOP": set(), "BUILD_PASTURE": set(), "DIG": set(), "FEED": set(),
        "COLLECT_FERTILIZER": set(), "CARE": set(), "PLACE_ANIMAL": set()
    }

    # PICKUP competes for a shared shed stock. Track only the item quantity;
    # each worker can submit only one worker action per turn.
    pickup_stock = dict(private(prev_obs).get("shed", {}))

    results: list[dict[str, Any]] = []
    for worker_index, action in enumerate(actions):
        op, item, amount = action_parts(action)
        position = tuple(worker_positions[worker_index]) if worker_index < len(worker_positions) else None
        tile = tile_at(prev_obs, player, position) if position is not None else None
        inventory = worker_inventory(prev_obs, worker_index)

        success: Optional[bool] = None
        effect_amount = 0
        effect_item: Optional[str] = item
        target_kind: Optional[str] = None

        if op == "PASS":
            success = True
        elif op in MOVES:
            if position is None:
                success = False
            else:
                x, y = position
                dx, dy = {"WEST": (-1, 0), "EAST": (1, 0), "NORTH": (0, -1), "SOUTH": (0, 1)}[op]
                success = 0 <= x + dx < board_size and 0 <= y + dy < board_size
        elif op == "PLANT":
            enough_seed_for_turn = bool(item) and plant_counts.get(item, 0) <= available_seeds.get(item, 0)
            success = bool(position is not None and tile is None and enough_seed_for_turn)
            if success and position in claimed_effect_positions["PLANT"]:
                success = False
            if success:
                claimed_effect_positions["PLANT"].add(position)
                effect_amount = 1
        elif op == "WATER":
            success = bool(position is not None and isinstance(tile, dict)
                           and tile.get("kind") == "PLANT"
                           and not tile.get("watered_today", False))
            if success and position in claimed_effect_positions["WATER"]:
                success = False
            if success:
                claimed_effect_positions["WATER"].add(position)
                effect_item = tile.get("crop")
                effect_amount = 1
        elif op == "HARVEST":
            success = bool(position is not None and isinstance(tile, dict)
                           and tile.get("yield_units", 0) > 0
                           and (tile.get("kind") == "PLANT" or "animal" in tile))
            if success and position in claimed_effect_positions["HARVEST"]:
                success = False
            if success:
                claimed_effect_positions["HARVEST"].add(position)
                effect_amount = int(tile.get("yield_units", 0))
                if tile.get("kind") == "PLANT":
                    target_kind = "plant"
                    effect_item = tile.get("crop")
                else:
                    target_kind = "animal"
                    animal = tile.get("animal")
                    effect_item = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}.get(animal, animal)
        elif op == "FERTILIZE":
            success = bool(position is not None and isinstance(tile, dict)
                           and tile.get("kind") == "PLANT"
                           and inventory.get("FERTILIZER", 0) > 0)
            if success and position in claimed_effect_positions["FERTILIZE"]:
                success = False
            if success:
                claimed_effect_positions["FERTILIZE"].add(position)
                effect_item = tile.get("crop")
                effect_amount = 1
        elif op in {"BUILD_COOP", "BUILD_PASTURE"}:
            success = bool(position is not None and tile is None)
            if success and position in claimed_effect_positions[op]:
                success = False
            if success:
                claimed_effect_positions[op].add(position)
                effect_amount = 1
        elif op == "DIG":
            success = bool(position is not None and isinstance(tile, dict)
                           and (tile.get("kind") in {"PLANT", "WEED"} or is_empty_structure(tile)))
            if success and position in claimed_effect_positions["DIG"]:
                success = False
            if success:
                claimed_effect_positions["DIG"].add(position)
                effect_amount = 1
        elif op == "PICKUP":
            if position is None or position not in shed_tiles or not item:
                success = False
            else:
                available = pickup_stock.get(item, 0)
                moved = min(max(amount, 1), available)
                success = moved > 0
                effect_amount = moved
                pickup_stock[item] = available - moved
        elif op == "PLACE":
            if not item or position is None:
                success = False
            elif item in ANIMALS:
                expected_structure = "COOP" if item == "GOOSE" else "PASTURE"
                success = bool(isinstance(tile, dict)
                               and tile.get("kind") == expected_structure
                               and "animal" not in tile
                               and inventory.get(item, 0) > 0)
                if success and position in claimed_effect_positions["PLACE_ANIMAL"]:
                    success = False
                if success:
                    claimed_effect_positions["PLACE_ANIMAL"].add(position)
                    effect_amount = 1
                    target_kind = "animal"
            elif position in shed_tiles:
                available = inventory.get(item, 0)
                success = available > 0
                effect_amount = min(max(amount, 1), available)
                target_kind = "shed"
            else:
                success = False
        elif op == "DROP":
            success = bool(position is not None and position in shed_tiles and sum(inventory.values()) > 0)
            if success:
                effect_amount = sum(inventory.values())
                target_kind = "shed"
        elif op == "FEED":
            success = bool(position is not None and is_animal_tile(tile)
                           and not tile.get("fed_today", False)
                           and inventory.get("WHEAT", 0) > 0)
            if success and position in claimed_effect_positions["FEED"]:
                success = False
            if success:
                claimed_effect_positions["FEED"].add(position)
                effect_item = tile.get("animal")
                effect_amount = 1
        elif op == "COLLECT_FERTILIZER":
            success = bool(position is not None and is_animal_tile(tile)
                           and tile.get("fertilizer_available", False))
            if success and position in claimed_effect_positions["COLLECT_FERTILIZER"]:
                success = False
            if success:
                claimed_effect_positions["COLLECT_FERTILIZER"].add(position)
                effect_item = "FERTILIZER"
                effect_amount = 1
        elif op == "CARE":
            success = bool(position is not None and is_animal_tile(tile)
                           and not tile.get("cared_today", False))
            if success and position in claimed_effect_positions["CARE"]:
                success = False
            if success:
                claimed_effect_positions["CARE"].add(position)
                effect_item = tile.get("animal")
                effect_amount = 1

        results.append({"worker_index": worker_index,
                        "worker_type": "farmer" if worker_index == 0 else "hand",
                        "x": position[0] if position else None,
                        "y": position[1] if position else None,
                        "action": op,
                        "category": action_category(op),
                        "item": item,
                        "amount_requested": amount if len(action) > 2 else (1 if item is not None else None),
                        "success": success,
                        "effect_amount": effect_amount,
                        "effect_item": effect_item,
                        "target_kind": target_kind})

    return results
