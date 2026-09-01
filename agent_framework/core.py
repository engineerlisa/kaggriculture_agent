from dataclasses import dataclass
from typing import Optional

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, CROPS, MARKET_PARAMS


SHED_CAPACITY = 100


@dataclass(frozen=True)
class Task:
    type: str
    x: int
    y: int
    crop: Optional[str] = None
    animal: Optional[str] = None
    item: Optional[str] = None
    amount: int = 1
    worker_index: Optional[int] = None
    tile: Optional[dict] = None


@dataclass
class Context:
    obs: dict
    me: dict
    private: dict
    day: int
    hour: int
    market: dict
    town: dict
    opponent: dict


@dataclass(frozen=True)
class Plan:
    crop_to_plant: Optional[str]
    animal_to_add: Optional[str]
    terminal_liquidation: bool


def build_context(obs):
    player = obs["player"]
    return Context(
        obs=obs,
        me=obs["farms"][player],
        private=obs["private"],
        day=obs["day"],
        hour=obs["hour"],
        market=obs["market"],
        town=obs["town"],
        opponent=obs["farms"][1 - player],
    )


def crop_age(tile, day):
    return day - tile["planted_day"]


def land_occupied_percentage(ctx):
    total_tiles = 0
    occupied_tiles = 0

    for row in ctx.me["tiles"]:
        for tile in row:
            if tile == "LOCKED":
                continue
            total_tiles += 1
            if tile is not None:
                occupied_tiles += 1

    return (occupied_tiles / total_tiles) * 100 if total_tiles else 0


def estimated_planting_batch(ctx):
    empty_tiles = sum(
        tile is None
        for row in ctx.me["tiles"]
        for tile in row
    )
    workers = 1 + len(ctx.me["hands"])
    return min(empty_tiles, workers)


def shed_access_tiles(ctx):
    half = len(ctx.me["tiles"]) // 2
    return [
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    ]


def carried_units(ctx, item):
    return sum(
        inventory.get(item, 0)
        for inventory in ctx.private["inventories"]
    )


def distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def distance_to_shed(ctx, x, y):
    return min(
        distance(x, y, sx, sy)
        for sx, sy in shed_access_tiles(ctx)
    )


def animal_tiles(farm):
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and "animal" in tile:
                yield x, y, tile


def animal_count(farm):
    return sum(1 for _ in animal_tiles(farm))


def pending_animal(ctx):
    for animal in ANIMALS:
        if any(
            inventory.get(animal, 0) > 0
            for inventory in ctx.private["inventories"]
        ):
            return animal

    for animal in ANIMALS:
        if ctx.private["shed"].get(animal, 0) > 0:
            return animal

    return None


def pending_animal_count(ctx):
    return sum(
        ctx.private["shed"].get(animal, 0)
        + sum(
            inventory.get(animal, 0)
            for inventory in ctx.private["inventories"]
        )
        for animal in ANIMALS
    )


def animal_added_today(ctx):
    return any(
        tile["placed_day"] == ctx.day
        for _, _, tile in animal_tiles(ctx.me)
    )


def empty_animal_structure(ctx, animal):
    if animal is None:
        return None

    structure = ANIMALS[animal]["structure"]
    candidates = []

    for y, row in enumerate(ctx.me["tiles"]):
        for x, tile in enumerate(row):
            if (
                isinstance(tile, dict)
                and tile.get("kind") == structure
                and "animal" not in tile
            ):
                candidates.append((x, y))

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda pos: distance_to_shed(ctx, *pos),
    )


def animal_structure_target(ctx):
    candidates = [
        (x, y)
        for y, row in enumerate(ctx.me["tiles"])
        for x, tile in enumerate(row)
        if tile is None
    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda pos: distance_to_shed(ctx, *pos),
    )


def have_seed(ctx, crop):
    return ctx.private["seeds"].get(crop, 0) > 0


def days_to_full_yield(crop):
    crop_data = CROPS[crop]

    if crop_data["ongoing"]:
        return (
            crop_data["first_yield_day"]
            + crop_data["interval"] * (crop_data["max_yield"] - 1)
        )

    # Preserve the existing baseline's effective melon maturity rule.
    if crop == "MELON":
        return 10

    return crop_data["max_yield_day"]


def worker_inventory(ctx, worker_index):
    inventories = ctx.private["inventories"]
    return inventories[worker_index] if worker_index < len(inventories) else {}


def worker_can_do_task(ctx, worker_index, task):
    inventory = worker_inventory(ctx, worker_index)

    if task.worker_index is not None and task.worker_index != worker_index:
        return False

    if task.type in ("critical_feed", "feed"):
        return inventory.get("WHEAT", 0) > 0

    if task.type == "place_animal":
        return inventory.get(task.animal, 0) > 0

    return True


def same_turn_inventory_gain(worker_position, task):
    if task is None:
        return 0

    if tuple(worker_position) != (task.x, task.y):
        return 0

    if task.type in ("harvest", "harvest_animal"):
        return task.tile.get("yield_units", 0)

    if task.type == "collect_fertilizer":
        return 1

    return 0


def projected_end_of_day_overflow(ctx, assignments):
    if ctx.hour != 23:
        return 0

    shed_units = sum(ctx.private["shed"].values())
    carried = sum(
        sum(inventory.values())
        for inventory in ctx.private["inventories"]
    )
    workers = [ctx.me["farmer"], *ctx.me["hands"]]
    incoming = sum(
        same_turn_inventory_gain(position, task)
        for position, task in zip(workers, assignments)
    )

    return max(0, shed_units + carried + incoming - SHED_CAPACITY)


def hire_cost(hires_today):
    a, b = 1, 1
    for _ in range(hires_today):
        a, b = b, a + b
    return a


def sellable_inventory_items(inventory):
    return [
        (item, amount)
        for item, amount in inventory.items()
        if item in MARKET_PARAMS and amount > 0
    ]
