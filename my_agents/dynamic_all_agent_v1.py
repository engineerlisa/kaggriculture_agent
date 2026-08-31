from dataclasses import dataclass
from typing import Optional
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, ANIMALS, MARKET_PARAMS, market_price, SHOPS, MAX_SHOP_INSTANCES


NAME = "dynamic_all_agent_v1"
TERMINAL_LIQUIDATION_HOUR = 13
FERTILIZER_COLLECTION_MIN_PRICE_RATIO = 0.5

# ============================================================
# 1. CONTEXT / ENVIRONMENT FACTS
# ============================================================
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

TASK_PRIORITY = {"deposit_product": 0,
                 "critical_feed": 0, "critical_water": 0,
                 "pickup_wheat": 1, "harvest_animal": 1, "harvest": 1,
                 "place_animal": 1,
                 "feed": 2, "water": 2,
                 "care": 3, "collect_fertilizer": 3,
                 "pickup_animal": 3, "build_structure": 3,
                 "plant": 4, "weed": 5,}

ANIMAL_PROFIT_HURDLE = 1.15
WHEAT_BUFFER_DAYS = 3
SHOP_UNLOCK_INTERVAL = 3
SHED_CAPACITY = 100
SHOP_SELL_INTERVAL = 4

UNFERTILIZED_YIELDS = {"WHEAT": 4,
                        "CARROT": 3,
                        "TOMATO": 4,
                        "STRAWBERRY": 4,
                        "MELON": 6,}

def _build_context(obs):
    player = obs["player"]

    return Context(obs=obs,
                    me=obs["farms"][player],
                    private=obs["private"],
                    day=obs["day"],
                    hour=obs["hour"],
                    market=obs["market"],
                    town=obs["town"],
                    opponent=obs["farms"][1 - player],)

def _crop_age(tile, day):
    return day - tile["planted_day"]

def _crop_yield(tile):
    return tile["yield_units"]

def _land_occupied_percentage(ctx):
    total_tiles = 0
    occupied_tiles = 0

    for row in ctx.me["tiles"]:
        for tile in row:
            if tile == "LOCKED":
                continue

            total_tiles += 1

            # Does this cause a meaningful issue related to weeds?
            if tile is not None:
                occupied_tiles += 1

    return (occupied_tiles / total_tiles) * 100 if total_tiles else 0

def _estimated_planting_batch(ctx):
    empty_tiles = sum(
        tile is None
        for row in ctx.me["tiles"]
        for tile in row)
    workers = 1 + len(ctx.me["hands"])
    return min(empty_tiles, workers)

def _projected_one_time_yield(tile, crop, day):
    crop_data = CROPS[crop]

    projected_yield = tile["yield_units"]
    age = _crop_age(tile, day)

    window_start = (crop_data["max_yield_day"] + 1) // 2

    for future_age in range(
        max(age, window_start),
        crop_data["max_yield_day"] + 1,
    ):
        future_day = tile["planted_day"] + future_age

        # If already watered today, today's bonus is already
        # included in yield_units.
        if future_day == day and tile["watered_today"]:
            continue

        bonus = 2 if tile["fertilized_until_day"] >= future_day else 1

        projected_yield = min(
            crop_data["max_yield"],
            projected_yield + bonus,
        )

        if projected_yield >= crop_data["max_yield"]:
            break

    return projected_yield

def _growing_crop_supply(farm, crop, day, hour):
    crop_data = CROPS[crop]
    total_supply = 0

    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if (not isinstance(tile, dict)
                or tile.get("kind") != "PLANT"
                or tile.get("crop") != crop):
                continue

            worker_positions = [farm["farmer"], *farm["hands"]]

            will_die_tonight = (
                hour == 23
                and not tile["watered_today"]
                and tile["consecutive_unwatered"] >= 1
                and [x, y] not in worker_positions
            )

            if will_die_tonight:
                continue

            if not crop_data["ongoing"]:
                total_supply += _projected_one_time_yield(tile, crop, day)
                continue

            age = _crop_age(tile, day)

            future_units = sum(units for production_day, units in _production_schedule(crop)
                               if production_day > age)

            total_supply += tile["yield_units"] + future_units

    return total_supply

def _committed_crop_supply(ctx, crop):
    my_shed_units = ctx.private["shed"].get(crop, 0)
    my_carried_units = sum(inventory.get(crop, 0) for inventory in ctx.private["inventories"])

    my_pipeline = _growing_crop_supply(ctx.me, crop, ctx.day, ctx.hour)

    opponent_pipeline = _growing_crop_supply(ctx.opponent, crop, ctx.day, ctx.hour)

    return (my_shed_units + my_carried_units + my_pipeline  + opponent_pipeline )

def _expected_town_consumption(ctx, crop, days):
    demand_per_day = 1  # town center

    for shop_name in ctx.town["unlocked_shops"]:
        products = SHOPS[shop_name]

        if crop in products:
            demand_per_day += 12 if len(products) == 1 else 6

    return demand_per_day * days

def _crop_profit_per_tile_day(ctx, crop, candidate_plants):
    days_to_harvest = _days_to_full_yield(crop)
    revenue = _projected_crop_revenue(ctx, crop, candidate_plants)
    seed_cost = CROPS[crop]["seed"] * candidate_plants

    expected_profit = revenue - seed_cost
    tile_days = candidate_plants * days_to_harvest

    return expected_profit / tile_days

def _choose_crop_to_plant(ctx):
    if _land_occupied_percentage(ctx) >= 100:
        return None

    candidate_plants = _estimated_planting_batch(ctx)
    if candidate_plants == 0:
        return None

    best_crop = None
    best_profit_per_tile_day = float("-inf")

    for crop in _crops_available_to_plant(ctx):
        if crop not in UNFERTILIZED_YIELDS:
            continue

        profit_per_tile_day = _crop_profit_per_tile_day(ctx, crop, candidate_plants)

        if profit_per_tile_day > best_profit_per_tile_day:
            best_profit_per_tile_day = profit_per_tile_day
            best_crop = crop

    return best_crop

def _shed_access_tiles(ctx):
    half = len(ctx.me["tiles"]) // 2
    return [(half - 1, half - 1), (half, half - 1),
            (half - 1, half), (half, half)]

def _is_terminal_liquidation(ctx):
    return ctx.day == 29 and ctx.hour >= TERMINAL_LIQUIDATION_HOUR


def _carried_units(ctx, item):
    return sum(inventory.get(item, 0) for inventory in ctx.private["inventories"])

def _same_turn_inventory_gain(worker_position, task):
    if task is None:
        return 0

    # The worker will move rather than execute the task this turn.
    if tuple(worker_position) != (task.x, task.y):
        return 0

    if task.type in ("harvest", "harvest_animal"):
        return task.tile.get("yield_units", 0)

    if task.type == "collect_fertilizer":
        return 1

    return 0


def _projected_end_of_day_overflow(ctx, farmer_task, hand_tasks):
    if ctx.hour != 23:
        return 0

    shed_units = sum(ctx.private["shed"].values())

    carried_units = sum(
        sum(inventory.values())
        for inventory in ctx.private["inventories"]
    )

    workers = [ctx.me["farmer"], *ctx.me["hands"]]
    tasks = [farmer_task, *hand_tasks]

    incoming_units = sum(
        _same_turn_inventory_gain(position, task)
        for position, task in zip(workers, tasks)
    )

    projected_total = shed_units + carried_units + incoming_units

    return max(0, projected_total - SHED_CAPACITY)

def _terminal_deposit_tasks(ctx):
    if not _is_terminal_liquidation(ctx):
        return []

    workers = [ctx.me["farmer"], *ctx.me["hands"]]
    tasks = []

    for worker_index, (position, inventory) in enumerate(
            zip(workers, ctx.private["inventories"])):

        sellable_items = [(item, amount) for item, amount in inventory.items()
                          if item in MARKET_PARAMS and amount > 0]

        if not sellable_items:
            continue

        wx, wy = position
        sx, sy = min(_shed_access_tiles(ctx),
                     key=lambda pos: _distance(wx, wy, pos[0], pos[1]))

        for item, amount in sellable_items:
            tasks.append(Task(type="deposit_product", x=sx, y=sy, item=item,
                              amount=amount, worker_index=worker_index))

    return tasks

def _distance_to_shed(ctx, x, y):
    return min(_distance(x, y, sx, sy) for sx, sy in _shed_access_tiles(ctx))


def _animal_tiles(farm):
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and "animal" in tile:
                yield x, y, tile


def _animal_count(farm):
    return sum(1 for _ in _animal_tiles(farm))


def _pending_animal(ctx):
    for animal in ANIMALS:
        if any(inventory.get(animal, 0) > 0 for inventory in ctx.private["inventories"]):
            return animal

    for animal in ANIMALS:
        if ctx.private["shed"].get(animal, 0) > 0:
            return animal

    return None


def _pending_animal_count(ctx):
    return sum(ctx.private["shed"].get(animal, 0)
               + sum(inventory.get(animal, 0) for inventory in ctx.private["inventories"])
               for animal in ANIMALS)


def _animal_added_today(ctx):
    return any(tile["placed_day"] == ctx.day for _, _, tile in _animal_tiles(ctx.me))


def _empty_animal_structure(ctx, animal):
    if animal is None:
        return None

    structure = ANIMALS[animal]["structure"]
    candidates = []

    for y, row in enumerate(ctx.me["tiles"]):
        for x, tile in enumerate(row):
            if (isinstance(tile, dict) and tile.get("kind") == structure
                    and "animal" not in tile):
                candidates.append((x, y))

    return min(candidates, key=lambda pos: _distance_to_shed(ctx, *pos)) if candidates else None


def _animal_structure_target(ctx):
    candidates = [(x, y) for y, row in enumerate(ctx.me["tiles"])
                  for x, tile in enumerate(row) if tile is None]

    return min(candidates, key=lambda pos: _distance_to_shed(ctx, *pos)) if candidates else None

def _crops_available_to_plant(ctx):
    # Only consider crops that can be planted and harvested before the end of the episode and only crops that we can afford to buy seeds for.
    crops_available = [crop for crop in CROPS.keys() if ctx.day + _days_to_full_yield(crop) < 29 
                       and ctx.me["money"] >= CROPS[crop]["seed"]*6]

    # If day == 0, remove melons, because they take too long to grow for intial day. 
    # This is a hack to avoid planting melons on day 0, which would be a bad strategy.
    if ctx.day == 0 and "MELON" in crops_available:
        crops_available.remove("MELON")

    return crops_available
    

def _projected_crop_revenue(ctx, crop, candidate_plants):
    days_to_harvest = _days_to_full_yield(crop)
    yield_per_plant = UNFERTILIZED_YIELDS[crop]

    committed_supply = _committed_crop_supply(ctx, crop)
    town_consumption = _expected_town_consumption(ctx, crop, days_to_harvest,)

    inventory_before_candidate = (ctx.market["inventory"][crop] + committed_supply
                                  - town_consumption)

    candidate_units = candidate_plants * yield_per_plant
    revenue = sum(market_price(crop, inventory_before_candidate + i)
                                for i in range(candidate_units))

    return revenue

def _production_schedule(crop):
    crop_data = CROPS[crop]

    if not crop_data["ongoing"]:
        return [(_days_to_full_yield(crop), UNFERTILIZED_YIELDS[crop])]

    return [(crop_data["first_yield_day"] + i * crop_data["interval"], 1, )
            for i in range(crop_data["max_yield"])]

def _have_seed(ctx, crop):
    return ctx.private["seeds"].get(crop, 0) > 0

def _distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def _days_to_full_yield(crop):
    crop_data = CROPS[crop]

    if crop_data["ongoing"]:
        return (crop_data["first_yield_day"]
                + crop_data["interval"] * (crop_data["max_yield"] - 1))

    if crop == "MELON":
        return 10

    return crop_data["max_yield_day"]

# ============================================================
# 2. TASK DISCOVERY
#
# Describe potentially useful actions and their relevant facts.
# Do not decide whether we WANT to do them here.
# ============================================================

def _animal_produces_tonight(ctx, tile):
    animal_data = ANIMALS[tile["animal"]]
    days_since_first = ctx.day + 1 - tile["placed_day"] - animal_data["first_yield_day"]

    return days_since_first >= 0 and days_since_first % animal_data["interval"] == 0


def _should_harvest_animal(ctx, tile):

    held_units = tile.get("yield_units", 0)

    if ctx.day == 29 and ctx.hour >= 15:
        return held_units > 0

    animal_data = ANIMALS[tile["animal"]]
    held_units = tile.get("yield_units", 0)

    if held_units <= 0:
        return False

    # Early game: realize animal revenue sooner.
    early_harvest_threshold = max(1, animal_data["max_held"] // 2)

    if ctx.day < 14 and held_units >= early_harvest_threshold:
        return True

    if held_units >= animal_data["max_held"]:
        return True

    if not _animal_produces_tonight(ctx, tile):
        return False

    incoming_units = 1 + tile.get("pending_care_bonus", 0)
    return held_units + incoming_units > animal_data["max_held"]

def _find_tasks(ctx, crop_to_plant, animal_to_add):
    tasks = []
    have_seed = _have_seed(ctx, crop_to_plant)

    pending_animal = _pending_animal(ctx)
    setup_animal = pending_animal or animal_to_add
    structure_tile = _empty_animal_structure(ctx, setup_animal)
    build_target = None

    if setup_animal is not None and structure_tile is None:
        build_target = _animal_structure_target(ctx)

    unfed_animals = 0

    for y, row in enumerate(ctx.me["tiles"]):
        for x, tile in enumerate(row):
            if tile is None:
                if build_target == (x, y):
                    tasks.append(Task(type="build_structure", x=x, y=y, animal=setup_animal))
                elif have_seed:
                    tasks.append(Task(type="plant", x=x, y=y, crop=crop_to_plant, tile=tile))
                continue

            if not isinstance(tile, dict):
                continue

            if "animal" in tile:
                animal = tile["animal"]

                if not tile["fed_today"]:
                    unfed_animals += 1
                    feed_type = "critical_feed" if tile["consecutive_unfed"] >= 1 else "feed"
                    tasks.append(Task(type=feed_type, x=x, y=y, animal=animal, tile=tile))

                if not tile["cared_today"]:
                    tasks.append(Task(type="care", x=x, y=y, animal=animal, tile=tile))

                if tile.get("fertilizer_available") and _should_collect_fertilizer(ctx):
                    tasks.append(Task(type="collect_fertilizer", x=x, y=y,
                                    animal=animal, tile=tile))

                if _should_harvest_animal(ctx, tile):
                    tasks.append(Task(type="harvest_animal", x=x, y=y,
                                      animal=animal, tile=tile))

                continue

            kind = tile.get("kind")

            if kind == "WEED":
                tasks.append(Task(type="weed", x=x, y=y, tile=tile))

            elif kind == "PLANT":
                crop = tile["crop"]

                finished_ongoing_crop = (CROPS[crop]["ongoing"]
                                         and _crop_age(tile, ctx.day) >= _days_to_full_yield(crop)
                                         and tile["yield_units"] == 0)

                if finished_ongoing_crop:
                    tasks.append(Task(type="weed", x=x, y=y, crop=crop, tile=tile))
                    continue

                if not tile["watered_today"]:
                    water_type = ("critical_water"
                                  if tile["consecutive_unwatered"] >= 1 else "water")
                    tasks.append(Task(type=water_type, x=x, y=y, crop=crop, tile=tile))

                tasks.append(Task(type="harvest", x=x, y=y, crop=crop, tile=tile))

    if pending_animal is not None and structure_tile is not None:
        carried_animal = sum(inventory.get(pending_animal, 0)
                             for inventory in ctx.private["inventories"])

        if carried_animal > 0:
            tasks.append(Task(type="place_animal", x=structure_tile[0],
                              y=structure_tile[1], animal=pending_animal))

        elif ctx.private["shed"].get(pending_animal, 0) > 0:
            shed_tile = _shed_access_tiles(ctx)[1]
            tasks.append(Task(type="pickup_animal", x=shed_tile[0],
                              y=shed_tile[1], animal=pending_animal))

    carried_wheat = sum(inventory.get("WHEAT", 0)
                        for inventory in ctx.private["inventories"])
    wheat_needed = max(0, unfed_animals - carried_wheat)
    wheat_in_shed = ctx.private["shed"].get("WHEAT", 0)

    if wheat_needed > 0 and wheat_in_shed > 0:
        shed_tile = _shed_access_tiles(ctx)[0]
        tasks.append(Task(type="pickup_wheat", x=shed_tile[0], y=shed_tile[1],
                          amount=min(wheat_needed, wheat_in_shed)))

    tasks.extend(_terminal_deposit_tasks(ctx))

    return tasks

def _future_animal_production(tile, current_day):
    animal_data = ANIMALS[tile["animal"]]
    production_day = tile["placed_day"] + animal_data["first_yield_day"]

    while production_day <= current_day:
        production_day += animal_data["interval"]

    events = {}
    first_future_production = True

    while production_day <= 29:
        if first_future_production:
            days_until_production = production_day - current_day
            bonus = tile.get("pending_care_bonus", 0) + max(0, days_until_production - 1)
            units = min(animal_data["max_held"], 1 + bonus)
            first_future_production = False
        else:
            units = min(animal_data["max_held"], 1 + animal_data["interval"])

        events[production_day] = units
        production_day += animal_data["interval"]

    return events


def _animal_product_supply(farm, product, current_day, current_hour):
    held_units = 0
    future_events = {}

    worker_positions = [farm["farmer"], *farm["hands"]]

    for x, y, tile in _animal_tiles(farm):
        animal_data = ANIMALS[tile["animal"]]

        if animal_data["product"] != product:
            continue
        will_escape_tonight = (current_hour == 23 and not tile["fed_today"]
                               and tile["consecutive_unfed"] >= 1
                               and [x, y] not in worker_positions)

        if will_escape_tonight:
            continue

        held_units += tile.get("yield_units", 0)

        for day, units in _future_animal_production(tile, current_day).items():
            future_events[day] = future_events.get(day, 0) + units

    return held_units, future_events


def _shop_product_demand_per_day(shop_name, product):
    products = SHOPS[shop_name]

    if product not in products:
        return 0

    return 12 if len(products) == 1 else 6

def _shop_consumes_product_this_turn(ctx, product):
    if ctx.hour % SHOP_SELL_INTERVAL != 0:
        return False

    return any(
        product in SHOPS[shop_name]
        for shop_name in ctx.town["unlocked_shops"]
    )

def _expected_future_shop_demand(product):
    total_demand = sum(_shop_product_demand_per_day(shop_name, product)
                       for shop_name in SHOPS)
    return total_demand / len(SHOPS)


def _future_shop_instances(ctx, day):
    remaining_slots = MAX_SHOP_INSTANCES - len(ctx.town["unlocked_shops"])
    future_unlocks = sum(unlock_day % SHOP_UNLOCK_INTERVAL == 0
                         for unlock_day in range(ctx.day + 1, day + 1))

    return min(remaining_slots, future_unlocks)


def _expected_animal_town_demand(ctx, product, day):
    demand_per_day = 1

    for shop_name in ctx.town["unlocked_shops"]:
        demand_per_day += _shop_product_demand_per_day(shop_name, product)

    future_instances = _future_shop_instances(ctx, day)
    demand_per_day += future_instances * _expected_future_shop_demand(product)

    return demand_per_day

def _projected_animal_revenue(ctx, animal):
    product = ANIMALS[animal]["product"]

    candidate_tile = {
        "animal": animal,
        "placed_day": ctx.day,
        "pending_care_bonus": 0,
    }
    candidate_events = _future_animal_production(candidate_tile, ctx.day)

    if not candidate_events:
        return 0

    my_held, my_events = _animal_product_supply(ctx.me, product, ctx.day, ctx.hour)

    opponent_held, opponent_events = _animal_product_supply(ctx.opponent, product, ctx.day, ctx.hour)

    committed_inventory = ctx.private["shed"].get(product, 0)
    committed_inventory += sum(inventory.get(product, 0)
                               for inventory in ctx.private["inventories"])

    inventory = ctx.market["inventory"][product] + committed_inventory
    inventory += my_held + opponent_held

    revenue = 0

    for day in range(ctx.day + 1, 30):
        inventory -= _expected_animal_town_demand(ctx, product, day)
        inventory += my_events.get(day, 0) + opponent_events.get(day, 0)

        for _ in range(candidate_events.get(day, 0)):
            revenue += market_price(product, inventory)
            inventory += 1

    return revenue

def _worker_inventory(ctx, worker_index):
    inventories = ctx.private["inventories"]
    return inventories[worker_index] if worker_index < len(inventories) else {}


def _worker_can_do_task(ctx, worker_index, task):
    inventory = _worker_inventory(ctx, worker_index)

    if task.worker_index is not None and task.worker_index != worker_index:
        return False

    if task.type in ("critical_feed", "feed"):
        return inventory.get("WHEAT", 0) > 0

    if task.type == "place_animal":
        return inventory.get(task.animal, 0) > 0

    return True

def _projected_feed_cost(ctx):
    feed_days = 30 - ctx.day
    market_inventory = ctx.market["inventory"]["WHEAT"]

    return sum(market_price("WHEAT", market_inventory - i - 1)
               for i in range(feed_days))


def _animal_profit_per_tile_day(ctx, animal):
    animal_data = ANIMALS[animal]

    if ctx.day + animal_data["first_yield_day"] > 29:
        return float("-inf")

    days_occupied = 30 - ctx.day
    revenue = _projected_animal_revenue(ctx, animal)
    expected_profit = revenue - animal_data["cost"] - _projected_feed_cost(ctx)

    return expected_profit / days_occupied


def _can_add_animal(ctx, animal):
    return (_empty_animal_structure(ctx, animal) is not None
            or _animal_structure_target(ctx) is not None)

def _choose_animal_to_add(ctx, crop_to_plant):
    if _pending_animal(ctx) is not None or _animal_added_today(ctx):
        return None

    candidate_plants = _estimated_planting_batch(ctx)
    crop_value = (_crop_profit_per_tile_day(ctx, crop_to_plant, candidate_plants)
                  if crop_to_plant is not None and candidate_plants > 0 else 0)

    best_animal = None
    best_value = float("-inf")

    for animal, animal_data in ANIMALS.items():
        if not _can_add_animal(ctx, animal):
            continue

        if ctx.day + animal_data["first_yield_day"] > 29:
            continue

        minimum_cash = animal_data["cost"] + WHEAT_BUFFER_DAYS * ctx.market["prices"]["WHEAT"]
        if ctx.me["money"] < minimum_cash:
            continue

        value = _animal_profit_per_tile_day(ctx, animal)

        if value > best_value:
            best_value = value
            best_animal = animal

    hurdle = max(0, crop_value) * ANIMAL_PROFIT_HURDLE
    return best_animal if best_value > hurdle else None

# ============================================================
# 3. STRATEGY
#
# 
# ============================================================

def _assign_tasks(ctx, tasks):
    workers = [
        ctx.me["farmer"],
        *ctx.me["hands"],
    ]

    available_seeds = dict(ctx.private["seeds"])

    # Build every possible worker-task pairing.
    candidates = []

    for worker_index, (worker_x, worker_y) in enumerate(workers):
        for task in tasks:

            if not _task_is_desirable(ctx, task):
                continue

            if not _worker_can_do_task(ctx, worker_index, task):
                continue

            candidates.append(
                (
                    TASK_PRIORITY[task.type],
                    _distance(worker_x, worker_y, task.x, task.y),
                    worker_index,
                    task,
                )
            )

    # Globally prefer:
    # 1. More important tasks
    # 2. Workers closer to those tasks
    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))

    assignments = [None] * len(workers)
    assigned_tiles = set()

    for _, _, worker_index, task in candidates:

        # This worker already has a task.
        if assignments[worker_index] is not None:
            continue

        # Another worker already owns this tile.
        if task.type != "deposit_product" and (task.x, task.y) in assigned_tiles:
            continue

        # Don't assign more planting tasks than we have seeds.
        if task.type == "plant":
            if available_seeds.get(task.crop, 0) <= 0:
                continue

            available_seeds[task.crop] -= 1

        assignments[worker_index] = task
        if task.type != "deposit_product":
            assigned_tiles.add((task.x, task.y))

    return assignments[0], assignments[1:]

def _should_collect_fertilizer(ctx):
    min_price = (MARKET_PARAMS["FERTILIZER"]["base"]
                 * FERTILIZER_COLLECTION_MIN_PRICE_RATIO)

    return ctx.market["prices"]["FERTILIZER"] > min_price

def _task_is_desirable(ctx, task):
    if _is_terminal_liquidation(ctx):
        if task.type == "deposit_product":
            return True

        if task.type == "harvest_animal":
            return task.tile.get("yield_units", 0) > 0

        if task.type == "harvest":
            crop_data = CROPS[task.crop]
            mature = _crop_age(task.tile, ctx.day) >= crop_data["first_yield_day"]
            return mature and task.tile["yield_units"] > 0

        if task.type == "collect_fertilizer":
            return _should_collect_fertilizer(ctx)

        return False

    if task.type == "harvest":
        return task.tile["yield_units"] >= UNFERTILIZED_YIELDS[task.crop]

    if task.type == "harvest_animal":
        return _should_harvest_animal(ctx, task.tile)

    if task.type in ("plant"):
        return ctx.hour < 23

    return True

def _wheat_owned(ctx):
    shed_wheat = ctx.private["shed"].get("WHEAT", 0)
    carried_wheat = sum(inventory.get("WHEAT", 0)
                        for inventory in ctx.private["inventories"])

    return shed_wheat + carried_wheat


def _wheat_feed_target(ctx, animal_to_add=None):
    animal_count = _animal_count(ctx.me) + _pending_animal_count(ctx)

    if animal_to_add is not None:
        animal_count += 1

    return animal_count * WHEAT_BUFFER_DAYS


def _projected_product_purchase_cost(ctx, product, quantity):
    market_inventory = ctx.market["inventory"][product]

    return sum(market_price(product, market_inventory - i - 1)
               for i in range(quantity))


def _affordable_product_purchase(ctx, product, quantity, budget):
    market_inventory = ctx.market["inventory"][product]
    units = 0
    cost = 0

    for i in range(quantity):
        price = market_price(product, market_inventory - i - 1)

        if cost + price > budget:
            break

        units += 1
        cost += price

    return units, cost


# ============================================================
# 4. EXECUTION
# ============================================================

def _execute_task(worker_position, task):
    if task is None:
        return ["PASS"]

    fx, fy = worker_position

    if (fx, fy) == (task.x, task.y):
        return _task_action(task)

    return [
        _step_toward(
            fx,
            fy,
            task.x,
            task.y,
        )
    ]


def _task_action(task):
    if task.type == "plant":
        return ["PLANT", task.crop]

    if task.type == "build_structure":
        if ANIMALS[task.animal]["structure"] == "COOP":
            return ["BUILD_COOP"]

        return ["BUILD_PASTURE"]

    if task.type == "pickup_wheat":
        return ["PICKUP", "WHEAT", task.amount]

    if task.type == "pickup_animal":
        return ["PICKUP", task.animal, 1]

    if task.type == "place_animal":
        return ["PLACE", task.animal]

    if task.type == "deposit_product":
        return ["PLACE", task.item, task.amount]

    actions = {
        "critical_feed": ["FEED"],
        "critical_water": ["WATER"],
        "harvest_animal": ["HARVEST"],
        "harvest": ["HARVEST"],
        "feed": ["FEED"],
        "water": ["WATER"],
        "care": ["CARE"],
        "collect_fertilizer": ["COLLECT_FERTILIZER"],
        "weed": ["DIG"],
    }

    return actions[task.type]


def _step_toward(fx, fy, tx, ty):
    if fx > tx:
        return "WEST"
    if fx < tx:
        return "EAST"
    if fy > ty:
        return "NORTH"
    if fy < ty:
        return "SOUTH"

    return "PASS"


# ============================================================
# 5. MARKET POLICY
#
# ============================================================

def _market_actions(ctx, crop_to_plant, animal_to_add, farmer_task, hand_tasks,):
    overflow_to_clear = _projected_end_of_day_overflow(ctx, farmer_task, hand_tasks,)
    actions = []

    seeds = ctx.private["seeds"].get(crop_to_plant, 0)

    money_available = ctx.me["money"]
    target_hires = 8
    hires_needed = (0 if ctx.hour == 23 else target_hires - ctx.me["hires_today"])

    for i in range(hires_needed):
        hire_number = ctx.me["hires_today"] + i
        cost = _hire_cost(hire_number)

        if money_available >= cost:
            actions.append(["HIRE"])
            money_available -= cost
        else:
            break

    if ctx.day == 29:
        terminal = _is_terminal_liquidation(ctx)

        for item in MARKET_PARAMS:
            available = ctx.private["shed"].get(item, 0)

            if terminal:
                available += _carried_units(ctx, item)

            if available <= 0:
                continue

            quantity = _sell_quantity(ctx, item, available)

            if quantity > 0:
                actions.append(["SELL", item, quantity])

        return actions

    animal_purchase = False

    current_wheat_target = _wheat_feed_target(ctx)
    planned_wheat_target = _wheat_feed_target(ctx, animal_to_add)
    planned_wheat_needed = max(0, planned_wheat_target - _wheat_owned(ctx))
    planned_wheat_cost = _projected_product_purchase_cost(
        ctx, "WHEAT", planned_wheat_needed)

    if ctx.hour < 23 and animal_to_add is not None:
        animal_cost = ANIMALS[animal_to_add]["cost"]

        if money_available >= animal_cost + planned_wheat_cost:
            actions.append(["BUY_ANIMAL", animal_to_add, 1])
            money_available -= animal_cost
            animal_purchase = True

    wheat_target = planned_wheat_target if animal_purchase else current_wheat_target
    wheat_needed = max(0, wheat_target - _wheat_owned(ctx))

    if ctx.hour < 23 and wheat_needed > 0:
        quantity, wheat_cost = _affordable_product_purchase(
            ctx, "WHEAT", wheat_needed, money_available)

        if quantity > 0:
            actions.append(["BUY_PRODUCT", "WHEAT", quantity])
            money_available -= wheat_cost

    if crop_to_plant is not None:
        seeds = ctx.private["seeds"].get(crop_to_plant, 0)
        seed_cost = CROPS[crop_to_plant]["seed"] * 5

        if seeds <= 2 and money_available >= seed_cost:
            actions.append(["BUY_SEED", crop_to_plant, 5])
            money_available -= seed_cost

    wheat_target = _wheat_feed_target(ctx, animal_to_add if animal_purchase else None)
    carried_wheat = sum(inventory.get("WHEAT", 0)
                        for inventory in ctx.private["inventories"])

    shed_items = list(ctx.private["shed"].items())

    if overflow_to_clear > 0:
        shed_items.sort(key=lambda item: ctx.market["prices"].get(item[0], 0), reverse=True,)

    for item_in_shed, inventory in shed_items:
        if inventory <= 0 or item_in_shed not in MARKET_PARAMS:
            continue

        available_to_sell = inventory

        if item_in_shed == "WHEAT":
            wheat_reserve = max(0, wheat_target - carried_wheat)
            available_to_sell = max(0, inventory - wheat_reserve)

        # quantity = _sell_quantity(ctx, item_in_shed, available_to_sell)

        # if quantity > 0:
        #     actions.append(["SELL", item_in_shed, quantity])
        quantity = _sell_quantity(ctx,  item_in_shed, available_to_sell,)

        # End-of-day capacity protection.
        # Selling for a poor price is still better than having the unit discarded.
        if overflow_to_clear > quantity:
            extra_needed = overflow_to_clear - quantity
            quantity = min(inventory, quantity + extra_needed,)

        overflow_to_clear = max(0,overflow_to_clear - quantity,)

        if quantity > 0:
            actions.append(["SELL", item_in_shed, quantity])


    LAND_COSTS = [1000, 2000, 4000]
    extra_quadrants = len(ctx.me["unlocked_quadrants"]) - 1
    if extra_quadrants < len(LAND_COSTS):
        land_cost = LAND_COSTS[extra_quadrants]

        if (_land_occupied_percentage(ctx) > 80 and money_available >= land_cost * 1.6
            and ctx.day < 28):
            actions.append(["BUY_LAND"])
            money_available -= land_cost

    return actions

def _sell_quantity(ctx, crop, available):
    # Final liquidation.
    if _is_terminal_liquidation(ctx):
        return available

    if crop == "FERTILIZER":
        return available

    # Preserve your emergency shed-full behavior for now.
    if sum(ctx.private["shed"].values()) == 100:
        return available

    min_price = MARKET_PARAMS[crop]["base"] * 0.83
    market_inventory = ctx.market["inventory"][crop]

    quantity = 0

    for i in range(available):
        price = market_price(crop, market_inventory + i, )

        if price <= min_price:
            break

        quantity += 1

    return quantity


# def _should_sell_crop(ctx, crop):
#     # Only sell if the price is above the base price.
#     if ctx.market["prices"][crop] > MARKET_PARAMS[crop]["base"] * 0.9:
#         return True

#     # If total crops in shed is 100 or day 29, sell them regardless of price.
#     if sum(ctx.private["shed"].values()) == 100:
#         return True

#     if ctx.day == 29 and ctx.hour >= 20:
#         return True
#     return False

def _hire_cost(hires_today):
    # The cost of hiring workers follows the Fibonacci sequence.
    a, b = 1, 1

    for _ in range(hires_today):
        a, b = b, a + b

    return a

def agent(obs):
    ctx = _build_context(obs)

    crop_to_plant = _choose_crop_to_plant(ctx)
    animal_to_add = _choose_animal_to_add(ctx, crop_to_plant)

    tasks = _find_tasks(ctx, crop_to_plant, animal_to_add)
    farmer_task, hand_tasks = _assign_tasks(ctx, tasks)

    return {
        "farmer": _execute_task(ctx.me["farmer"], farmer_task),
        "hands": [_execute_task(position, task)
                  for position, task in zip(ctx.me["hands"], hand_tasks)],
        "market": _market_actions(ctx, crop_to_plant, animal_to_add, farmer_task,hand_tasks,),
    }