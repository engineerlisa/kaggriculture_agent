from dataclasses import dataclass
from typing import Optional
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, ANIMALS, MARKET_PARAMS, market_price, SHOPS


TASK_PRIORITY = {"critical_water": 0, "harvest": 1, "water": 2,  "plant": 3, "weed": 4, }
NAME = "dynamic_crop_agent_v2"



# ============================================================
# 1. CONTEXT / ENVIRONMENT FACTS
# ============================================================


@dataclass(frozen=True)
class Task:
    type: str
    x: int
    y: int
    crop: Optional[str] = None
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

UNFERTILIZED_YIELDS = {
    "WHEAT": 4,
    "CARROT": 3,
    # "TOMATO": 4,
    # "STRAWBERRY": 4,
    "MELON": 6,
}

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

            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                occupied_tiles += 1

    return (occupied_tiles / total_tiles) * 100 if total_tiles else 0

def _estimated_planting_batch(ctx):
    empty_tiles = sum(
        tile is None
        for row in ctx.me["tiles"]
        for tile in row)
    workers = 1 + len(ctx.me["hands"])
    return min(empty_tiles, workers)

def _growing_crop_supply(farm, crop):
    growing_plants = sum(
        1
        for row in farm["tiles"]
        for tile in row
        if isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and tile.get("crop") == crop
    )

    return growing_plants * UNFERTILIZED_YIELDS[crop]

def _committed_crop_supply(ctx, crop):
    my_shed_units = ctx.private["shed"].get(crop, 0)

    my_pipeline = _growing_crop_supply(ctx.me, crop, )

    opponent_pipeline = _growing_crop_supply(ctx.opponent, crop,)

    return (my_shed_units + my_pipeline  + opponent_pipeline )

def _expected_town_consumption(ctx, crop, days):
    demand_per_day = 1  # town center

    for shop_name in ctx.town["unlocked_shops"]:
        products = SHOPS[shop_name]

        if crop in products:
            demand_per_day += 12 if len(products) == 1 else 6

    return demand_per_day * days

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

        days_to_harvest = _days_to_max_unfertilized_yield(crop)

        revenue = _projected_crop_revenue(ctx, crop, candidate_plants,)

        seed_cost = CROPS[crop]["seed"] * candidate_plants

        expected_profit = revenue - seed_cost

        tile_days = candidate_plants * days_to_harvest

        profit_per_tile_day = expected_profit / tile_days

        if profit_per_tile_day > best_profit_per_tile_day:
            best_profit_per_tile_day = profit_per_tile_day
            best_crop = crop

    return best_crop

def _crops_available_to_plant(ctx):
    # Only consider crops that can be planted and harvested before the end of the episode and only crops that we can afford to buy seeds for.
    crops_available = [crop for crop in CROPS.keys() if ctx.day + _days_to_max_unfertilized_yield(crop) < 29 
                       and ctx.me["money"] >= CROPS[crop]["seed"]*6]

    # If day == 0, remove melons, because they take too long to grow for intial day. 
    # This is a hack to avoid planting melons on day 0, which would be a bad strategy.
    if ctx.day == 0 and "MELON" in crops_available:
        crops_available.remove("MELON")

    return crops_available

def _projected_crop_revenue(ctx, crop, candidate_plants):
    days_to_harvest = _days_to_max_unfertilized_yield(crop)
    yield_per_plant = UNFERTILIZED_YIELDS[crop]

    committed_supply = _committed_crop_supply(ctx, crop)
    town_consumption = _expected_town_consumption(ctx, crop, days_to_harvest,)

    inventory_before_candidate = (ctx.market["inventory"][crop] + committed_supply
                                  - town_consumption)

    candidate_units = candidate_plants * yield_per_plant
    revenue = sum(market_price(crop, inventory_before_candidate + i)
                                for i in range(candidate_units))

    return revenue

def _have_seed(ctx, crop):
    return ctx.private["seeds"].get(crop, 0) > 0

def _distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def _days_to_max_unfertilized_yield(crop):
    if crop == "MELON":
        return 10
    return CROPS[crop]["max_yield_day"]

# ============================================================
# 2. TASK DISCOVERY
#
# Describe potentially useful actions and their relevant facts.
# Do not decide whether we WANT to do them here.
# ============================================================

def _find_tasks(ctx, crop_to_plant):
    tasks = []

    have_seed = _have_seed(ctx, crop_to_plant)

    for y, row in enumerate(ctx.me["tiles"]):
        for x, tile in enumerate(row):

            if tile is None:
                if have_seed:
                    tasks.append(Task(type="plant", x=x,  y=y, crop=crop_to_plant, tile=tile))
                continue

            if not isinstance(tile, dict):
                continue

            kind = tile.get("kind")

            if kind == "WEED":
                tasks.append( Task( type="weed", x=x, y=y, tile=tile))

            elif kind == "PLANT":

                if not tile["watered_today"]:
                    if tile["consecutive_unwatered"] >= 1:
                        water_type = "critical_water"
                    else:
                        water_type = "water"

                    tasks.append(Task(type=water_type, x=x, y=y, crop=tile["crop"], tile=tile,))

                # Discovery exposes harvesting as an option.
                # Strategy decides whether we actually want it.
                tasks.append(Task(type="harvest", x=x, y=y, crop=tile["crop"], tile=tile, ))

    return tasks


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
        if (task.x, task.y) in assigned_tiles:
            continue

        # Don't assign more planting tasks than we have seeds.
        if task.type == "plant":
            if available_seeds.get(task.crop, 0) <= 0:
                continue

            available_seeds[task.crop] -= 1

        assignments[worker_index] = task
        assigned_tiles.add((task.x, task.y))

    return assignments[0], assignments[1:]


def _task_is_desirable(ctx, task):
    if task.type == "harvest":
        # Only harvest if the crop is at max_yield_day.
        return task.tile["yield_units"] >= UNFERTILIZED_YIELDS[task.crop]

    if task.type == "plant":
        # Plant if it is not the end of the day (hour < 23). 
        return ctx.hour < 23

    return True


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

    actions = {"critical_water": ["WATER"],
               "water": ["WATER"],
               "harvest": ["HARVEST"],
               "weed": ["DIG"],}

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

def _market_actions(ctx, crop_to_plant):
    actions = []

    seeds = ctx.private["seeds"].get(crop_to_plant, 0)

    money_available = ctx.me["money"]
    target_hires = 7
    hires_needed = target_hires - ctx.me["hires_today"]

    for i in range(hires_needed):
        hire_number = ctx.me["hires_today"] + i
        cost = _hire_cost(hire_number)

        if money_available >= cost:
            actions.append(["HIRE"])
            money_available -= cost
        else:
            break

    # If I have no seeds and enough money, buy 4 seeds of the crop to plant.
    if crop_to_plant is not None:
        seed_cost = CROPS[crop_to_plant]["seed"] * 4
        if seeds <= 3 and money_available >= seed_cost:
            actions.append(["BUY_SEED", crop_to_plant, 4])
            money_available -= seed_cost

    for crop_in_shed, inventory in ctx.private["shed"].items():
        if inventory <= 0:
            continue

        quantity = _sell_quantity(ctx, crop_in_shed, inventory, )

        if quantity > 0:
            actions.append([ "SELL", crop_in_shed, quantity, ])


    LAND_COSTS = [1000, 2000, 4000]
    extra_quadrants = len(ctx.me["unlocked_quadrants"]) - 1
    if extra_quadrants < len(LAND_COSTS):
        land_cost = LAND_COSTS[extra_quadrants]

        if (_land_occupied_percentage(ctx) > 80 and money_available >= land_cost * 1.6):
            actions.append(["BUY_LAND"])
            money_available -= land_cost

    return actions

def _sell_quantity(ctx, crop, available):
    # Final liquidation.
    if ctx.day == 29 and ctx.hour >= 20:
        return available

    # Preserve your emergency shed-full behavior for now.
    if sum(ctx.private["shed"].values()) == 100:
        return available

    min_price = MARKET_PARAMS[crop]["base"] * 0.9
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

    tasks = _find_tasks(ctx, crop_to_plant)
    farmer_task, hand_tasks = _assign_tasks(ctx, tasks)

    return {"farmer": _execute_task(ctx.me["farmer"], farmer_task),
            "hands": [_execute_task(position, task)
                      for position, task in zip(ctx.me["hands"], hand_tasks)],
    "market": _market_actions(ctx, crop_to_plant),
}