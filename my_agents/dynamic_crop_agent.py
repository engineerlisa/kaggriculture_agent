from dataclasses import dataclass
from typing import Optional
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, ANIMALS, MARKET_PARAMS


TASK_PRIORITY = {"harvest": 0, "water": 1,  "plant": 2, "weed": 3, }
NAME = "dynamic_crop_agent"

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


def _choose_crop_to_plant(ctx):

    # If land is 100% occupied, don't plant anything.
    if _land_occupied_percentage(ctx) >= 100:
        return None

    # Choose the crop with the highest expected profit per day.
    best_crop = None
    best_profit_per_day = float("-inf")

    crops_available = _crops_available_to_plant(ctx)

    for crop in crops_available:
        if crop in UNFERTILIZED_YIELDS:
            seed_cost = CROPS[crop]["seed"]
            crop_price_today = ctx.market["prices"][crop]
            max_yield_day = _days_to_max_unfertilized_yield(crop)
            max_yield_units = UNFERTILIZED_YIELDS[crop]

            expected_profit = (crop_price_today * max_yield_units) - seed_cost
            profit_per_day = expected_profit / max_yield_day

            if profit_per_day > best_profit_per_day:
                best_profit_per_day = profit_per_day
                best_crop = crop
    return best_crop

def _crops_available_to_plant(ctx):
    # Only consider crops that can be planted and harvested before the end of the episode and only crops that we can afford to buy seeds for.

    crops_available = [crop for crop in CROPS.keys() if ctx.day + _days_to_max_unfertilized_yield(crop) < 30 
                       and ctx.me["money"] >= CROPS[crop]["seed"]*6]

    return crops_available

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
                    tasks.append(
                        Task(type="water", x=x, y=y, crop=tile["crop"], tile=tile) )

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

    available_tasks = list(tasks)
    available_seeds = dict(ctx.private["seeds"])
    assignments = []

    for worker_position in workers:
        worker_tasks = [task for task in available_tasks if (task.type != "plant" or available_seeds.get(task.crop, 0) > 0)]

        task = _choose_task(ctx, worker_position, worker_tasks)
        assignments.append(task)

        if task is None:
            continue

        # Don't send another worker to the same tile.
        available_tasks = [other for other in available_tasks if (other.x, other.y) != (task.x, task.y)]

        # Reserve the seed for this worker.
        if task.type == "plant":
            available_seeds[task.crop] -= 1

    return assignments[0], assignments[1:]

def _choose_task(ctx, worker_position, tasks):
    eligible_tasks = [task for task in tasks if _task_is_desirable(ctx, task)]

    if not eligible_tasks:
        return None

    fx, fy = worker_position

    return min(eligible_tasks,
               key=lambda task: (TASK_PRIORITY[task.type], 
                                 _distance(fx, fy, task.x, task.y),),
    )


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

    actions = { "water": ["WATER"],
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
    target_hires = 6
    hires_needed = target_hires - ctx.me["hires_today"]

    for i in range(hires_needed):
        hire_number = ctx.me["hires_today"] + i
        cost = _hire_cost(hire_number)

        if money_available >= cost:
            actions.append(["HIRE"])
            money_available -= cost
        else:
            break

    # If I have no seeds and enough money, buy 5 seeds of the crop to plant.
    if crop_to_plant is not None:
        seed_cost = CROPS[crop_to_plant]["seed"] * 5
        if seeds <= 2 and money_available >= seed_cost:
            actions.append(["BUY_SEED", crop_to_plant, 5])
            money_available -= seed_cost

    for crop_in_shed, inventory in ctx.private["shed"].items():
        if inventory > 0 and _should_sell_crop(ctx, crop_in_shed):
            actions.append(["SELL", crop_in_shed, inventory])


    # If more than 70% of the land is occupied and I have enough money, buy more land.
    if _land_occupied_percentage(ctx) > 70 and money_available >= 1000 * 1.6:
        actions.append(["BUY_LAND"])    
        money_available -= 1000

    return actions


def _should_sell_crop(ctx, crop):
    # Only sell if the price is above the base price.
    if ctx.market["prices"][crop] > MARKET_PARAMS[crop]["base"] * 0.9:
        return True

    # If total crops in shed is 100, sell them regardless of price.
    if sum(ctx.private["shed"].values()) == 100:
        return True

    return False

def _hire_cost(hires_today):
    # The cost of hiring workers follows the Fibonacci sequence.
    a, b = 1, 1

    for _ in range(hires_today):
        a, b = b, a + b

    return a