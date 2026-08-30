from dataclasses import dataclass
from typing import Optional
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, ANIMALS, MARKET_PARAMS


TASK_PRIORITY = {"harvest": 0, "water": 1,  "plant": 2, "weed": 3, }
NAME = "wheat_agent"
CROP_TYPE = "WHEAT"

def agent(obs):
    ctx = _build_context(obs)

    tasks = _find_tasks(ctx)
    farmer_task, hand_tasks = _assign_tasks(ctx, tasks)

    return {"farmer": _execute_task(ctx.me["farmer"], farmer_task),
            "hands": [_execute_task(position, task)
                      for position, task in zip(ctx.me["hands"], hand_tasks)],
    "market": _market_actions(ctx),
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


def _have_seed(ctx, crop):
    return ctx.private["seeds"].get(crop, 0) > 0

def _distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


# ============================================================
# 2. TASK DISCOVERY
#
# Describe potentially useful actions and their relevant facts.
# Do not decide whether we WANT to do them here.
# ============================================================

def _find_tasks(ctx):
    tasks = []

    have_seed = _have_seed(ctx, CROP_TYPE)

    for y, row in enumerate(ctx.me["tiles"]):
        for x, tile in enumerate(row):

            if tile is None:
                if have_seed:
                    tasks.append(Task(type="plant", x=x,  y=y, crop=CROP_TYPE, tile=tile))
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
        return (_crop_age(task.tile, ctx.day) >= CROPS[task.crop]["max_yield_day"] and task.tile["watered_today"] == True
)
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

def _market_actions(ctx):
    actions = []

    seeds = ctx.private["seeds"].get(CROP_TYPE, 0)
    crop_in_shed = ctx.private["shed"].get(CROP_TYPE, 0)

    if len(ctx.me["hands"]) == 0 and ctx.me["money"] >= 1:
        actions.append(["HIRE"])
        actions.append(["HIRE"])
        actions.append(["HIRE"])
        actions.append(["HIRE"])
        actions.append(["HIRE"])

    if seeds == 0 and ctx.me["money"] >= 100:
        actions.append(["BUY_SEED", CROP_TYPE, 5])

    if crop_in_shed > 0 and _should_sell_crop(ctx, CROP_TYPE):
        actions.append(["SELL", CROP_TYPE, crop_in_shed])

    if _land_occupied_percentage(ctx) > 70 and ctx.me["money"] >= 1000:
        actions.append(["BUY_LAND"])    

    return actions


def _should_sell_crop(ctx, crop):
    # Only sell if the price is above the base price.
    if ctx.market["prices"][crop] > MARKET_PARAMS[crop]["base"] * 0.9:
        return True

    crop_in_shed = ctx.private["shed"].get(crop, 0)

    if crop_in_shed == 100:
        return True

    return False