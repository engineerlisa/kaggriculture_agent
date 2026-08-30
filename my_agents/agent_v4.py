from dataclasses import dataclass
from typing import Optional


TASK_PRIORITY = {"harvest": 0, "water": 1,  "plant": 2, "weed": 3, }


@dataclass(frozen=True)
class Task:
    type: str
    x: int
    y: int
    crop: Optional[str] = None
    age: Optional[int] = None
    yield_units: Optional[int] = None
    watered_today: Optional[bool] = None


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

    have_wheat_seed = ctx.private["seeds"].get("WHEAT", 0) > 0

    for y, row in enumerate(ctx.me["tiles"]):
        for x, tile in enumerate(row):

            if tile is None:
                if have_wheat_seed:
                    tasks.append(Task(type="plant", x=x,  y=y, crop="WHEAT", ))
                continue

            if not isinstance(tile, dict):
                continue

            kind = tile.get("kind")

            if kind == "WEED":
                tasks.append( Task( type="weed", x=x, y=y,))

            elif kind == "PLANT":
                age = _crop_age(tile, ctx.day)

                if not tile["watered_today"]:
                    tasks.append(
                        Task(type="water", x=x, y=y, crop=tile["crop"], age=age, yield_units=tile["yield_units"], watered_today=False, ) )

                # Discovery exposes harvesting as an option.
                # Strategy decides whether we actually want it.
                tasks.append(
                    Task(type="harvest", x=x, y=y, crop=tile["crop"], age=age, yield_units=tile["yield_units"], watered_today=tile["watered_today"], )
                )

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
               key=lambda task: (TASK_PRIORITY[task.type], _distance(fx, fy, task.x, task.y),
        ),
    )


def _task_is_desirable(ctx, task):
    if task.type == "harvest":
        # Original strategy:
        # harvest once crop yield reaches 4.
        return task.yield_units >= 4
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

    wheat_seeds = ctx.private["seeds"].get("WHEAT", 0)
    wheat_in_shed = ctx.private["shed"].get("WHEAT", 0)

    if len(ctx.me["hands"]) == 0 and ctx.me["money"] >= 1:
        actions.append(["HIRE"])

    if wheat_seeds == 0 and ctx.me["money"] >= 30:
        actions.append(["BUY_SEED", "WHEAT", 3])

    if wheat_in_shed > 0:
        actions.append(["SELL", "WHEAT", wheat_in_shed])

    return actions