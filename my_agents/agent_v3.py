from dataclasses import dataclass
from typing import Optional


TASK_PRIORITY = {
    "harvest": 0,
    "water": 1,
    "weed": 2,
    "plant": 3,
}


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
    task = _choose_task(ctx, tasks)

    return {
        "farmer": _execute_task(ctx, task),
        "hands": [],
        "market": _market_actions(ctx),
    }


# ============================================================
# 1. CONTEXT / ENVIRONMENT FACTS
# ============================================================

def _build_context(obs):
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


def _crop_age(tile, day):
    return day - tile["planted_day"]


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
                    tasks.append(
                        Task(
                            type="plant",
                            x=x,
                            y=y,
                            crop="WHEAT",
                        )
                    )
                continue

            if not isinstance(tile, dict):
                continue

            kind = tile.get("kind")

            if kind == "WEED":
                tasks.append(
                    Task(
                        type="weed",
                        x=x,
                        y=y,
                    )
                )

            elif kind == "PLANT":
                age = _crop_age(tile, ctx.day)

                if not tile["watered_today"]:
                    tasks.append(
                        Task(
                            type="water",
                            x=x,
                            y=y,
                            crop=tile["crop"],
                            age=age,
                            yield_units=tile["yield_units"],
                            watered_today=False,
                        )
                    )

                # Discovery exposes harvesting as an option.
                # Strategy decides whether we actually want it.
                tasks.append(
                    Task(
                        type="harvest",
                        x=x,
                        y=y,
                        crop=tile["crop"],
                        age=age,
                        yield_units=tile["yield_units"],
                        watered_today=tile["watered_today"],
                    )
                )

    return tasks


# ============================================================
# 3. STRATEGY
#
# Preserve the original strategy exactly.
# ============================================================

def _choose_task(ctx, tasks):
    eligible_tasks = [
        task
        for task in tasks
        if _task_is_desirable(task)
    ]

    if not eligible_tasks:
        return None

    fx, fy = ctx.me["farmer"]

    return min(
        eligible_tasks,
        key=lambda task: (
            TASK_PRIORITY[task.type],
            _distance(fx, fy, task.x, task.y),
        ),
    )


def _task_is_desirable(task):
    if task.type == "harvest":
        # Original strategy:
        # harvest once crop age reaches 3.
        return task.age >= 3

    return True


# ============================================================
# 4. EXECUTION
# ============================================================

def _execute_task(ctx, task):
    if task is None:
        return ["PASS"]

    fx, fy = ctx.me["farmer"]

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

    actions = {
        "water": ["WATER"],
        "harvest": ["HARVEST"],
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
# Original strategy unchanged.
# ============================================================

def _market_actions(ctx):
    actions = []

    wheat_seeds = ctx.private["seeds"].get("WHEAT", 0)
    wheat_in_shed = ctx.private["shed"].get("WHEAT", 0)

    if wheat_seeds == 0 and ctx.me["money"] >= 10:
        actions.append(["BUY_SEED", "WHEAT", 1])

    if wheat_in_shed > 0:
        actions.append(["SELL", "WHEAT", wheat_in_shed])

    return actions