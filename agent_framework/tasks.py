from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, CROPS

from .core import (
    Task,
    animal_structure_target,
    crop_age,
    days_to_full_yield,
    distance,
    empty_animal_structure,
    have_seed,
    pending_animal,
    shed_access_tiles,
    sellable_inventory_items,
    worker_can_do_task,
)


def generate_candidate_tasks(ctx, plan):
    """Describe mechanically possible work for the selected high-level plan.

    Strategic desirability is intentionally deferred to the policy ranker.
    """
    tasks = []
    crop_to_plant = plan.crop_to_plant
    animal_to_add = plan.animal_to_add
    can_plant_selected_crop = have_seed(ctx, crop_to_plant)

    current_pending_animal = pending_animal(ctx)
    setup_animal = current_pending_animal or animal_to_add
    structure_tile = empty_animal_structure(ctx, setup_animal)
    build_target = None

    if setup_animal is not None and structure_tile is None:
        build_target = animal_structure_target(ctx)

    unfed_animals = 0

    for y, row in enumerate(ctx.me["tiles"]):
        for x, tile in enumerate(row):
            if tile is None:
                if build_target == (x, y):
                    tasks.append(
                        Task(
                            type="build_structure",
                            x=x,
                            y=y,
                            animal=setup_animal,
                        )
                    )
                elif can_plant_selected_crop:
                    tasks.append(
                        Task(
                            type="plant",
                            x=x,
                            y=y,
                            crop=crop_to_plant,
                            tile=tile,
                        )
                    )
                continue

            if not isinstance(tile, dict):
                continue

            if "animal" in tile:
                animal = tile["animal"]

                if not tile["fed_today"]:
                    unfed_animals += 1
                    feed_type = (
                        "critical_feed"
                        if tile["consecutive_unfed"] >= 1
                        else "feed"
                    )
                    tasks.append(
                        Task(
                            type=feed_type,
                            x=x,
                            y=y,
                            animal=animal,
                            tile=tile,
                        )
                    )

                if not tile["cared_today"]:
                    tasks.append(
                        Task(
                            type="care",
                            x=x,
                            y=y,
                            animal=animal,
                            tile=tile,
                        )
                    )

                if tile.get("fertilizer_available"):
                    tasks.append(
                        Task(
                            type="collect_fertilizer",
                            x=x,
                            y=y,
                            animal=animal,
                            tile=tile,
                        )
                    )

                # Harvest eligibility is a policy decision. Candidate discovery
                # exposes the action whenever there is product to harvest.
                if tile.get("yield_units", 0) > 0:
                    tasks.append(
                        Task(
                            type="harvest_animal",
                            x=x,
                            y=y,
                            animal=animal,
                            tile=tile,
                        )
                    )

                continue

            kind = tile.get("kind")

            if kind == "WEED":
                tasks.append(Task(type="weed", x=x, y=y, tile=tile))
                continue

            if kind != "PLANT":
                continue

            crop = tile["crop"]
            finished_ongoing_crop = (
                CROPS[crop]["ongoing"]
                and crop_age(tile, ctx.day) >= days_to_full_yield(crop)
                and tile["yield_units"] == 0
            )

            if finished_ongoing_crop:
                tasks.append(
                    Task(type="weed", x=x, y=y, crop=crop, tile=tile)
                )
                continue

            if not tile["watered_today"]:
                water_type = (
                    "critical_water"
                    if tile["consecutive_unwatered"] >= 1
                    else "water"
                )
                tasks.append(
                    Task(
                        type=water_type,
                        x=x,
                        y=y,
                        crop=crop,
                        tile=tile,
                    )
                )

            tasks.append(
                Task(
                    type="harvest",
                    x=x,
                    y=y,
                    crop=crop,
                    tile=tile,
                )
            )

    if current_pending_animal is not None and structure_tile is not None:
        carried_animal = sum(
            inventory.get(current_pending_animal, 0)
            for inventory in ctx.private["inventories"]
        )

        if carried_animal > 0:
            tasks.append(
                Task(
                    type="place_animal",
                    x=structure_tile[0],
                    y=structure_tile[1],
                    animal=current_pending_animal,
                )
            )
        elif ctx.private["shed"].get(current_pending_animal, 0) > 0:
            shed_tile = shed_access_tiles(ctx)[1]
            tasks.append(
                Task(
                    type="pickup_animal",
                    x=shed_tile[0],
                    y=shed_tile[1],
                    animal=current_pending_animal,
                )
            )

    carried_wheat = sum(
        inventory.get("WHEAT", 0)
        for inventory in ctx.private["inventories"]
    )
    wheat_needed = max(0, unfed_animals - carried_wheat)
    wheat_in_shed = ctx.private["shed"].get("WHEAT", 0)

    if wheat_needed > 0 and wheat_in_shed > 0:
        shed_tile = shed_access_tiles(ctx)[0]
        tasks.append(
            Task(
                type="pickup_wheat",
                x=shed_tile[0],
                y=shed_tile[1],
                amount=min(wheat_needed, wheat_in_shed),
            )
        )

    if plan.terminal_liquidation:
        tasks.extend(_terminal_deposit_tasks(ctx))

    return tasks


def _terminal_deposit_tasks(ctx):
    workers = [ctx.me["farmer"], *ctx.me["hands"]]
    tasks = []

    for worker_index, (position, inventory) in enumerate(
        zip(workers, ctx.private["inventories"])
    ):
        items = sellable_inventory_items(inventory)
        if not items:
            continue

        wx, wy = position
        sx, sy = min(
            shed_access_tiles(ctx),
            key=lambda pos: distance(wx, wy, pos[0], pos[1]),
        )

        for item, amount in items:
            tasks.append(
                Task(
                    type="deposit_product",
                    x=sx,
                    y=sy,
                    item=item,
                    amount=amount,
                    worker_index=worker_index,
                )
            )

    return tasks


def assign_tasks(ctx, tasks, rank_task):
    """Assign tasks using a policy-provided worker/task ranking.

    `rank_task(...)` returns a sortable tuple, or None to reject a task.
    Resource/legality constraints stay here so policy implementations do not
    need to relearn them.
    """
    workers = [ctx.me["farmer"], *ctx.me["hands"]]
    available_seeds = dict(ctx.private["seeds"])
    candidates = []

    for worker_index, worker_position in enumerate(workers):
        for task in tasks:
            if not worker_can_do_task(ctx, worker_index, task):
                continue

            rank = rank_task(
                ctx,
                worker_index,
                worker_position,
                task,
            )
            if rank is None:
                continue

            candidates.append((rank, worker_index, task))

    candidates.sort(key=lambda candidate: candidate[0])

    assignments = [None] * len(workers)
    assigned_tiles = set()

    for _, worker_index, task in candidates:
        if assignments[worker_index] is not None:
            continue

        if (
            task.type != "deposit_product"
            and (task.x, task.y) in assigned_tiles
        ):
            continue

        if task.type == "plant":
            if available_seeds.get(task.crop, 0) <= 0:
                continue
            available_seeds[task.crop] -= 1

        assignments[worker_index] = task
        if task.type != "deposit_product":
            assigned_tiles.add((task.x, task.y))

    return assignments


def execute_assignments(ctx, assignments):
    workers = [ctx.me["farmer"], *ctx.me["hands"]]
    actions = [
        execute_task(position, task)
        for position, task in zip(workers, assignments)
    ]
    return actions[0], actions[1:]


def execute_task(worker_position, task):
    if task is None:
        return ["PASS"]

    fx, fy = worker_position
    if (fx, fy) == (task.x, task.y):
        return task_action(task)

    return [step_toward(fx, fy, task.x, task.y)]


def task_action(task):
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


def step_toward(fx, fy, tx, ty):
    if fx > tx:
        return "WEST"
    if fx < tx:
        return "EAST"
    if fy > ty:
        return "NORTH"
    if fy < ty:
        return "SOUTH"
    return "PASS"
