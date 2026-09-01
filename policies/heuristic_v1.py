"""Behavior-preserving policy extracted from dynamic_all_agent_v1.

This module contains the baseline's opinions: value estimates, thresholds,
ranking, purchase/sale rules, and high-level choices. The framework owns
state extraction, hard task feasibility, assignment bookkeeping, and action
execution.
"""

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    MARKET_PARAMS,
    MAX_SHOP_INSTANCES,
    SHOPS,
    market_price,
)

from agent_framework.core import (
    animal_added_today,
    animal_count,
    animal_structure_target,
    animal_tiles,
    carried_units,
    crop_age,
    days_to_full_yield,
    distance,
    empty_animal_structure,
    estimated_planting_batch,
    hire_cost,
    land_occupied_percentage,
    pending_animal,
    pending_animal_count,
    projected_end_of_day_overflow,
)


NAME = "heuristic_v1"

TERMINAL_LIQUIDATION_HOUR = 13
FERTILIZER_COLLECTION_MIN_PRICE_RATIO = 0.5
ANIMAL_PROFIT_HURDLE = 1.15
WHEAT_BUFFER_DAYS = 3
SHOP_UNLOCK_INTERVAL = 3
SHED_CAPACITY = 100
MAX_MARKET_ORDERS = 10

TASK_PRIORITY = {
    "deposit_product": 0,
    "critical_feed": 0,
    "critical_water": 0,
    "pickup_wheat": 1,
    "harvest_animal": 1,
    "harvest": 1,
    "place_animal": 1,
    "feed": 2,
    "water": 2,
    "care": 3,
    "collect_fertilizer": 3,
    "pickup_animal": 3,
    "build_structure": 3,
    "plant": 4,
    "weed": 5,
}

UNFERTILIZED_YIELDS = {
    "WHEAT": 4,
    "CARROT": 3,
    "TOMATO": 4,
    "STRAWBERRY": 4,
    "MELON": 6,
}


def is_terminal_liquidation(ctx):
    return ctx.day == 29 and ctx.hour >= TERMINAL_LIQUIDATION_HOUR


# ---------------------------------------------------------------------------
# Crop valuation and selection
# ---------------------------------------------------------------------------

def _projected_one_time_yield(tile, crop, day):
    crop_data = CROPS[crop]
    projected_yield = tile["yield_units"]
    age = crop_age(tile, day)
    window_start = (crop_data["max_yield_day"] + 1) // 2

    for future_age in range(
        max(age, window_start),
        crop_data["max_yield_day"] + 1,
    ):
        future_day = tile["planted_day"] + future_age

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


def _production_schedule(crop):
    crop_data = CROPS[crop]

    if not crop_data["ongoing"]:
        return [(days_to_full_yield(crop), UNFERTILIZED_YIELDS[crop])]

    return [
        (
            crop_data["first_yield_day"] + i * crop_data["interval"],
            1,
        )
        for i in range(crop_data["max_yield"])
    ]


def _growing_crop_supply(farm, crop, day, hour):
    crop_data = CROPS[crop]
    total_supply = 0

    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if (
                not isinstance(tile, dict)
                or tile.get("kind") != "PLANT"
                or tile.get("crop") != crop
            ):
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

            age = crop_age(tile, day)
            future_units = sum(
                units
                for production_day, units in _production_schedule(crop)
                if production_day > age
            )
            total_supply += tile["yield_units"] + future_units

    return total_supply


def _committed_crop_supply(ctx, crop):
    my_shed_units = ctx.private["shed"].get(crop, 0)
    my_carried_units = sum(
        inventory.get(crop, 0)
        for inventory in ctx.private["inventories"]
    )
    my_pipeline = _growing_crop_supply(ctx.me, crop, ctx.day, ctx.hour)
    opponent_pipeline = _growing_crop_supply(
        ctx.opponent,
        crop,
        ctx.day,
        ctx.hour,
    )

    return (
        my_shed_units
        + my_carried_units
        + my_pipeline
        + opponent_pipeline
    )


def _expected_town_consumption(ctx, crop, days):
    demand_per_day = 1

    for shop_name in ctx.town["unlocked_shops"]:
        products = SHOPS[shop_name]
        if crop in products:
            demand_per_day += 12 if len(products) == 1 else 6

    return demand_per_day * days


def _projected_crop_revenue(ctx, crop, candidate_plants):
    days_to_harvest = days_to_full_yield(crop)
    yield_per_plant = UNFERTILIZED_YIELDS[crop]
    committed_supply = _committed_crop_supply(ctx, crop)
    town_consumption = _expected_town_consumption(
        ctx,
        crop,
        days_to_harvest,
    )

    inventory_before_candidate = (
        ctx.market["inventory"][crop]
        + committed_supply
        - town_consumption
    )
    candidate_units = candidate_plants * yield_per_plant

    return sum(
        market_price(crop, inventory_before_candidate + i)
        for i in range(candidate_units)
    )


def crop_value(ctx, crop, candidate_plants):
    """Baseline crop value estimator: expected profit per tile-day."""
    days_to_harvest = days_to_full_yield(crop)
    revenue = _projected_crop_revenue(ctx, crop, candidate_plants)
    seed_cost = CROPS[crop]["seed"] * candidate_plants
    expected_profit = revenue - seed_cost
    tile_days = candidate_plants * days_to_harvest
    return expected_profit / tile_days


def _crops_available_to_plant(ctx):
    crops_available = [
        crop
        for crop in CROPS
        if ctx.day + days_to_full_yield(crop) < 29
        and ctx.me["money"] >= CROPS[crop]["seed"] * 6
    ]

    # Preserve the baseline's day-0 melon restriction as policy, not mechanics.
    if ctx.day == 0 and "MELON" in crops_available:
        crops_available.remove("MELON")

    return crops_available


def select_crop(ctx):
    if land_occupied_percentage(ctx) >= 100:
        return None

    candidate_plants = estimated_planting_batch(ctx)
    if candidate_plants == 0:
        return None

    best_crop = None
    best_value = float("-inf")

    for crop in _crops_available_to_plant(ctx):
        if crop not in UNFERTILIZED_YIELDS:
            continue

        value = crop_value(ctx, crop, candidate_plants)
        if value > best_value:
            best_value = value
            best_crop = crop

    return best_crop


# ---------------------------------------------------------------------------
# Animal valuation and selection
# ---------------------------------------------------------------------------

def _future_animal_production(tile, current_day):
    animal_data = ANIMALS[tile["animal"]]
    production_day = (
        tile["placed_day"] + animal_data["first_yield_day"]
    )

    while production_day <= current_day:
        production_day += animal_data["interval"]

    events = {}
    first_future_production = True

    while production_day <= 29:
        if first_future_production:
            days_until_production = production_day - current_day
            bonus = tile.get("pending_care_bonus", 0) + max(
                0,
                days_until_production - 1,
            )
            units = min(animal_data["max_held"], 1 + bonus)
            first_future_production = False
        else:
            units = min(
                animal_data["max_held"],
                1 + animal_data["interval"],
            )

        events[production_day] = units
        production_day += animal_data["interval"]

    return events


def _animal_product_supply(farm, product, current_day, current_hour):
    held_units = 0
    future_events = {}
    worker_positions = [farm["farmer"], *farm["hands"]]

    for x, y, tile in animal_tiles(farm):
        animal_data = ANIMALS[tile["animal"]]
        if animal_data["product"] != product:
            continue

        will_escape_tonight = (
            current_hour == 23
            and not tile["fed_today"]
            and tile["consecutive_unfed"] >= 1
            and [x, y] not in worker_positions
        )
        if will_escape_tonight:
            continue

        held_units += tile.get("yield_units", 0)
        for day, units in _future_animal_production(
            tile,
            current_day,
        ).items():
            future_events[day] = future_events.get(day, 0) + units

    return held_units, future_events


def _shop_product_demand_per_day(shop_name, product):
    products = SHOPS[shop_name]
    if product not in products:
        return 0
    return 12 if len(products) == 1 else 6


def _expected_future_shop_demand(product):
    total_demand = sum(
        _shop_product_demand_per_day(shop_name, product)
        for shop_name in SHOPS
    )
    return total_demand / len(SHOPS)


def _future_shop_instances(ctx, day):
    remaining_slots = (
        MAX_SHOP_INSTANCES - len(ctx.town["unlocked_shops"])
    )
    future_unlocks = sum(
        unlock_day % SHOP_UNLOCK_INTERVAL == 0
        for unlock_day in range(ctx.day + 1, day + 1)
    )
    return min(remaining_slots, future_unlocks)


def _expected_animal_town_demand(ctx, product, day):
    demand_per_day = 1

    for shop_name in ctx.town["unlocked_shops"]:
        demand_per_day += _shop_product_demand_per_day(
            shop_name,
            product,
        )

    future_instances = _future_shop_instances(ctx, day)
    demand_per_day += (
        future_instances * _expected_future_shop_demand(product)
    )
    return demand_per_day


def _projected_animal_revenue(ctx, animal):
    product = ANIMALS[animal]["product"]
    candidate_tile = {
        "animal": animal,
        "placed_day": ctx.day,
        "pending_care_bonus": 0,
    }
    candidate_events = _future_animal_production(
        candidate_tile,
        ctx.day,
    )

    if not candidate_events:
        return 0

    my_held, my_events = _animal_product_supply(
        ctx.me,
        product,
        ctx.day,
        ctx.hour,
    )
    opponent_held, opponent_events = _animal_product_supply(
        ctx.opponent,
        product,
        ctx.day,
        ctx.hour,
    )

    committed_inventory = ctx.private["shed"].get(product, 0)
    committed_inventory += sum(
        inventory.get(product, 0)
        for inventory in ctx.private["inventories"]
    )

    inventory = (
        ctx.market["inventory"][product]
        + committed_inventory
        + my_held
        + opponent_held
    )
    revenue = 0

    for day in range(ctx.day + 1, 30):
        inventory -= _expected_animal_town_demand(
            ctx,
            product,
            day,
        )
        inventory += (
            my_events.get(day, 0)
            + opponent_events.get(day, 0)
        )

        for _ in range(candidate_events.get(day, 0)):
            revenue += market_price(product, inventory)
            inventory += 1

    return revenue


def _projected_feed_cost(ctx):
    feed_days = 30 - ctx.day
    market_inventory = ctx.market["inventory"]["WHEAT"]
    return sum(
        market_price("WHEAT", market_inventory - i - 1)
        for i in range(feed_days)
    )


def animal_value(ctx, animal):
    """Baseline animal value estimator: expected profit per tile-day."""
    animal_data = ANIMALS[animal]

    if ctx.day + animal_data["first_yield_day"] > 29:
        return float("-inf")

    days_occupied = 30 - ctx.day
    revenue = _projected_animal_revenue(ctx, animal)
    expected_profit = (
        revenue
        - animal_data["cost"]
        - _projected_feed_cost(ctx)
    )
    return expected_profit / days_occupied


def _can_add_animal(ctx, animal):
    return (
        empty_animal_structure(ctx, animal) is not None
        or animal_structure_target(ctx) is not None
    )


def select_animal(ctx, crop_to_plant):
    if pending_animal(ctx) is not None or animal_added_today(ctx):
        return None

    candidate_plants = estimated_planting_batch(ctx)
    crop_score = (
        crop_value(ctx, crop_to_plant, candidate_plants)
        if crop_to_plant is not None and candidate_plants > 0
        else 0
    )

    best_animal = None
    best_value = float("-inf")

    for animal, animal_data in ANIMALS.items():
        if not _can_add_animal(ctx, animal):
            continue

        if ctx.day + animal_data["first_yield_day"] > 29:
            continue

        minimum_cash = (
            animal_data["cost"]
            + WHEAT_BUFFER_DAYS * ctx.market["prices"]["WHEAT"]
        )
        if ctx.me["money"] < minimum_cash:
            continue

        value = animal_value(ctx, animal)
        if value > best_value:
            best_value = value
            best_animal = animal

    hurdle = max(0, crop_score) * ANIMAL_PROFIT_HURDLE
    return best_animal if best_value > hurdle else None


# ---------------------------------------------------------------------------
# Task desirability and ranking
# ---------------------------------------------------------------------------

def _animal_produces_tonight(ctx, tile):
    animal_data = ANIMALS[tile["animal"]]
    days_since_first = (
        ctx.day
        + 1
        - tile["placed_day"]
        - animal_data["first_yield_day"]
    )
    return (
        days_since_first >= 0
        and days_since_first % animal_data["interval"] == 0
    )


def _should_harvest_animal(ctx, tile):
    held_units = tile.get("yield_units", 0)

    if ctx.day == 29 and ctx.hour >= 15:
        return held_units > 0

    animal_data = ANIMALS[tile["animal"]]
    held_units = tile.get("yield_units", 0)

    if held_units <= 0:
        return False

    early_harvest_threshold = max(
        1,
        animal_data["max_held"] // 2,
    )
    if ctx.day < 14 and held_units >= early_harvest_threshold:
        return True

    if held_units >= animal_data["max_held"]:
        return True

    if not _animal_produces_tonight(ctx, tile):
        return False

    incoming_units = 1 + tile.get("pending_care_bonus", 0)
    return (
        held_units + incoming_units
        > animal_data["max_held"]
    )


def _should_collect_fertilizer(ctx):
    min_price = (
        MARKET_PARAMS["FERTILIZER"]["base"]
        * FERTILIZER_COLLECTION_MIN_PRICE_RATIO
    )
    return ctx.market["prices"]["FERTILIZER"] > min_price


def _task_is_desirable(ctx, task):
    if is_terminal_liquidation(ctx):
        if task.type == "deposit_product":
            return True

        if task.type == "harvest_animal":
            # The original generator only emitted this task when the normal
            # harvest rule was already true, even during liquidation.
            return _should_harvest_animal(ctx, task.tile)

        if task.type == "harvest":
            crop_data = CROPS[task.crop]
            mature = (
                crop_age(task.tile, ctx.day)
                >= crop_data["first_yield_day"]
            )
            return mature and task.tile["yield_units"] > 0

        if task.type == "collect_fertilizer":
            return _should_collect_fertilizer(ctx)

        return False

    if task.type == "harvest":
        return (
            task.tile["yield_units"]
            >= UNFERTILIZED_YIELDS[task.crop]
        )

    if task.type == "harvest_animal":
        return _should_harvest_animal(ctx, task.tile)

    if task.type == "collect_fertilizer":
        return _should_collect_fertilizer(ctx)

    if task.type == "plant":
        return ctx.hour < 23

    return True


def rank_task(ctx, worker_index, worker_position, task):
    """Return baseline ordering for a worker/task pair, or None to reject it."""
    del worker_index  # Included for learned policies that need worker identity.

    if not _task_is_desirable(ctx, task):
        return None

    worker_x, worker_y = worker_position
    return (
        TASK_PRIORITY[task.type],
        distance(worker_x, worker_y, task.x, task.y),
    )


# ---------------------------------------------------------------------------
# Market policy
# ---------------------------------------------------------------------------

def _wheat_owned(ctx):
    shed_wheat = ctx.private["shed"].get("WHEAT", 0)
    carried_wheat = sum(
        inventory.get("WHEAT", 0)
        for inventory in ctx.private["inventories"]
    )
    return shed_wheat + carried_wheat


def _wheat_feed_target(ctx, animal_to_add=None):
    count = animal_count(ctx.me) + pending_animal_count(ctx)
    if animal_to_add is not None:
        count += 1
    return count * WHEAT_BUFFER_DAYS


def _projected_product_purchase_cost(ctx, product, quantity):
    market_inventory = ctx.market["inventory"][product]
    return sum(
        market_price(product, market_inventory - i - 1)
        for i in range(quantity)
    )


def _affordable_product_purchase(ctx, product, quantity, budget):
    market_inventory = ctx.market["inventory"][product]
    units = 0
    cost = 0

    for i in range(quantity):
        price = market_price(
            product,
            market_inventory - i - 1,
        )
        if cost + price > budget:
            break
        units += 1
        cost += price

    return units, cost


def _sell_quantity(ctx, crop, available):
    if is_terminal_liquidation(ctx):
        return available

    if crop == "FERTILIZER":
        return available

    if sum(ctx.private["shed"].values()) == SHED_CAPACITY:
        return available

    min_price = MARKET_PARAMS[crop]["base"] * 0.83
    market_inventory = ctx.market["inventory"][crop]
    quantity = 0

    for i in range(available):
        price = market_price(crop, market_inventory + i)
        if price <= min_price:
            break
        quantity += 1

    return quantity


def market_actions(ctx, plan, assignments):
    """Preserve the current baseline's market-order sequence exactly.

    This is intentionally still a heuristic policy. It is split from worker
    policy so a learned market policy can later replace it independently.
    """
    overflow_to_clear = projected_end_of_day_overflow(
        ctx,
        assignments,
    )
    actions = []
    money_available = ctx.me["money"]

    money_available = _append_hires(
        ctx,
        actions,
        money_available,
    )

    if ctx.day == 29:
        _append_day_29_sales(ctx, actions)
        return actions[:MAX_MARKET_ORDERS]

    animal_purchase, money_available = _append_animal_purchase(
        ctx,
        plan.animal_to_add,
        actions,
        money_available,
    )

    wheat_target = _wheat_feed_target(
        ctx,
        plan.animal_to_add if animal_purchase else None,
    )
    money_available = _append_wheat_purchase(
        ctx,
        wheat_target,
        actions,
        money_available,
    )
    money_available = _append_seed_purchase(
        ctx,
        plan.crop_to_plant,
        actions,
        money_available,
    )

    _append_sales(
        ctx,
        wheat_target,
        overflow_to_clear,
        actions,
    )

    _append_land_purchase(ctx, actions, money_available)
    return actions[:MAX_MARKET_ORDERS]


def _append_hires(ctx, actions, money_available):
    target_hires = 8
    hires_needed = (
        0
        if ctx.hour == 23
        else target_hires - ctx.me["hires_today"]
    )

    for i in range(hires_needed):
        hire_number = ctx.me["hires_today"] + i
        cost = hire_cost(hire_number)
        if money_available < cost:
            break
        actions.append(["HIRE"])
        money_available -= cost

    return money_available


def _append_day_29_sales(ctx, actions):
    terminal = is_terminal_liquidation(ctx)

    for item in MARKET_PARAMS:
        available = ctx.private["shed"].get(item, 0)
        if terminal:
            available += carried_units(ctx, item)
        if available <= 0:
            continue

        quantity = _sell_quantity(ctx, item, available)
        if quantity > 0:
            actions.append(["SELL", item, quantity])


def _append_animal_purchase(
    ctx,
    animal_to_add,
    actions,
    money_available,
):
    planned_wheat_target = _wheat_feed_target(ctx, animal_to_add)
    planned_wheat_needed = max(
        0,
        planned_wheat_target - _wheat_owned(ctx),
    )
    planned_wheat_cost = _projected_product_purchase_cost(
        ctx,
        "WHEAT",
        planned_wheat_needed,
    )

    if ctx.hour >= 23 or animal_to_add is None:
        return False, money_available

    animal_cost = ANIMALS[animal_to_add]["cost"]
    if money_available < animal_cost + planned_wheat_cost:
        return False, money_available

    actions.append(["BUY_ANIMAL", animal_to_add, 1])
    return True, money_available - animal_cost


def _append_wheat_purchase(
    ctx,
    wheat_target,
    actions,
    money_available,
):
    wheat_needed = max(0, wheat_target - _wheat_owned(ctx))
    if ctx.hour >= 23 or wheat_needed <= 0:
        return money_available

    quantity, wheat_cost = _affordable_product_purchase(
        ctx,
        "WHEAT",
        wheat_needed,
        money_available,
    )
    if quantity > 0:
        actions.append(["BUY_PRODUCT", "WHEAT", quantity])
        money_available -= wheat_cost

    return money_available


def _append_seed_purchase(
    ctx,
    crop_to_plant,
    actions,
    money_available,
):
    if crop_to_plant is None:
        return money_available

    seeds = ctx.private["seeds"].get(crop_to_plant, 0)
    seed_cost = CROPS[crop_to_plant]["seed"] * 5

    if seeds <= 2 and money_available >= seed_cost:
        actions.append(["BUY_SEED", crop_to_plant, 5])
        money_available -= seed_cost

    return money_available


def _append_sales(
    ctx,
    wheat_target,
    overflow_to_clear,
    actions,
):
    carried_wheat = sum(
        inventory.get("WHEAT", 0)
        for inventory in ctx.private["inventories"]
    )
    shed_items = list(ctx.private["shed"].items())

    if overflow_to_clear > 0:
        shed_items.sort(
            key=lambda item: ctx.market["prices"].get(item[0], 0),
            reverse=True,
        )

    for item_in_shed, inventory in shed_items:
        if inventory <= 0 or item_in_shed not in MARKET_PARAMS:
            continue

        available_to_sell = inventory
        if item_in_shed == "WHEAT":
            wheat_reserve = max(0, wheat_target - carried_wheat)
            available_to_sell = max(0, inventory - wheat_reserve)

        quantity = _sell_quantity(
            ctx,
            item_in_shed,
            available_to_sell,
        )

        if overflow_to_clear > quantity:
            extra_needed = overflow_to_clear - quantity
            quantity = min(inventory, quantity + extra_needed)

        overflow_to_clear = max(0, overflow_to_clear - quantity)

        if quantity > 0:
            actions.append(["SELL", item_in_shed, quantity])



def _append_land_purchase(ctx, actions, money_available):
    land_costs = [1000, 2000, 4000]
    extra_quadrants = len(ctx.me["unlocked_quadrants"]) - 1

    if extra_quadrants >= len(land_costs):
        return

    land_cost = land_costs[extra_quadrants]
    if (
        land_occupied_percentage(ctx) > 80
        and money_available >= land_cost * 1.6
        and ctx.day < 28
    ):
        actions.append(["BUY_LAND"])
