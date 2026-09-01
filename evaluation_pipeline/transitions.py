from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import (ANIMAL_PRODUCTION_SCHEDULE, EpisodeIdentity, event, farm,
                     is_animal_tile, is_empty_structure)


@dataclass
class TransitionResult:
    counters: dict[str, float] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    escapes: int = 0

    def add(self, name: str, amount: float = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount


def is_decay_tick(prev_tile: dict[str, Any], prev_step: int) -> bool:
    max_lifespan_step = prev_tile.get("max_lifespan_step", -1)
    return (max_lifespan_step >= 0
            and prev_step >= max_lifespan_step
            and (prev_step - max_lifespan_step) % 2 == 0)


def is_natural_decay(prev_tile: dict[str, Any], prev_step: int) -> bool:
    return is_decay_tick(prev_tile, prev_step) and prev_tile.get("yield_units", 0) <= 1


def process_structural_transitions(prev_obs: dict[str, Any],
                                   curr_obs: dict[str, Any],
                                   identity: EpisodeIdentity,
                                   effective_by_pos: dict[tuple[int, int], list[dict[str, Any]]],
                                   *,
                                   prev_step: int,
                                   result_step: int,
                                   day: int,
                                   hour: int,
                                   day_rollover: bool) -> TransitionResult:
    result = TransitionResult()
    prev_tiles = farm(prev_obs, identity.player)["tiles"]
    curr_tiles = farm(curr_obs, identity.player)["tiles"]

    for y in range(len(prev_tiles)):
        for x in range(len(prev_tiles[y])):
            prev_tile = prev_tiles[y][x]
            curr_tile = curr_tiles[y][x]
            pos = (x, y)
            effective_here = effective_by_pos.get(pos, [])
            was_harvested_or_dug = any(action["action"] in {"HARVEST", "DIG"}
                                       for action in effective_here)

            if isinstance(prev_tile, dict) and prev_tile.get("kind") == "PLANT":
                if not was_harvested_or_dug and is_decay_tick(prev_tile, prev_step):
                    prev_yield = int(prev_tile.get("yield_units", 0))

                    if isinstance(curr_tile, dict) and curr_tile.get("kind") == "PLANT":
                        curr_yield = int(curr_tile.get("yield_units", 0))
                    elif isinstance(curr_tile, dict) and curr_tile.get("kind") == "WEED":
                        curr_yield = 0
                    else:
                        curr_yield = prev_yield

                    units_lost = max(0, prev_yield - curr_yield)
                    if units_lost:
                        result.add("crop_units_lost_to_decay", units_lost)
                        result.events.append(event(identity,
                                                   step=result_step,
                                                   day=day,
                                                   hour=hour,
                                                   event="CROP_UNITS_LOST_TO_DECAY",
                                                   item=prev_tile.get("crop"),
                                                   x=x,
                                                   y=y,
                                                   amount=units_lost))

                became_weed = isinstance(curr_tile, dict) and curr_tile.get("kind") == "WEED"
                if became_weed and not was_harvested_or_dug:
                    if is_natural_decay(prev_tile, prev_step):
                        result.add("natural_decays")
                        result.events.append(event(identity, step=result_step, day=day, hour=hour,
                                                   event="PLANT_NATURAL_DECAY", item=prev_tile.get("crop"),
                                                   x=x, y=y, amount=1))
                    else:
                        result.add("watering_deaths")
                        result.events.append(event(identity, step=result_step, day=day, hour=hour,
                                                   event="PLANT_DIED_UNWATERED", item=prev_tile.get("crop"),
                                                   x=x, y=y, amount=1))

            # A last-hour successful planting can die during the same transition,
            # so there is no prior plant tile to compare.
            if (day_rollover and prev_tile is None and isinstance(curr_tile, dict)
                    and curr_tile.get("kind") == "WEED"):
                planted_here = next((action for action in effective_here if action["action"] == "PLANT"), None)
                if planted_here is not None:
                    result.add("watering_deaths")
                    result.add("plant_water_opportunities")
                    result.add("missed_water_days")
                    result.events.append(event(identity, step=result_step, day=day, hour=hour,
                                               event="PLANT_DIED_SAME_DAY_UNWATERED",
                                               item=planted_here.get("item"), x=x, y=y, amount=1))

            if is_empty_structure(prev_tile) and is_animal_tile(curr_tile):
                result.add("animals_placed")
                result.events.append(event(identity, step=result_step, day=day, hour=hour,
                                           event="ANIMAL_PLACED", item=curr_tile.get("animal"),
                                           x=x, y=y, amount=1))

            if is_animal_tile(prev_tile) and is_empty_structure(curr_tile):
                result.add("animal_escapes")
                result.escapes += 1
                result.events.append(event(identity, step=result_step, day=day, hour=hour,
                                           event="ANIMAL_ESCAPED", item=prev_tile.get("animal"),
                                           x=x, y=y, amount=1))

    return result


def process_day_rollover(prev_obs: dict[str, Any],
                         curr_obs: dict[str, Any],
                         identity: EpisodeIdentity,
                         effective_by_pos: dict[tuple[int, int], list[dict[str, Any]]],
                         *,
                         prev_step: int,
                         result_step: int,
                         day: int,
                         hour: int,
                         result_day: int) -> TransitionResult:
    result = TransitionResult()
    prev_tiles = farm(prev_obs, identity.player)["tiles"]
    curr_tiles = farm(curr_obs, identity.player)["tiles"]

    effective_water = {pos for pos, actions in effective_by_pos.items()
                       if any(action["action"] == "WATER" for action in actions)}
    effective_feed = {pos for pos, actions in effective_by_pos.items()
                      if any(action["action"] == "FEED" for action in actions)}
    effective_collect = {pos for pos, actions in effective_by_pos.items()
                         if any(action["action"] == "COLLECT_FERTILIZER" for action in actions)}

    for y, row in enumerate(prev_tiles):
        for x, prev_tile in enumerate(row):
            pos = (x, y)
            curr_tile = curr_tiles[y][x]

            if isinstance(prev_tile, dict) and prev_tile.get("kind") == "PLANT":
                # Once decay has begun, watering is no longer a useful care
                # opportunity and should not count against the agent.
                max_life = prev_tile.get("max_lifespan_step", -1)
                still_requires_water = not (max_life >= 0 and prev_step >= max_life)

                if still_requires_water:
                    result.add("plant_water_opportunities")
                    watered = bool(prev_tile.get("watered_today", False) or pos in effective_water)
                    critical = bool(not prev_tile.get("watered_today", False)
                                    and prev_tile.get("consecutive_unwatered", 0) >= 1)

                    if critical:
                        result.add("critical_water_events")
                        if pos in effective_water:
                            result.add("critical_water_rescues")
                            result.events.append(event(identity, step=result_step, day=day, hour=hour,
                                                       event="CRITICAL_WATER_RESCUED",
                                                       item=prev_tile.get("crop"), x=x, y=y, amount=1))

                    if not watered:
                        result.add("missed_water_days")
                        result.events.append(event(identity, step=result_step, day=day, hour=hour,
                                                   event="MISSED_WATER_DAY", item=prev_tile.get("crop"),
                                                   x=x, y=y, amount=1))

                    if (critical and isinstance(curr_tile, dict) and curr_tile.get("kind") == "WEED"
                            and not is_natural_decay(prev_tile, prev_step)):
                        result.add("critical_water_failures")
                        result.events.append(event(identity, step=result_step, day=day, hour=hour,
                                                   event="CRITICAL_WATER_FAILED",
                                                   item=prev_tile.get("crop"), x=x, y=y, amount=1))

            if is_animal_tile(prev_tile):
                animal = prev_tile.get("animal")
                same_animal_survives = (is_animal_tile(curr_tile)
                                        and curr_tile.get("animal") == animal)

                result.add("animal_feed_opportunities")
                fed = bool(prev_tile.get("fed_today", False) or pos in effective_feed)
                critical = (not prev_tile.get("fed_today", False)
                            and prev_tile.get("consecutive_unfed", 0) >= 1)

                if critical:
                    result.add("critical_feed_events")
                    if pos in effective_feed:
                        result.add("critical_feed_rescues")
                        result.events.append(event(identity,
                                                   step=result_step,
                                                   day=day,
                                                   hour=hour,
                                                   event="CRITICAL_FEED_RESCUED",
                                                   item=animal,
                                                   x=x,
                                                   y=y,
                                                   amount=1))

                if not fed:
                    result.add("unfed_animal_days")
                    result.events.append(event(identity,
                                               step=result_step,
                                               day=day,
                                               hour=hour,
                                               event="UNFED_ANIMAL_DAY",
                                               item=animal,
                                               x=x,
                                               y=y,
                                               amount=1))

                    schedule = ANIMAL_PRODUCTION_SCHEDULE.get(animal)
                    placed_day = int(prev_tile.get("placed_day", -1))

                    if schedule and placed_day >= 0:
                        first_production_day = placed_day + schedule["first_yield_day"]
                        is_production_day = (result_day >= first_production_day
                                             and (result_day - first_production_day) % schedule["interval"] == 0)

                        if is_production_day:
                            pending_bonus = int(prev_tile.get("pending_care_bonus", 0))
                            held_units = int(prev_tile.get("yield_units", 0))

                            # Base production still happens when unfed. Only count
                            # bonus units that actually had room to be stored.
                            capacity_after_base = max(0, schedule["max_held"] - held_units - 1)
                            forfeited_bonus = min(pending_bonus, capacity_after_base)

                            result.add("unfed_production_days")
                            result.add("care_bonus_units_forfeited", forfeited_bonus)
                            result.events.append(event(identity,
                                                       step=result_step,
                                                       day=day,
                                                       hour=hour,
                                                       event="UNFED_ANIMAL_PRODUCTION_DAY",
                                                       item=animal,
                                                       x=x,
                                                       y=y,
                                                       amount=1,
                                                       details={"care_bonus_forfeited": forfeited_bonus}))

                # Fertilizer does not accumulate. If one was already available and
                # wasn't collected before rollover, the next day's opportunity was lost.
                if (prev_tile.get("fertilizer_available", False)
                        and pos not in effective_collect
                        and same_animal_survives):
                    result.add("fertilizer_collection_opportunities_missed")
                    result.events.append(event(identity,
                                               step=result_step,
                                               day=day,
                                               hour=hour,
                                               event="FERTILIZER_COLLECTION_OPPORTUNITY_MISSED",
                                               item=animal,
                                               x=x,
                                               y=y,
                                               amount=1))

                if critical and is_empty_structure(curr_tile):
                    result.add("critical_feed_failures")
                    result.events.append(event(identity,
                                               step=result_step,
                                               day=day,
                                               hour=hour,
                                               event="CRITICAL_FEED_FAILED",
                                               item=animal,
                                               x=x,
                                               y=y,
                                               amount=1))

    return result
