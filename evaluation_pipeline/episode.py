from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import gzip
import json
import math

import pandas as pd

from .common import (ANIMALS, ANIMAL_PURCHASE_COSTS, CROP_SEED_COSTS, EpisodeIdentity,
                     base_row, event, farm, hire_cost, obs, private)
from .market import estimate_shed_overflow, process_market_orders
from .state_metrics import (collect_state_samples, owned_animals, private_nonseed_units,
                            private_sellable_units, private_sellable_value_at_current_prices)
from .transitions import process_day_rollover, process_structural_transitions
from .worker_actions import effective_worker_actions


@dataclass
class EpisodeAnalysis:
    episode: dict[str, Any]
    events: list[dict[str, Any]]
    worker_actions: list[dict[str, Any]]
    market_orders: list[dict[str, Any]]
    daily_states: list[dict[str, Any]]

    def frames(self) -> tuple[pd.DataFrame,
                              pd.DataFrame,
                              pd.DataFrame,
                              pd.DataFrame,
                              pd.DataFrame]:
        return (pd.DataFrame([self.episode]),
                pd.DataFrame(self.events),
                pd.DataFrame(self.worker_actions),
                pd.DataFrame(self.market_orders),
                pd.DataFrame(self.daily_states))


def load_episode(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _new_counters() -> dict[str, float]:
    return {"plants_planted": 0,
            "plant_harvest_actions": 0,
            "crop_units_harvested": 0,
            "watering_deaths": 0,
            "natural_decays": 0,
            "crop_units_lost_to_decay": 0,
            "missed_water_days": 0,
            "critical_water_events": 0,
            "critical_water_rescues": 0,
            "critical_water_failures": 0,
            "plant_water_opportunities": 0,
            "animals_bought_observed": 0,
            "animals_placed": 0,
            "animal_escapes": 0,
            "unfed_animal_days": 0,
            "unfed_production_days": 0,
            "care_bonus_units_forfeited": 0,
            "fertilizer_collection_opportunities_missed": 0,
            "critical_feed_events": 0,
            "critical_feed_rescues": 0,
            "critical_feed_failures": 0,
            "animal_feed_opportunities": 0,
            "animal_harvest_actions": 0,
            "animal_product_units_harvested": 0,
            "fertilizer_collected": 0,
            "care_actions_effective": 0,
            "feed_actions_effective": 0,
            "worker_turns": 0,
            "productive_actions_submitted": 0,
            "productive_actions_effective": 0,
            "movement_actions": 0,
            "logistics_actions": 0,
            "pass_actions": 0,
            "known_noop_actions": 0,
            "hands_hired_observed": 0,
            "hire_cost_observed": 0,
            "market_orders_submitted": 0,
            "market_orders_over_limit": 0,
            "market_orders_within_limit": 0,
            "market_turns_over_limit": 0,
            "shed_full_turns": 0,
            "estimated_shed_overflow_units": 0}


def _merge_counters(counters: dict[str, float], updates: dict[str, float]) -> None:
    for name, amount in updates.items():
        counters[name] = counters.get(name, 0) + amount


def _record_worker_actions(effective_actions: list[dict[str, Any]],
                           identity: EpisodeIdentity,
                           counters: dict[str, float],
                           events: list[dict[str, Any]],
                           worker_rows: list[dict[str, Any]],
                           *,
                           result_step: int,
                           day: int,
                           hour: int,
                           result_day: int,
                           result_hour: int) -> dict[tuple[int, int], list[dict[str, Any]]]:
    effective_by_pos: dict[tuple[int, int], list[dict[str, Any]]] = {}

    for worker_action in effective_actions:
        row = base_row(identity)
        row.update({"step": result_step,
                    "decision_day": day,
                    "decision_hour": hour,
                    "result_day": result_day,
                    "result_hour": result_hour,
                    **worker_action})
        worker_rows.append(row)

        counters["worker_turns"] += 1
        category = worker_action["category"]

        if category == "productive":
            counters["productive_actions_submitted"] += 1
            if worker_action["success"] is True:
                counters["productive_actions_effective"] += 1
        elif category == "movement":
            counters["movement_actions"] += 1
        elif category == "logistics":
            counters["logistics_actions"] += 1
        elif category == "idle":
            counters["pass_actions"] += 1

        if worker_action["success"] is False:
            counters["known_noop_actions"] += 1

        if (worker_action["x"] is not None and worker_action["y"] is not None
                and worker_action["success"] is True):
            pos = (int(worker_action["x"]), int(worker_action["y"]))
            effective_by_pos.setdefault(pos, []).append(worker_action)

        if worker_action["success"] is not True:
            continue

        op = worker_action["action"]
        if op == "PLANT":
            counters["plants_planted"] += 1
            events.append(event(identity, step=result_step, day=day, hour=hour,
                                event="PLANT_PLANTED", item=worker_action["item"],
                                x=worker_action["x"], y=worker_action["y"], amount=1))
        elif op == "HARVEST" and worker_action["target_kind"] == "plant":
            counters["plant_harvest_actions"] += 1
            counters["crop_units_harvested"] += worker_action["effect_amount"]
            events.append(event(identity, step=result_step, day=day, hour=hour,
                                event="CROP_HARVESTED", item=worker_action["effect_item"],
                                x=worker_action["x"], y=worker_action["y"],
                                amount=worker_action["effect_amount"]))
        elif op == "HARVEST" and worker_action["target_kind"] == "animal":
            counters["animal_harvest_actions"] += 1
            counters["animal_product_units_harvested"] += worker_action["effect_amount"]
            events.append(event(identity, step=result_step, day=day, hour=hour,
                                event="ANIMAL_PRODUCT_HARVESTED", item=worker_action["effect_item"],
                                x=worker_action["x"], y=worker_action["y"],
                                amount=worker_action["effect_amount"]))
        elif op == "COLLECT_FERTILIZER":
            counters["fertilizer_collected"] += worker_action["effect_amount"]
            events.append(event(identity, step=result_step, day=day, hour=hour,
                                event="FERTILIZER_COLLECTED", item="FERTILIZER",
                                x=worker_action["x"], y=worker_action["y"],
                                amount=worker_action["effect_amount"]))
        elif op == "CARE":
            counters["care_actions_effective"] += 1
        elif op == "FEED":
            counters["feed_actions_effective"] += 1

    return effective_by_pos


def _terminal_inventory_metrics(final_obs: dict[str, Any]) -> dict[str, float]:
    final_private = private(final_obs)

    stranded_animals_end = sum(final_private.get("shed", {}).get(animal, 0) for animal in ANIMALS)
    stranded_animals_end += sum(sum(inv.get(animal, 0) for animal in ANIMALS)
                                for inv in final_private.get("inventories", []))

    unused_seeds_end = sum(final_private.get("seeds", {}).values())
    unused_seed_cost_end = sum(quantity * CROP_SEED_COSTS.get(crop, 0)
                               for crop, quantity in final_private.get("seeds", {}).items())

    stranded_animal_cost_end = sum(final_private.get("shed", {}).get(animal, 0)
                                   * ANIMAL_PURCHASE_COSTS[animal]
                                   for animal in ANIMALS)
    stranded_animal_cost_end += sum(sum(inv.get(animal, 0) * ANIMAL_PURCHASE_COSTS[animal]
                                        for animal in ANIMALS)
                                    for inv in final_private.get("inventories", []))

    return {"unsold_sellable_units_end": private_sellable_units(final_obs),
            "stranded_animals_end": stranded_animals_end,
            "stranded_animal_cost_end": stranded_animal_cost_end,
            "unused_seeds_end": unused_seeds_end,
            "unused_seed_cost_end": unused_seed_cost_end,
            "terminal_sellable_value_at_current_prices": private_sellable_value_at_current_prices(final_obs),
            "nonseed_inventory_units_end": private_nonseed_units(final_obs)}


def analyze_player_episode(data: dict[str, Any], identity: EpisodeIdentity) -> EpisodeAnalysis:
    player = identity.player
    opponent_player = 1 - player
    steps = data["steps"]

    configuration = data.get("configuration", {})
    turns_per_day = int(configuration.get("turnsPerDay", 24))
    shed_capacity = int(configuration.get("shedCapacity", 100))
    market_order_limit = int(configuration.get("maxMarketOrdersPerTurn", 10))
    hire_multiplier = int(configuration.get("farmHandCostMult", 1))

    counters = _new_counters()
    events: list[dict[str, Any]] = []
    worker_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []

    samples = collect_state_samples(steps, player, identity, shed_capacity)
    counters["shed_full_turns"] = samples.shed_full_turns

    for i in range(1, len(steps)):
        prev_obs = obs(steps[i - 1], player)
        curr_obs = obs(steps[i], player)
        action_dict = steps[i][player].get("action") or {"farmer": ["PASS"], "hands": [], "market": []}

        prev_step = int(prev_obs.get("step", i - 1) if prev_obs.get("step") is not None else i - 1)
        result_step = int(curr_obs.get("step", i) if curr_obs.get("step") is not None else i)
        day = int(prev_obs.get("day", 0))
        hour = int(prev_obs.get("hour", 0))
        result_day = int(curr_obs.get("day", day))
        result_hour = int(curr_obs.get("hour", hour))
        day_rollover = result_day != day

        effective_actions = effective_worker_actions(prev_obs, curr_obs, player, action_dict)
        effective_by_pos = _record_worker_actions(effective_actions,
                                                  identity,
                                                  counters,
                                                  events,
                                                  worker_rows,
                                                  result_step=result_step,
                                                  day=day,
                                                  hour=hour,
                                                  result_day=result_day,
                                                  result_hour=result_hour)

        market_result = process_market_orders(prev_obs,
                                              identity,
                                              action_dict,
                                              result_step=result_step,
                                              day=day,
                                              hour=hour,
                                              result_day=result_day,
                                              result_hour=result_hour,
                                              market_order_limit=market_order_limit)
        market_rows.extend(market_result.rows)
        events.extend(market_result.events)
        counters["market_orders_submitted"] += market_result.submitted
        counters["market_orders_over_limit"] += market_result.over_limit
        counters["market_orders_within_limit"] += market_result.within_limit
        counters["market_turns_over_limit"] += market_result.turns_over_limit

        structural = process_structural_transitions(prev_obs,
                                                    curr_obs,
                                                    identity,
                                                    effective_by_pos,
                                                    prev_step=prev_step,
                                                    result_step=result_step,
                                                    day=day,
                                                    hour=hour,
                                                    day_rollover=day_rollover)
        _merge_counters(counters, structural.counters)
        events.extend(structural.events)

        # Successful animal purchases are observable from total animal ownership
        # after adding back any animals that escaped during the same transition.
        animal_delta = owned_animals(curr_obs, player) - owned_animals(prev_obs, player) + structural.escapes
        if animal_delta > 0:
            counters["animals_bought_observed"] += animal_delta

        if day_rollover:
            rollover = process_day_rollover(prev_obs,
                                            curr_obs,
                                            identity,
                                            effective_by_pos,
                                            prev_step=prev_step,
                                            result_step=result_step,
                                            day=day,
                                            hour=hour,
                                            result_day=result_day)
            _merge_counters(counters, rollover.counters)
            events.extend(rollover.events)

            estimated_overflow, overflow_event = estimate_shed_overflow(
                prev_obs,
                effective_actions,
                action_dict.get("market", []),
                identity,
                result_step=result_step,
                day=day,
                hour=hour,
                market_order_limit=market_order_limit,
                shed_capacity=shed_capacity)

            if estimated_overflow > 0 and overflow_event is not None:
                counters["estimated_shed_overflow_units"] += estimated_overflow
                events.append(overflow_event)

        # Exact successful hire count is observable from hires_today except at a
        # day rollover, where workers reset immediately after the last action.
        if not day_rollover:
            prev_hires = int(farm(prev_obs, player).get("hires_today", 0))
            curr_hires = int(farm(curr_obs, player).get("hires_today", 0))
            hires = max(0, curr_hires - prev_hires)

            if hires:
                counters["hands_hired_observed"] += hires
                counters["hire_cost_observed"] += hire_cost(prev_hires, hires, hire_multiplier)

    initial_obs = obs(steps[0], player)
    final_obs = obs(steps[-1], player)
    final_reward = steps[-1][player].get("reward")
    opponent_reward = steps[-1][opponent_player].get("reward")

    final_cash = float(final_reward if final_reward is not None else farm(final_obs, player).get("money", 0))
    opponent_cash = float(opponent_reward if opponent_reward is not None
                          else farm(final_obs, opponent_player).get("money", 0))
    margin = final_cash - opponent_cash

    episode = base_row(identity)
    episode.update({"status": steps[-1][player].get("status"),
                    "steps": len(steps),
                    "final_cash": final_cash,
                    "opponent_cash": opponent_cash,
                    "margin": margin,
                    "win": int(margin > 0),
                    "tie": int(margin == 0),
                    "starting_cash": float(farm(initial_obs, player).get("money", 0)),
                    **{name: int(value) if float(value).is_integer() else float(value)
                       for name, value in counters.items()},
                    "productive_utilization": counters["productive_actions_effective"] / counters["worker_turns"] if counters["worker_turns"] else 0.0,
                    "water_adherence_rate": 1.0 - counters["missed_water_days"] / counters["plant_water_opportunities"] if counters["plant_water_opportunities"] else math.nan,
                    "feed_adherence_rate": 1.0 - counters["unfed_animal_days"] / counters["animal_feed_opportunities"] if counters["animal_feed_opportunities"] else math.nan,
                    "critical_water_rescue_rate": counters["critical_water_rescues"] / counters["critical_water_events"] if counters["critical_water_events"] else math.nan,
                    "critical_feed_rescue_rate": counters["critical_feed_rescues"] / counters["critical_feed_events"] if counters["critical_feed_events"] else math.nan,
                    "movement_rate": counters["movement_actions"] / counters["worker_turns"] if counters["worker_turns"] else 0.0,
                    "pass_rate": counters["pass_actions"] / counters["worker_turns"] if counters["worker_turns"] else 0.0,
                    "known_noop_rate": counters["known_noop_actions"] / counters["worker_turns"] if counters["worker_turns"] else 0.0,
                    "travel_per_productive_action": counters["movement_actions"] / counters["productive_actions_effective"] if counters["productive_actions_effective"] else math.nan,
                    "mean_land_occupancy": sum(samples.land_occupancy_samples) / len(samples.land_occupancy_samples) if samples.land_occupancy_samples else 0.0,
                    "crop_tile_days": samples.crop_tile_turns / turns_per_day,
                    "animal_tile_days": samples.animal_tile_turns / turns_per_day,
                    "crop_units_per_tile_day": counters["crop_units_harvested"] / (samples.crop_tile_turns / turns_per_day) if samples.crop_tile_turns else math.nan,
                    "animal_product_units_per_tile_day": counters["animal_product_units_harvested"] / (samples.animal_tile_turns / turns_per_day) if samples.animal_tile_turns else math.nan,
                    "harvested_units_per_worker_turn": (counters["crop_units_harvested"] + counters["animal_product_units_harvested"] + counters["fertilizer_collected"]) / counters["worker_turns"] if counters["worker_turns"] else math.nan,
                    "productive_actions_per_hire": counters["productive_actions_effective"] / counters["hands_hired_observed"] if counters["hands_hired_observed"] else math.nan,
                    "weed_tile_days": samples.weed_tile_turns / turns_per_day,
                    "empty_structure_tile_days": samples.structure_tile_turns / turns_per_day,
                    "max_shed_inventory": max(samples.shed_samples) if samples.shed_samples else 0,
                    "mean_shed_inventory": sum(samples.shed_samples) / len(samples.shed_samples) if samples.shed_samples else 0.0,
                    **_terminal_inventory_metrics(final_obs)})

    return EpisodeAnalysis(episode=episode,
                           events=events,
                           worker_actions=worker_rows,
                           market_orders=market_rows,
                           daily_states=samples.daily_rows)
