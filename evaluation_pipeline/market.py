from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import EpisodeIdentity, action_parts, base_row, event, private


@dataclass
class MarketResult:
    rows: list[dict[str, Any]]
    events: list[dict[str, Any]]
    submitted: int
    over_limit: int
    within_limit: int
    turns_over_limit: int


def process_market_orders(prev_obs: dict[str, Any],
                          identity: EpisodeIdentity,
                          action_dict: dict[str, Any],
                          *,
                          result_step: int,
                          day: int,
                          hour: int,
                          result_day: int,
                          result_hour: int,
                          market_order_limit: int) -> MarketResult:
    market_actions = action_dict.get("market", [])
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for order_index, order in enumerate(market_actions):
        op, item, amount = action_parts(order)
        row = base_row(identity)
        row.update({"step": result_step,
                    "decision_day": day,
                    "decision_hour": hour,
                    "result_day": result_day,
                    "result_hour": result_hour,
                    "order_index": order_index,
                    "action": op,
                    "item": item,
                    "quantity_requested": amount if len(order) > 2 else (1 if op == "HIRE" else None),
                    "market_price": prev_obs.get("market", {}).get("prices", {}).get(item) if item else None,
                    "market_inventory": prev_obs.get("market", {}).get("inventory", {}).get(item) if item else None,
                    "within_order_limit": order_index < market_order_limit,
                    "drop_reason": None if order_index < market_order_limit else "OVER_MARKET_ORDER_LIMIT"})
        rows.append(row)

        if order_index >= market_order_limit:
            events.append(event(identity, step=result_step, day=day, hour=hour,
                                event="MARKET_ORDER_DROPPED_LIMIT", item=item, amount=amount,
                                details={"action": op, "order_index": order_index, "limit": market_order_limit}))

    return MarketResult(rows=rows,
                        events=events,
                        submitted=len(market_actions),
                        over_limit=max(0, len(market_actions) - market_order_limit),
                        within_limit=min(len(market_actions), market_order_limit),
                        turns_over_limit=int(len(market_actions) > market_order_limit))


def estimate_shed_overflow(prev_obs: dict[str, Any],
                           effective_actions: list[dict[str, Any]],
                           market_actions: list[Any],
                           identity: EpisodeIdentity,
                           *,
                           result_step: int,
                           day: int,
                           hour: int,
                           market_order_limit: int,
                           shed_capacity: int) -> tuple[int, dict[str, Any] | None]:
    # Estimate overflow using the pre-refresh shed+carried inventory, known
    # same-turn worker gains, and SELL orders that were within the queue limit.
    # The field remains deliberately labeled estimated because partial market
    # fills can make exact attribution impossible from the serialized state alone.
    prev_private = private(prev_obs)
    shed_units = sum(prev_private.get("shed", {}).values())
    carried_units = sum(sum(inv.values()) for inv in prev_private.get("inventories", []))
    incoming = 0
    outgoing_worker = 0

    for worker_action in effective_actions:
        if worker_action["success"] is not True:
            continue

        if worker_action["action"] == "HARVEST":
            incoming += worker_action["effect_amount"]
        elif worker_action["action"] == "COLLECT_FERTILIZER":
            incoming += worker_action["effect_amount"]
        elif worker_action["action"] == "FEED":
            outgoing_worker += 1
        elif worker_action["action"] == "PLACE" and worker_action["target_kind"] == "animal":
            outgoing_worker += worker_action["effect_amount"]

    sell_reduction = 0
    running_shed = dict(prev_private.get("shed", {}))

    for order_index, order in enumerate(market_actions[:market_order_limit]):
        op, item, amount = action_parts(order)
        if op == "SELL" and item:
            sold = min(max(amount, 1), running_shed.get(item, 0))
            sell_reduction += sold
            running_shed[item] = running_shed.get(item, 0) - sold

    projected = shed_units + carried_units + incoming - outgoing_worker - sell_reduction
    estimated_overflow = max(0, projected - shed_capacity)

    if estimated_overflow <= 0:
        return 0, None

    overflow_event = event(identity, step=result_step, day=day, hour=hour,
                           event="SHED_OVERFLOW_ESTIMATED", amount=estimated_overflow,
                           details={"projected_units": projected, "capacity": shed_capacity})
    return estimated_overflow, overflow_event
