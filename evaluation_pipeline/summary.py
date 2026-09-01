from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import math

import pandas as pd

DEFAULT_SUMMARY_METRICS = ["final_cash",
                           "margin",
                           "win",
                           "watering_deaths",
                           "natural_decays",
                           "crop_units_lost_to_decay",
                           "animal_escapes",
                           "missed_water_days",
                           "water_adherence_rate",
                           "unfed_animal_days",
                           "unfed_production_days",
                           "care_bonus_units_forfeited",
                           "feed_adherence_rate",
                           "fertilizer_collection_opportunities_missed",
                           "estimated_shed_overflow_units",
                           "stranded_animals_end",
                           "stranded_animal_cost_end",
                           "unused_seeds_end",
                           "unused_seed_cost_end",
                           "terminal_sellable_value_at_current_prices",
                           "productive_utilization",
                           "movement_rate",
                           "pass_rate",
                           "known_noop_rate",
                           "travel_per_productive_action",
                           "crop_units_harvested",
                           "animal_product_units_harvested",
                           "fertilizer_collected",
                           "mean_land_occupancy",
                           "market_orders_over_limit",
                           "market_turns_over_limit",
                           "max_shed_inventory",
                           "unsold_sellable_units_end"]


METRIC_DIRECTIONS = {"final_cash": "higher",
                     "win": "higher",
                     "critical_water_rescue_rate": "higher",
                     "critical_feed_rescue_rate": "higher",

                     "watering_deaths": "lower",
                     "crop_units_lost_to_decay": "lower",
                     "animal_escapes": "lower",
                     "missed_water_days": "lower",
                     "unfed_animal_days": "lower",
                     "unfed_production_days": "lower",
                     "care_bonus_units_forfeited": "lower",
                     "fertilizer_collection_opportunities_missed": "lower",
                     "estimated_shed_overflow_units": "lower",
                     "stranded_animals_end": "lower",
                     "stranded_animal_cost_end": "lower",
                     "unused_seeds_end": "lower",
                     "unused_seed_cost_end": "lower",
                     "terminal_sellable_value_at_current_prices": "lower",
                     "known_noop_rate": "lower",
                     "market_orders_over_limit": "lower",
                     "market_turns_over_limit": "lower",
                     "unsold_sellable_units_end": "lower",

                     "natural_decays": "diagnostic",
                     "water_adherence_rate": "diagnostic",
                     "feed_adherence_rate": "diagnostic",
                     "productive_utilization": "diagnostic",
                     "movement_rate": "diagnostic",
                     "pass_rate": "diagnostic",
                     "travel_per_productive_action": "diagnostic",
                     "crop_units_harvested": "diagnostic",
                     "animal_product_units_harvested": "diagnostic",
                     "fertilizer_collected": "diagnostic",
                     "mean_land_occupancy": "diagnostic",
                     "max_shed_inventory": "diagnostic"}


def build_summary(episodes: pd.DataFrame,
                  *,
                  test_agent: Optional[str] = None,
                  baseline_agent: Optional[str] = None,
                  metrics: Iterable[str] = DEFAULT_SUMMARY_METRICS) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()

    if test_agent is None:
        test_rows = episodes[episodes["role"] == "test"]
        test_agent = test_rows["agent"].iloc[0] if not test_rows.empty else episodes["agent"].iloc[0]

    if baseline_agent is None:
        baseline_rows = episodes[episodes["role"] == "baseline"]
        candidates = baseline_rows["agent"].unique().tolist()
        baseline_agent = candidates[0] if candidates else None

    rows = []

    for metric in metrics:
        if metric not in episodes.columns:
            continue

        direction = METRIC_DIRECTIONS.get(metric, "diagnostic")

        test_values = pd.to_numeric(episodes.loc[episodes["agent"] == test_agent, metric],
                                    errors="coerce").dropna()

        # Margin already equals test_cash - opponent_cash.
        # Comparing it with the opponent's mirrored margin would double the same quantity.
        compare_to_baseline = baseline_agent is not None and metric != "margin"

        baseline_values = (
            pd.to_numeric(episodes.loc[episodes["agent"] == baseline_agent, metric],
                          errors="coerce").dropna()
            if compare_to_baseline
            else pd.Series(dtype=float)
        )

        test_mean = test_values.mean() if not test_values.empty else math.nan
        baseline_mean = baseline_values.mean() if not baseline_values.empty else math.nan

        delta = (
            test_mean - baseline_mean
            if compare_to_baseline and pd.notna(baseline_mean)
            else math.nan
        )

        delta_pct = (
            delta / abs(baseline_mean)
            if compare_to_baseline and pd.notna(baseline_mean) and baseline_mean != 0
            else math.nan
        )

        paired = (
            episodes[episodes["agent"].isin([test_agent, baseline_agent])][
                ["match_id", "agent", metric]
            ].copy()
            if compare_to_baseline
            else pd.DataFrame()
        )

        paired_delta_mean = math.nan
        paired_delta_std = math.nan
        paired_ci_low = math.nan
        paired_ci_high = math.nan
        improved_rate = math.nan

        if not paired.empty:
            wide = paired.pivot_table(index="match_id",
                                      columns="agent",
                                      values=metric,
                                      aggfunc="first")

            if test_agent in wide.columns and baseline_agent in wide.columns:
                deltas = (
                    pd.to_numeric(wide[test_agent], errors="coerce")
                    - pd.to_numeric(wide[baseline_agent], errors="coerce")
                ).dropna()

                if not deltas.empty:
                    paired_delta_mean = deltas.mean()
                    paired_delta_std = deltas.std(ddof=1) if len(deltas) > 1 else 0.0
                    se = paired_delta_std / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0

                    paired_ci_low = paired_delta_mean - 1.96 * se
                    paired_ci_high = paired_delta_mean + 1.96 * se

                    if direction == "higher":
                        improved_rate = (deltas > 0).mean()
                    elif direction == "lower":
                        improved_rate = (deltas < 0).mean()

        rows.append({"metric": metric,
                     "direction": direction,
                     "test_agent": test_agent,
                     "baseline_agent": baseline_agent,
                     "test_mean": test_mean,
                     "baseline_mean": baseline_mean,
                     "delta": delta,
                     "delta_pct": delta_pct,
                     "test_median": test_values.median() if not test_values.empty else math.nan,
                     "baseline_median": baseline_values.median() if not baseline_values.empty else math.nan,
                     "test_std": test_values.std(ddof=1) if len(test_values) > 1 else 0.0,
                     "baseline_std": baseline_values.std(ddof=1) if len(baseline_values) > 1 else 0.0,
                     "paired_delta_mean": paired_delta_mean,
                     "paired_delta_std": paired_delta_std,
                     "paired_95ci_low": paired_ci_low,
                     "paired_95ci_high": paired_ci_high,
                     "paired_improved_rate": improved_rate,
                     "n_test": len(test_values),
                     "n_baseline": len(baseline_values)})

    return pd.DataFrame(rows)


def write_summary_markdown(summary: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    if summary.empty:
        path.write_text("No summary data.\n", encoding="utf-8")
        return

    preferred = [
        "final_cash", "margin", "win", "watering_deaths", "animal_escapes",
        "estimated_shed_overflow_units", "productive_utilization", "movement_rate",
        "pass_rate", "travel_per_productive_action", "market_orders_over_limit",
    ]
    view = summary[summary["metric"].isin(preferred)].copy()
    if view.empty:
        view = summary.head(12).copy()

    lines = ["# Evaluation summary", "", "| Metric | Test mean | Baseline mean | Delta | Paired 95% CI |", "|---|---:|---:|---:|---:|"]
    for _, row in view.iterrows():
        ci = f"[{row['paired_95ci_low']:.3g}, {row['paired_95ci_high']:.3g}]" if pd.notna(row["paired_95ci_low"]) else ""
        lines.append(
            f"| {row['metric']} | {row['test_mean']:.4g} | {row['baseline_mean']:.4g} | {row['delta']:.4g} | {ci} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
