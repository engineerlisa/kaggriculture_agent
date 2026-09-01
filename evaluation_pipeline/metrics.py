"""Notebook-friendly compatibility wrapper for the richer evaluation pipeline.

This intentionally keeps the old `evaluate_and_log(...)` calling shape while
saving raw episodes and producing Parquet analysis tables instead of appending
one lossy aggregate CSV row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from evaluation_pipeline import parse_run, run_evaluation


def evaluate_and_log(
    test_agent: Any,
    baseline_agent: Any,
    seeds: int,
    test_agent_name: str,
    baseline_agent_name: str,
    *,
    output_root: str | Path = "evaluations",
    run_name: Optional[str] = None,
    max_workers: Optional[int] = None,
):
    run_dir = run_evaluation(
        test_agent=test_agent,
        baseline_agent=baseline_agent,
        seeds=seeds,
        test_agent_name=test_agent_name,
        baseline_agent_name=baseline_agent_name,
        output_root=output_root,
        run_name=run_name,
        max_workers=max_workers,
    )
    frames = parse_run(run_dir)

    episodes = frames["episodes"]
    test = episodes[episodes["role"] == "test"]
    if not test.empty:
        print(
            f"{test_agent_name} vs {baseline_agent_name} | "
            f"Win Rate: {test['win'].mean() * 100:.1f}% | "
            f"Mean Cash: {test['final_cash'].mean():.1f} | "
            f"Mean Margin: {test['margin'].mean():.1f} | "
            f"Watering Deaths: {test['watering_deaths'].mean():.2f} | "
            f"Animal Escapes: {test['animal_escapes'].mean():.2f} | "
            f"Productive Utilization: {test['productive_utilization'].mean() * 100:.1f}%"
        )
        print(f"Saved evaluation: {run_dir}")

    return run_dir, frames
