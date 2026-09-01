from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import json

import pandas as pd

from .episode import EpisodeIdentity, analyze_player_episode, load_episode
from .summary import build_summary, write_summary_markdown


PARQUET_FILES = {"episodes": "episodes.parquet",
                 "events": "events.parquet",
                 "worker_actions": "worker_actions.parquet",
                 "market_orders": "market_orders.parquet",
                 "daily_states": "daily_states.parquet"}


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Writing Parquet requires pyarrow. Install it with `pip install pyarrow`."
        ) from exc


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def parse_run(run_dir: str | Path, *, write_outputs: bool = True) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    raw_dir = run_dir / "raw"
    metadata = _load_metadata(run_dir)
    run_id = metadata.get("run_id", run_dir.name)

    match_meta_by_file = {
        item["file"]: item
        for item in metadata.get("matches", [])
        if "file" in item
    }

    episode_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    worker_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    raw_files = sorted(raw_dir.glob("*.json")) + sorted(raw_dir.glob("*.json.gz"))
    if not raw_files:
        raise FileNotFoundError(f"No episode JSON files found under {raw_dir}")

    for path in raw_files:
        data = load_episode(path)
        meta = match_meta_by_file.get(path.name, {})
        requested_seed = meta.get("requested_seed")
        resolved_seed = data.get("info", {}).get("seed")
        match_id = meta.get("match_id", path.stem.replace(".json", ""))
        episode_id = data.get("id", match_id)

        player_agents = meta.get("player_agents", ["player_0", "player_1"])
        player_roles = meta.get("player_roles", ["unknown", "unknown"])

        for player in (0, 1):
            identity = EpisodeIdentity(
                run_id=run_id,
                match_id=match_id,
                episode_id=episode_id,
                source_file=path.name,
                requested_seed=requested_seed,
                resolved_seed=resolved_seed,
                player=player,
                agent=player_agents[player],
                opponent=player_agents[1 - player],
                role=player_roles[player],
                opponent_role=player_roles[1 - player],
            )
            analysis = analyze_player_episode(data, identity)
            episode_rows.append(analysis.episode)
            event_rows.extend(analysis.events)
            worker_rows.extend(analysis.worker_actions)
            market_rows.extend(analysis.market_orders)
            daily_rows.extend(analysis.daily_states)
    frames = {"episodes": pd.DataFrame(episode_rows),
          "events": pd.DataFrame(event_rows),
          "worker_actions": pd.DataFrame(worker_rows),
          "market_orders": pd.DataFrame(market_rows),
          "daily_states": pd.DataFrame(daily_rows)}

    test_agent = metadata.get("test_agent")
    baseline_agent = metadata.get("baseline_agent")
    frames["summary"] = build_summary(frames["episodes"], test_agent=test_agent, baseline_agent=baseline_agent)

    if write_outputs:
        processed_dir = run_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        for key, filename in PARQUET_FILES.items():
            _write_parquet(frames[key], processed_dir / filename)
        frames["summary"].to_csv(processed_dir / "summary.csv", index=False)
        write_summary_markdown(frames["summary"], processed_dir / "summary.md")

    return frames


def parse_episode_file(
    episode_path: str | Path,
    *,
    output_dir: Optional[str | Path] = None,
    agent_names: tuple[str, str] = ("player_0", "player_1"),
) -> dict[str, pd.DataFrame]:
    """Parse one raw episode outside an evaluation-run directory.

    Useful for validating the parser against an existing env.toJSON() file.
    """
    path = Path(episode_path)
    data = load_episode(path)
    episode_id = data.get("id", path.stem)
    resolved_seed = data.get("info", {}).get("seed")
    rows = {"episodes": [],
        "events": [],
        "worker_actions": [],
        "market_orders": [],
        "daily_states": []}

    for player in (0, 1):
        identity = EpisodeIdentity(
            run_id="standalone",
            match_id=episode_id,
            episode_id=episode_id,
            source_file=path.name,
            requested_seed=None,
            resolved_seed=resolved_seed,
            player=player,
            agent=agent_names[player],
            opponent=agent_names[1 - player],
            role="unknown",
            opponent_role="unknown",
        )
        analysis = analyze_player_episode(data, identity)
        rows["episodes"].append(analysis.episode)
        rows["events"].extend(analysis.events)
        rows["worker_actions"].extend(analysis.worker_actions)
        rows["market_orders"].extend(analysis.market_orders)
        rows["daily_states"].extend(analysis.daily_states)

    frames = {key: pd.DataFrame(value) for key, value in rows.items()}
    frames["summary"] = build_summary(frames["episodes"], test_agent=agent_names[0], baseline_agent=agent_names[1])

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for key, filename in PARQUET_FILES.items():
            _write_parquet(frames[key], out / filename)
        frames["summary"].to_csv(out / "summary.csv", index=False)
        write_summary_markdown(frames["summary"], out / "summary.md")

    return frames
