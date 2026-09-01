from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import gzip
import json
import os
import re
import uuid

SUPPORTED_BUILTIN_AGENT_STRINGS = {"starter"}


def _validate_agent_spec(agent: Any, agent_name: str, role: str) -> None:
    if callable(agent):
        return

    if isinstance(agent, str) and agent in SUPPORTED_BUILTIN_AGENT_STRINGS:
        return

    raise TypeError(f"{role} agent {agent_name!r} must be an imported Python callable. "
                    f"Only these built-in string agents are intentionally supported here: "
                    f"{sorted(SUPPORTED_BUILTIN_AGENT_STRINGS)}.")

def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "agent"


def _write_json_gz(path: Path, data: dict[str, Any]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(data, f, separators=(",", ":"))


def _run_match(args: tuple[Any, ...]) -> dict[str, Any]:
    from kaggle_environments import make

    (
        requested_seed,
        matchup_index,
        agents,
        player_agents,
        player_roles,
        output_dir,
        episode_steps,
    ) = args

    env = make(
        "kaggriculture",
        configuration={
            "episodeSteps": episode_steps,
            # Kaggriculture's reproducibility input is `seed`, not randomSeed.
            "seed": requested_seed,
        },
    )
    env.run(agents)
    episode = env.toJSON()
    resolved_seed = episode.get("info", {}).get("seed")
    if resolved_seed != requested_seed:
        raise RuntimeError(f"Seed mismatch: requested {requested_seed}, "
                           f"environment resolved {resolved_seed}.")

    test_player = player_roles.index("test")
    file_name = f"seed_{requested_seed:05d}_match_{matchup_index}_test_p{test_player}.json.gz"
    _write_json_gz(Path(output_dir) / file_name, episode)

    rewards = episode.get("rewards") or [step.get("reward") for step in episode["steps"][-1]]
    return {
        "match_id": f"seed_{requested_seed:05d}_match_{matchup_index}",
        "file": file_name,
        "requested_seed": requested_seed,
        "resolved_seed": resolved_seed,
        "test_player": test_player,
        "player_agents": list(player_agents),
        "player_roles": list(player_roles),
        "rewards": rewards,
    }


def run_evaluation(
    test_agent: Any,
    baseline_agent: Any,
    *,
    seeds: int,
    test_agent_name: str,
    baseline_agent_name: str,
    output_root: str | Path = "evaluations",
    run_name: Optional[str] = None,
    episode_steps: int = 720,
    max_workers: Optional[int] = None,
) -> Path:
    """Run paired Kaggriculture matches and save raw env.toJSON() episodes.

    For every requested seed, the test agent plays once as player 0 and once as
    player 1. The raw JSON is gzip-compressed but otherwise unmodified.
    """
    _validate_agent_spec(test_agent, test_agent_name, "test")
    _validate_agent_spec(baseline_agent, baseline_agent_name, "baseline")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if run_name is None:
        run_name = f"{_slug(test_agent_name)}_vs_{_slug(baseline_agent_name)}_{timestamp}"

    run_dir = Path(output_root) / run_name
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)

    worker_args = []
    for seed in range(seeds):
        worker_args.append((
            seed, 0,
            [test_agent, baseline_agent],
            [test_agent_name, baseline_agent_name],
            ["test", "baseline"],
            str(raw_dir), episode_steps,
        ))
        worker_args.append((
            seed, 1,
            [baseline_agent, test_agent],
            [baseline_agent_name, test_agent_name],
            ["baseline", "test"],
            str(raw_dir), episode_steps,
        ))

    if max_workers == 1:
        matches = [_run_match(args) for args in worker_args]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            matches = list(executor.map(_run_match, worker_args))

    metadata = {
        "run_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_agent": test_agent_name,
        "baseline_agent": baseline_agent_name,
        "requested_seed_count": seeds,
        "match_count": len(matches),
        "episode_steps": episode_steps,
        "seed_configuration_key": "seed",
        "raw_format": "env.toJSON() gzip-compressed",
        "matches": matches,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return run_dir
