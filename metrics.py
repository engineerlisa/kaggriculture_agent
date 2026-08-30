import csv
import os
import numpy as np
from datetime import datetime
from kaggle_environments import make
from concurrent.futures import ProcessPoolExecutor

LOG_FILE = "evaluation_logs.csv"

HEADERS = [
    "Timestamp",
    "Test Agent",
    "Baseline Agent",
    "Seeds",
    "Games",
    # "Wins",
    # "Losses",
    # "Ties",
    "Win Rate",
    "Test Mean",
    # "Baseline Mean",
    "Mean Margin",
    "Std Margin",
    "Watering Deaths",
    "Natural Decays"]


def ensure_log_file():
    needs_header = (
        not os.path.exists(LOG_FILE)
        or os.path.getsize(LOG_FILE) == 0
    )

    with open(LOG_FILE, "a", newline="") as f:
        if needs_header:
            csv.writer(f).writerow(HEADERS)


def _count_plant_weed_causes(steps, player_index):
    watering_deaths = 0
    natural_decays = 0

    for i in range(1, len(steps)):
        prev_obs = steps[i - 1][0].observation
        curr_obs = steps[i][0].observation

        prev_tiles = prev_obs["farms"][player_index]["tiles"]
        curr_tiles = curr_obs["farms"][player_index]["tiles"]

        processed_step = prev_obs["step"]

        for y in range(len(prev_tiles)):
            for x in range(len(prev_tiles[y])):
                prev_tile = prev_tiles[y][x]
                curr_tile = curr_tiles[y][x]

                was_plant = (isinstance(prev_tile, dict) and prev_tile.get("kind") == "PLANT")

                is_weed = (isinstance(curr_tile, dict) and curr_tile.get("kind") == "WEED")

                if not (was_plant and is_weed):
                    continue

                max_lifespan_step = prev_tile["max_lifespan_step"]

                decays_this_step = (max_lifespan_step >= 0
                                    and processed_step >= max_lifespan_step
                                    and (processed_step - max_lifespan_step) % 2 == 0
                                    and prev_tile["yield_units"] <= 1)

                if decays_this_step:
                    natural_decays += 1
                else:
                    watering_deaths += 1

    return watering_deaths, natural_decays

def _run_single_seed(args):
    """Helper function to run the environment in parallel processes."""
    seed, test_agent, baseline_agent = args
    matchups = [
        ([test_agent, baseline_agent], 0),
        ([baseline_agent, test_agent], 1),
    ]

    results = []
    for agents, test_index in matchups:
        env = make(
            "kaggriculture",
            configuration={
                "episodeSteps": 720,
                "randomSeed": seed,
            },
        )

        steps = env.run(agents)
        watering_deaths, natural_decays = _count_plant_weed_causes(steps, test_index)
        final_step = steps[-1]

        baseline_index = 1 - test_index

        test_score = final_step[test_index].reward
        baseline_score = final_step[baseline_index].reward
        
        results.append((test_score, baseline_score, watering_deaths, natural_decays,))
        
    return results


def evaluate_and_log(test_agent,
                     baseline_agent,
                     seeds,
                     test_agent_name,
                     baseline_agent_name,
                    ):
    ensure_log_file()

    wins = 0
    losses = 0
    ties = 0

    test_scores = []
    baseline_scores = []
    margins = []

    watering_deaths = []
    natural_decays = []

    # Prepare arguments for multiprocessing
    worker_args = [(seed, test_agent, baseline_agent) for seed in range(seeds)]

    # Distribute the simulation workloads across all available CPU cores
    with ProcessPoolExecutor() as executor:
        all_results = executor.map(_run_single_seed, worker_args)

    # Process the gathered results
    for seed_results in all_results:
        for test_score, baseline_score, watering_count, decay_count in seed_results:
            test_scores.append(test_score)
            baseline_scores.append(baseline_score)

            watering_deaths.append(watering_count)
            natural_decays.append(decay_count)

            margin = test_score - baseline_score
            margins.append(margin)

            if margin > 0:
                wins += 1
            elif margin < 0:
                losses += 1
            else:
                ties += 1

    games = wins + losses + ties
    win_rate = wins / games * 100

    mean_test = np.mean(test_scores)
    mean_baseline = np.mean(baseline_scores)
    mean_margin = np.mean(margins)
    std_margin = np.std(margins)
    mean_watering_deaths = np.mean(watering_deaths)
    mean_natural_decays = np.mean(natural_decays)

    timestamp = datetime.now().strftime("%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            timestamp,
            test_agent_name,
            baseline_agent_name,
            seeds,
            games,
            # wins,
            # losses,
            # ties,
            int(win_rate),
            int(mean_test),
            # mean_baseline,
            int(mean_margin),
            int(std_margin),
            round(mean_watering_deaths, 1),
            round(mean_natural_decays, 1),
        ],)

    print(
        f"{test_agent_name} vs {baseline_agent_name} | "
        f"{wins}-{losses}-{ties} | "
        f"Win Rate: {win_rate:.1f}% | "
        f"Mean Margin: {mean_margin:.1f} | "
        f"Means: {mean_test:.1f} to {mean_baseline:.1f}",
        f"Watering Deaths: {mean_watering_deaths:.1f}",
        f"Natural Decays: {mean_natural_decays:.1f}"
    )