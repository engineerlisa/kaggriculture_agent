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
    # "Std Margin",
]


def ensure_log_file():
    needs_header = (
        not os.path.exists(LOG_FILE)
        or os.path.getsize(LOG_FILE) == 0
    )

    with open(LOG_FILE, "a", newline="") as f:
        if needs_header:
            csv.writer(f).writerow(HEADERS)


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
        final_step = steps[-1]

        baseline_index = 1 - test_index

        test_score = final_step[test_index].reward
        baseline_score = final_step[baseline_index].reward
        
        results.append((test_score, baseline_score))
        
    return results


def evaluate_and_log(test_agent,
                     baseline_agent,
                     seeds,
                     test_agent_name="Test Agent",
                     baseline_agent_name="Baseline Agent",
                    ):
    ensure_log_file()

    wins = 0
    losses = 0
    ties = 0

    test_scores = []
    baseline_scores = []
    margins = []

    # Prepare arguments for multiprocessing
    worker_args = [(seed, test_agent, baseline_agent) for seed in range(seeds)]

    # Distribute the simulation workloads across all available CPU cores
    with ProcessPoolExecutor() as executor:
        all_results = executor.map(_run_single_seed, worker_args)

    # Process the gathered results
    for seed_results in all_results:
        for test_score, baseline_score in seed_results:
            test_scores.append(test_score)
            baseline_scores.append(baseline_score)

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
            # std_margin,
        ],)

    print(
        f"{test_agent_name} vs {baseline_agent_name} | "
        f"{wins}-{losses}-{ties} | "
        f"Win Rate: {win_rate:.1f}% | "
        f"Mean Margin: {mean_margin:.1f} | "
        f"Means: {mean_test:.1f} to {mean_baseline:.1f}"
    )