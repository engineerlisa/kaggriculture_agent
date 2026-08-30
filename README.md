# Kaggriculture Agent

Heuristic agent development for Kaggle's **Kaggriculture** competition.

## Development Approach

The agent is being built incrementally, with an emphasis on understanding the game mechanics and validating one strategy change at a time through local simulation.

Each meaningful strategy change is preserved as a new agent iteration so that it can be evaluated against earlier baselines rather than overwritten.

The current architecture separates the agent into:

1. **Context** — extracts relevant state from the observation.
2. **Task discovery** — identifies possible actions such as planting, watering, harvesting, and clearing weeds.
3. **Strategy** — determines which tasks are desirable and prioritizes them.
4. **Task assignment** — assigns work across the farmer and hired hands while avoiding duplicate tile assignments.
5. **Execution** — moves workers toward assigned tasks and performs actions.
6. **Market policy** — handles seed purchases, crop sales, hiring, and land expansion.

## Agent Iterations

### `agent_v1`

Initial minimal wheat agent.

* Operates only on the farmer's current tile.
* Buys one wheat seed when out of seeds.
* Plants wheat on an empty tile.
* Waters and harvests planted wheat.
* Sells harvested wheat.

### `agent_v2`

First correction to the basic crop lifecycle logic.

* Prioritizes watering before harvesting.
* Delays wheat harvest relative to `agent_v1`.
* Keeps the intentionally simple single-tile strategy.

### `agent_v3`

Introduces the first general task-based architecture.

* Adds `Context` and `Task` data structures.
* Scans the farm to discover planting, watering, harvesting, and weed-clearing tasks.
* Moves the farmer toward tasks instead of acting only on the current tile.
* Separates task discovery, strategy, execution, and market policy.
* Uses task priority first and distance second when choosing work.

### `agent_v4`

Extends the task architecture to multiple workers.

* Assigns tasks to both the farmer and hired hands.
* Prevents multiple workers from being assigned to the same tile.
* Reserves seeds during assignment so workers cannot over-allocate planting tasks.
* Adds worker hiring.
* Avoids planting at the very end of the day.
* Uses crop yield rather than only crop age as the harvest trigger.

### `wheat_agent`

Establishes a more complete single-crop wheat baseline.

* Uses game crop parameters when deciding when wheat is mature.
* Hires multiple hands to increase available labor.
* Buys seeds in larger batches.
* Adds price-aware crop selling.
* Adds land expansion when unlocked land becomes sufficiently occupied.
* Generalizes several crop-specific helpers while still planting only wheat.

### `carrot_agent`

Creates a carrot-specific comparison agent using the same task architecture.

* Plants carrots instead of wheat.
* Adjusts seed purchasing to carrot economics.
* Uses a carrot-specific selling threshold.
* Retains the multi-worker assignment, hiring, and land-expansion framework.

### `dynamic_crop_agent_v0`

Introduces dynamic crop selection and becomes the current baseline.

* Chooses among supported crops using expected **profit per day** based on current market price, seed cost, expected unfertilized yield, and time to maximum yield.
* Avoids crops that cannot mature before the episode ends.
* Adds a day-0 restriction to avoid immediately committing to slow-growing melons.
* Adds `critical_water` as a higher-priority task for crops that have already gone unwatered.
* Gives a worker standing directly on a critical watering task an immediate local override.
* Harvests based on expected unfertilized yield.
* Stops assigning new planting tasks late in the day.
* Tracks escalating daily hire costs when deciding how many hands to hire.
* Buys seeds for the currently selected crop.
* Sells inventory across crop types when prices are acceptable or shed capacity is exhausted.
* Keeps a cash buffer before buying additional land.

### `dynamic_crop_agent_v1` — in development

Focus: **task allocation**.

The current allocator is sequential and greedy: workers are considered one at a time, and each worker takes the highest-priority nearest remaining task. This makes the result dependent on worker ordering and can send one worker to a task that another worker could complete much more efficiently.

The goal of `dynamic_crop_agent_v1` is to make assignment decisions across workers and tasks more coherently while preserving the existing task strategy.

Primary objectives:

* Reduce unnecessary worker travel.
* Avoid assignments where an early worker consumes a task that is much better positioned for a later worker.
* Preserve urgent watering behavior.
* Preserve the one-worker-per-tile constraint.
* Preserve seed reservation for planting tasks.
* Keep task allocation separate from task desirability and crop-selection strategy so allocation changes can be evaluated independently.

## Current Strategy

The current baseline is `dynamic_crop_agent_v0`.

Its behavior includes:

* Dynamically selecting a crop to plant using current crop economics.
* Planting on available land.
* Prioritizing crops at risk from missed watering.
* Watering crops to preserve them and maintain yield.
* Harvesting at expected unfertilized maximum yield.
* Clearing weeds.
* Hiring multiple farm hands while accounting for escalating hire cost.
* Reserving seeds during task assignment.
* Selling harvested crops when market prices are acceptable.
* Expanding the farm when existing unlocked land becomes sufficiently occupied and sufficient cash remains.

## Evaluation

Agents are evaluated locally across fixed random seeds and from both player positions.

Tracked metrics currently include:

* Win rate
* Mean final cash
* Mean margin versus the baseline agent
* Standard deviation of margin
* Watering deaths
* Natural crop decay

Changes are generally tested independently so their effect on performance can be measured before additional strategy is added.

## Current Work

The next iteration is **`dynamic_crop_agent_v1`**, focused specifically on improving worker-to-task allocation.

This iteration is intended to address allocation inefficiency without simultaneously changing crop selection, market policy, or the underlying definition and priority of tasks. That keeps the experiment interpretable: performance changes can be attributed primarily to the allocation strategy.

Later improvements may include market-price forecasting, fertilizer, animals, more sophisticated land-expansion decisions, and learned decision components.
