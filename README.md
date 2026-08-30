# Kaggriculture Agent

Heuristic agent development for Kaggle's **Kaggriculture** competition.

## Development Approach

The agent is being built incrementally, with an emphasis on understanding the game mechanics and validating one strategy change at a time through local simulation.

The current architecture separates the agent into:

1. **Context** — extracts relevant state from the observation.
2. **Task discovery** — identifies possible actions such as planting, watering, harvesting, and clearing weeds.
3. **Strategy** — determines which tasks are desirable and prioritizes them.
4. **Task assignment** — assigns work across the farmer and hired hands while avoiding duplicate tile assignments.
5. **Execution** — moves workers toward assigned tasks and performs actions.
6. **Market policy** — handles seed purchases, crop sales, hiring, and land expansion.

## Current Strategy

Development began with simple single-crop agents and progressed toward a more general farming framework.

Current behavior includes:

* Planting a selected crop on available land.
* Daily watering to preserve crops and maximize yield.
* Harvesting crops at an appropriate maturity point.
* Clearing weeds.
* Hiring multiple farm hands each day.
* Reserving seeds during task assignment to prevent multiple workers from attempting to use the same seed.
* Selling harvested crops when market prices are acceptable.
* Expanding the farm when existing unlocked land becomes sufficiently occupied.
* Using current market information as groundwork for future dynamic crop selection.

## Evaluation

Agents are evaluated locally across fixed random seeds and from both player positions.

Tracked metrics include:

* Win rate
* Mean final cash
* Mean margin versus the baseline agent

Changes are generally tested independently so their effect on performance can be measured before additional strategy is added.

## Current Work

The next major step is **dynamic crop selection**.

Rather than planting a fixed crop, the agent will compare crop economics using factors such as:

* Current market price
* Seed cost
* Expected unfertilized yield
* Time to production
* One-time versus ongoing harvest behavior

Later improvements may include better worker scheduling, market-price forecasting, fertilizer, animals, and more sophisticated land-expansion decisions.
