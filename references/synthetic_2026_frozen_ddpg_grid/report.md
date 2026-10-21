# Frozen-actor design grid

All 196 Cartesian combinations of PV capacity 0.5–7.0 kWp and battery capacity 0.5–7.0 kWh, in 0.5 increments, were evaluated using the final iteration-500 frozen-scoring DDPG checkpoint (training seed 43). No actor, critic, target-network or mixture updates were performed. Their unchanged states and the checkpoint hash were verified.

## Selection and assessment

Every design received the same 32 validation scenarios, seeds 1000042–1000073. The lowest mean simulated weekly total cost selected PV 7.0 kWp and battery 7.0 kWh, at 76.988 CHF. The original selected design, PV 2.728665 kWp and battery 5.493161 kWh, scored 90.639 CHF on those scenarios.

The grid choice was saved before assessing both designs on 320 fresh common scenario draws, seeds 70000000–70000319. Assessment was not used to reselect the design.

- Original design: 92.807 CHF mean weekly cost
- Grid-selected design: 79.110 CHF mean weekly cost
- Paired saving: 13.697 CHF per week, scenario standard error 0.212 CHF

This demonstrates a substantially better design under the same trained actor than the original density-based selected design. It identifies a design-search/selection shortfall; it does not establish that the actor failed to learn. The winning design is at both upper edges of this requested grid, so this sweep does not locate a global optimum. These are new scenario draws on existing validation weeks, not an unseen-week test or multiple training seeds.

## Records and budget

`selection_episodes.csv` contains 6,272 individual grid evaluations. `grid_results.csv` and `ranked_designs.csv` contain their means and scenario standard errors. `assessment_episodes.csv` contains 640 final assessments; `paired_differences.csv` preserves scenario-level savings. The 32 original-design selection checks bring the total to 6,944 episodes, or 1,166,592 hourly simulator steps.

All 196 combinations, seed matching, per-design means and paired assessment statistics were independently checked. The stochastic model was not changed. The critic was loaded and frozen but simulated costs, not critic predictions, determined the ranking.

The run finished before the user's clarification requesting early elimination was handled. No designs were eliminated and no early-stopping rule was applied retrospectively. Any future elimination rule must be specified before running, operate between complete episodes, preserve scenario matching and label incomplete evaluations explicitly.

Figure: `design_grid_results.png` (also available as SVG). Reproduction scripts are `../evaluate_grid.py` and `../plot_grid.py`; the evaluator refuses to overwrite this existing output directory.
