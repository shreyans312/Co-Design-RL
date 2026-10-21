# Original synthetic-environment design sweep

This directory preserves the complete grid evaluation from 18 October 2026, before the Cauz dataset/environment update. It is a historical reference, not a benchmark on the new CSV environment.

- Environment: original synthetic 2026 dataset, data seed 42
- Controller: frozen final custom DDPG actor from the frozen-scoring experiment, training seed 43, iteration 500
- Grid: all 196 PV/battery combinations from 0.5 to 7.0 in steps of 0.5
- Selection: 32 common validation scenarios per design
- Assessment: 320 fresh common draws on existing validation weeks, seed 70000000
- No training updates or early elimination during this sweep

## Results

The grid selected 7.0 kWp PV and 7.0 kWh battery at 76.988 CHF mean weekly selection cost. On the separate assessment, it cost 79.110 CHF versus 92.807 CHF for the original selected design. The paired saving was 13.697 CHF, with scenario standard error 0.212 CHF. The winner lies at the grid boundary.

## Files

- `grid_results.csv`: all 196 designs, mean selection cost and scenario standard error
- `ranked_designs.csv`: the same designs sorted by cost
- `selection_episodes.csv`: all 6,272 grid scenario results
- `assessment_episodes.csv` and `paired_differences.csv`: final comparison and paired savings
- `protocol.json`: original settings, checkpoint identity and source hashes
- `selection.json` and `result.json`: selected design and assessment summary
- `design_grid_results.png` / `.svg`: summary plot
- `report.md`: original experiment report

These files were copied without changing their contents from `runs/ddpg_frozen_scoring_seed43_vssh7N/grid_0p5_to7_seed70000000/`. Paths in the original protocol and report refer to that experiment location. The checkpoint is not duplicated here.

Compare algorithms on the same new dataset, splits and scenarios before drawing conclusions about algorithm improvements. Do not compare these old costs directly with new-environment costs. Preserve the year and units supplied by the new source rather than silently relabelling its timestamps.
