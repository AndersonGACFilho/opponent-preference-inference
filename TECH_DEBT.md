# Tech debt

Everything still pending or imperfect in the project. Update when an item is created or resolved
(move resolved items to the bottom section). Owners refer to the roles in the README (P1–P4).

Priority: **H** = blocks the short paper (due 2026-11-30), **M** = should be done before submission,
**L** = can wait for the extended version.

## Open

| ID | Area | Owner | Pri | Description |
|---|---|---|---|---|
| TD-01 | Evaluation | P1 | H | Exact best response against each stochastic opponent (opponent nodes: average weighted by `action_probs`; agent nodes: max) and the regret metric. Needed for the H1 comparison. |
| TD-02 | Opponents | P1 | H | Calibrate the softmax temperature `tau` (default in the env is a placeholder). |
| TD-03 | Opponents | P2 / P1 / P4 | H | Confirm the train/test split rule (whole groups of identical deterministic policies stay on the same side; 15 omegas give only 12 distinct behaviours). `splits.py` is a proposal. |
| TD-04 | Agents | P3 | H | Oracle agent (same policy architecture as the omega-hat agent, plus the true opponent omega as input). |
| TD-05 | Agents | P4 | H | Omega-hat inference module (supervised network over the move history) and omega-hat agent; error-per-move metric. |
| TD-06 | Experiments | P1 | H | 5 seeds x (own omega x test opponent omega) experiments, results table, figures (regret per omega, omega-hat error per move, regret map). |
| TD-07 | Paper | P1–P4 | H | Write all sections; complete the references; add the Overleaf link to the README (#34). |
| TD-08 | Validation | P4 | M | Confirm the framing with the professor. |
| TD-09 | Solver | P1 | M | Game rules (`play`, `has_four`, `terminal_outcome`, ...) live inside `solver.py`; consider splitting them into `game.py`. Not done yet to avoid conflicts with P1's edits. |
| TD-10 | Data | P1 | L | `data/q_tables_4x4.npz` still stores the old Portuguese `tie_break` string (metadata only, Q-values unaffected). Regenerate with `build_q_tables` (~2 min) when convenient. |
| TD-11 | Tooling | any | M | No CI workflow (`.github/` only has issue templates). Add one running `uv run pytest` and `uv run ruff check .`. |
| TD-12 | Tooling | any | L | Ruff only checks `E`, `F`, `I`. Enabling `UP` requires modernizing `solver.py` (`Dict` -> `dict`, `lru_cache` -> `cache`). |
| TD-13 | Tooling | any | L | `torch` pulls the CUDA wheels (large `.venv`). Training uses CPU only; consider a CPU-only torch index. |
| TD-14 | Extended version | all | L | MO-PPO-OM baseline, self-play with learning opponents, 5x4 board, MO-Breakthrough, D3PO, non-linear utilities. |
| TD-15 | Agents | P2 | M | MO-PPO keeps the paper's lr 0.01, which collapses policy entropy within ~40k steps. A single-seed diagnostic with lr 0.003 gave comparable test utility (+0.089 vs +0.105), so it was not changed; revisit if oracle/omega-hat agents behave differently. |
| TD-16 | Evaluation | P2 / P1 | M | `evaluate_policy` plays the greedy (argmax) policy, 100 games per (own omega, opponent, side) cell, own omegas only on the 15-point grid. Decide whether the paper needs stochastic evaluation or off-grid omegas. |
| TD-17 | Environment | P2 | L | `ConnectFourOpponentEnv` is hard-wired to the opponent assumption "same omega as the agent" through the Q tables; opponents that model the agent's true omega would need new tables. |

## Resolved

| ID | Description | Resolved |
|---|---|---|
| TD-R1 | Project not structured as a package (no `pyproject.toml`, uv, tests, `.gitignore` for PDFs/results). | 2026-11-24 (PR #39) |
| TD-R2 | Code and docs partly in Portuguese; project policy is English only. | 2026-11-24 |
