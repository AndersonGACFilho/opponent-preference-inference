# Project state

Last updated: 2026-11-24. Short paper due **2026-11-30**. See `TECH_DEBT.md` for everything pending.

## Branch / PR status

- `main`: includes the structured package (PR #39 merged).
- Working branch: `2-p2-environment-and-baseline` (P2 deliverables: MOMAland consistency test, Gym environment, MO-PPO baseline, evaluation harness).

## Done

- Exact solver by backward induction and opponent Q-tables for the 15-point omega grid (`src/opponent_inference/solver.py`, `data/q_tables_4x4.npz`).
- Project structured as a uv package; tests (`tests/test_solver.py`) and ruff configured (PR #39).
- Code, tests and README translated to English; `STATE.md` / `TECH_DEBT.md` added.

## In progress

- P2 plan (compressed, since M2 was due 11/22): consistency test and Gym environment on 11/24, train/test split and MO-PPO on 11/25, long training, evaluation harness and baseline results on 11/26.

## Next

1. MOMAland consistency test (`tests/test_momaland_consistency.py`, optional `consistency` dependency group).
2. `env.py` (Gym environment), `splits.py`, `agents/` (policy + PPO), `evaluate.py`, training/eval scripts.
3. Hand the policy/environment interface to P3 (oracle) and P4 (omega-hat agent).

## Key decisions

- Opponent assumes the agent has the same omega as itself (decision of 11/18); its table comes from `solve_spe(omega, omega)`, the same table serves X and O.
- Objectives: win, win fast, column control (average of MOMAland's per-column rewards); linear utility `omega . R`.
- The environment never computes omega-hat; the true opponent omega is exposed only in oracle mode and as a training label.
- MO-PPO hyperparameters follow Appendix A of the base paper (64-unit MLP, lr 0.01 with decay, GAE 0.95, clip 0.2, entropy 0.01, gamma 1, 4 envs, 256 steps, 4 minibatches, 4 epochs). Fallback if PPO does not learn: tabular Q-learning conditioned on the omega index.
- Everything in the project is written in English; commit messages are plain one-liners.

## How to run

```bash
uv sync                                      # environment
uv run pytest                                # tests
uv run ruff check .                          # lint
uv run python -m opponent_inference.solver   # solver demo (134,289 non-terminal states)
```

Results are written to `results/` (git-ignored); record headline numbers here.

## Environment notes

12 cores, 15 GB RAM, GTX 1660 Ti (CUDA available, but the small MLP runs faster on CPU; run seeds in parallel). `gh` is not installed; PRs are opened by hand.
