# Project state

Last updated: 2026-11-24. Short paper due **2026-11-30**. See `TECH_DEBT.md` for everything pending.

## Branch / PR status

- `main`: includes the structured package (PR #39 merged).
- Working branch: `2-p2-environment-and-baseline` (P2 deliverables). All code is committed; the PR is not opened yet.

## Done

- Exact solver by backward induction and opponent Q-tables for the 15-point omega grid (`src/opponent_inference/solver.py`, `data/q_tables_4x4.npz`).
- Project structured as a uv package; tests and ruff configured (PR #39). Code, tests and README are in English.
- **MOMAland consistency test** (`tests/test_momaland_consistency.py`, group `consistency`): 2,000 random 4x4 games, legal moves, turn order, termination and reward vectors match MOMAland's `moconnect4` (column objectives on). Verified that the test fails if the column score is broken.
- **Environment** (`env.py`): `ConnectFourOpponentEnv` (vector reward, softmax opponent, action mask, agent as X or O), `flatten_obs`, `LinearScalarization`. About 8,300 steps/s per core.
- **Train/test split** (`splits.py`): groups omegas by identical behaviour (12 groups, confirmed), splits whole groups. Default seed 0 gives train `[0..8, 12, 13, 14]`, test `[9, 10, 11]` = omegas (0.25,0,0.75), (0,1,0), (0,0.75,0.25). Proposal, to be confirmed (TD-03).
- **MO-PPO baseline** (`agents/policy.py`, `agents/ppo.py`, `scripts/train_mo_ppo.py`): paper hyperparameters; trained 5 seeds x 2M steps (about 10 min each, CPU).
- **Evaluation harness** (`evaluate.py`, `scripts/evaluate_agent.py`): per-cell CSV and summary with 95% CI over seeds.

## Baseline results (MO-PPO, tau = 0.1, 100 games per cell, greedy policy)

Test opponents `[9, 10, 11]`, all 15 own omegas, both sides, 5 training seeds. Regret is not available yet (TD-01).

| Agent | Mean utility | Win rate | Draw rate | Loss rate |
|---|---|---|---|---|
| MO-PPO | +0.147 +- 0.013 | 0.366 +- 0.043 | 0.556 | 0.079 |
| Random (1 seed) | -0.134 | 0.098 | 0.462 | 0.440 |

Files (git-ignored): `results/mo_ppo/seed<N>/{policy.pt,train_log.csv}`, `results/mo_ppo/eval_test.csv`. Regenerate with the commands below.

## Handoff interface for P3 (oracle) and P4 (omega-hat agent)

- Same architecture: `PolicyNet(input_dim, n_actions)` and trainer `train_ppo(envs, cfg, obs_fn)`; only `obs_fn` (observation dict -> input vector) changes.
- Oracle: build the env with `expose_opponent_omega=True`; `flatten_obs` then appends the true opponent omega automatically (`flat_obs_dim(h, w, True)` is the input size).
- Omega-hat agent: use `expose_opponent_omega=True` only during training to get the label (`info["opponent_omega"]`); the agent's `obs_fn` must append its own estimate omega-hat instead of the true value, and evaluation must use `expose_opponent_omega=False`.
- Each step, `info["opponent_move"]` gives the opponent's last move (also at `reset` when the opponent moves first), and `obs["board"]` is the board after it; this is the data for the history that the inference module consumes.
- Evaluate with `evaluate_policy(policy_fn, tables, opponent_ids, own_omega_ids, n_games, tau, seed)` and `summarize` so all agents are compared identically.

## Next

1. Open the PR for this branch (the user pushes; `gh` is not installed).
2. P1: exact best response and regret (TD-01), tau calibration (TD-02), confirm the split (TD-03).
3. P3: oracle agent. P4: omega-hat inference module and agent.
4. Run all agents with the same evaluation, build the results table and figures (TD-06), write the paper (TD-07).

## Key decisions

- Opponent assumes the agent has the same omega as itself (decision of 11/18); its table comes from `solve_spe(omega, omega)`, the same table serves X and O.
- Objectives: win, win fast, column control (average of MOMAland's per-column rewards); linear utility `omega . R`; reward only at the end of the game.
- The environment never computes omega-hat; the true opponent omega is exposed only when `expose_opponent_omega=True`.
- MO-PPO hyperparameters follow Appendix A of the base paper. PPO learns, so the tabular Q-learning fallback is not needed.
- Everything in the project is written in English; commit messages are plain one-liners.

## How to run

```bash
uv sync                                      # environment (add --group consistency for MOMAland)
uv run pytest                                # tests (consistency test is skipped without the group)
uv run ruff check .                          # lint
uv run python -m opponent_inference.solver   # solver demo (134,289 non-terminal states)
for s in 0 1 2 3 4; do uv run python scripts/train_mo_ppo.py --seed $s & done; wait
uv run python scripts/evaluate_agent.py --checkpoints results/mo_ppo/seed*/policy.pt --random --opponents test
```

## Environment notes

12 cores, 15 GB RAM, GTX 1660 Ti (CUDA available, but the small MLP runs faster on CPU; seeds run in parallel). `gh` is not installed; PRs are opened by hand.
