# MO-PPO baseline: results and how to read them

This is the baseline the omega-hat agent will be compared against: a PPO agent that sees the board
and its own omega, and has no model of the opponent. It was trained for 5 seeds x 2M steps on the
12 train opponents and evaluated on the 3 held-out test opponents (`[9, 10, 11]`). Commands to
reproduce everything are in the README, section "Running experiments".

## What was measured

Every trained agent plays 100 games for each combination of own omega (all 15 grid points),
opponent and side (X or O). In evaluation the agent always picks its most likely column, while
the opponents keep sampling from their softmax policy (`tau = 0.1`). Metrics are first averaged
inside each seed, then averaged across the 5 seeds; the +- values are 95% confidence intervals
across seeds. So they say how much two independently trained agents differ, not how noisy a
single game is.

## Results

| Agent | Opponents | Mean utility | Win | Draw | Loss |
|---|---|---|---|---|---|
| MO-PPO (5 seeds) | test `[9, 10, 11]` | +0.147 +- 0.013 | 36.6% | 55.6% | 7.9% |
| Random agent (1 seed) | test `[9, 10, 11]` | -0.134 | 9.8% | 46.2% | 44.0% |
| MO-PPO (5 seeds) | train (12 opponents) | +0.087 +- 0.005 | 19.3% | 74.1% | 6.6% |
| Random agent (1 seed) | train (12 opponents) | -0.180 | 6.3% | 41.8% | 51.9% |

## How to read them

- **The agent clearly learned.** Against the test opponents it loses 7.9% of its games, against
  44.0% for a random player, and its utility goes from -0.13 to +0.15. The five seeds agree on
  utility (CI of +-0.013).
- **Losses are the least stable number.** The loss rate has a CI of +-0.052 around a mean of
  0.079, so some seeds lose noticeably more often than others.
- **Learning is slow after the first million steps.** In the training log, the running utility
  climbs from about 0.05 to about 0.09 and then moves sideways; the win rate stays between 12%
  and 19%. Policy entropy fell from about 0.09 to 0.04 over training, so the policy is close to
  deterministic. This matches the entropy collapse already noted in TD-15.
- **Most games end in a draw.** On 4x4 against strong opponents a draw is the usual outcome, so
  there may be little room for any agent, including the omega-hat agent, to improve on this
  baseline. Only the regret against the exact best response will show how much room is left.

## What these numbers do NOT show

- **They do not prove or rule out a generalization gap.** The raw test utility (+0.147) is higher
  than the train utility (+0.087), but the two opponent sets have different difficulty, so the raw
  numbers cannot be compared. The random agent gives a rough yardstick for each set: MO-PPO beats
  it by +0.267 on the train opponents and by +0.281 on the test opponents, which is similar, so
  there is no sign of a drop on unseen opponents. This is only a proxy, because a random player
  is a weak reference. The right measure is regret against the exact best response (TD-01),
  computed per opponent.
- **They do not say anything about the proposed method yet.** The oracle and the omega-hat agent
  do not exist yet (TD-04, TD-05), so hypothesis H1 has not been tested. The random agent row is
  a single seed, so it has no confidence interval.
- **They are not comparable with the base paper's numbers.** The paper uses a larger board and two
  fixed utilities; this study uses 4x4 and a grid of 15 omegas.

## Where the raw data is

Everything under `results/` is ignored by git. After running the commands in the README you get:

- `results/mo_ppo/seed<N>/policy.pt` and `train_log.csv`: checkpoint and training curve per seed.
- `results/eval.csv`, `results/mo_ppo/eval_test.csv`, `results/mo_ppo/eval_train.csv`: one row per
  (own omega, opponent, side, seed). Use them to see which combinations the agent handles badly.
