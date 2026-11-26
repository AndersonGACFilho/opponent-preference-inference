# Opponent Preference Inference: Design

Date: 2026-11-26 · João Augusto

The agent infers the opponent's preference vector from its observed moves, then plays a best response to it. This design covers the competitive two-agent case first.

Source: Claude Docs document "Opponent Preference Inference: Design" (artifact `ECHVVayWRtZfkjgiXjojAz`, rev 14). This file is a copy for the repo; the Claude Docs version is the live one.

## Summary

The agent watches the opponent's moves, keeps a posterior over the preference vector ω the opponent follows, and plays a policy conditioned on that posterior. The first version targets a competitive two-agent game: MO-Connect4 on a 4×4 board, where the opponent is one of the exact softmax policies already in the repo.

- **In scope:** inferring the opponent's ω from its moves, and a best-response policy against it.
- **In scope:** evaluation against the same opponents as the MO-PPO baseline, with regret once TD-01 exists.
- **Out of scope for v1:** learning opponents, multi-agent teams, partial observability.

A result counts as success when the agent's utility beats the MO-PPO baseline on the held-out opponents.

## Background

The paper (Obiang & Rădulescu, 2026) gives three algorithms, but each assumes the agent already knows the peer's utility. The paper states that the peer's objective is "otherwise unidentifiable from actions alone" (Sec. 4). v1 must replace that assumption with inference.

- **MO-PPO-OM** (Sec. 4.1): a neural opponent network predicts the peer's next action. No gain over MO-PPO (Sec. 7).
- **MO-POLA** (Sec. 4.2): looks one update ahead on the peer's learning, using the peer's utility.
- **MO-Meta-MAPG** (Sec. 4.3): meta-learns over a chain of updates. In zero-sum MO-Connect4 it beats MO-POLA (Fig. 5).

The repo already provides the pieces v1 needs:

- The opponent is `softmax(Q_k / tau)` over exact tables for 15 grid ω values (`env.py`, `solver.py`).
- The opponent's move arrives in `info["opponent_move"]` at every step.
- `PolicyNet` and `train_ppo` take an `obs_fn`, so a new input vector needs no new trainer (`agents/ppo.py`).
- Only 12 distinct behaviours exist across the 15 grid values (TD-03), so identification stops at behaviour groups.

## Inference module

The opponent's policy class is known and finite, so inference is Bayesian filtering over the hypotheses. Each hypothesis is one grid index k, with its exact softmax policy π_k.

$$
P(k \mid h_t) \propto P(k) \prod_{\tau \le t} \pi_k(a^{\mathrm{opp}}_\tau \mid s_\tau)
$$

- **Prior:** uniform over the hypothesis set (v1).
- **Update:** one multiplication per opponent move, computed in log space.
- **Output:** the posterior vector over k, its MAP estimate ω̂, and the posterior itself as the policy input.
- **Hypotheses are the 15 raw grid indices, not behaviour groups.** Correction: an earlier draft said hypotheses in one behaviour group give identical likelihoods. That holds only at τ = 0. At τ = 0.1 the move probabilities of two omegas in group {1, 3, 6} differ by up to 0.39, so the filter must keep raw k. Group mass is derived by summing the posterior over a group, for reporting.
- **Assumption:** τ is known. TD-02 leaves it as a placeholder.
- **Limitation:** the opponent's table is `solve_spe(ω, ω)` (STATE.md), so the likelihood ignores the agent's own ω.

Phase 0 check, measured (`scripts/phase0_identification.py`, 100 games per opponent, random agent, full-grid hypotheses): the mass on the true behaviour group does not reach 0.9 within the 8 opponent moves a 4×4 game allows. Pass rates: 0.00 at τ = 0.1, 0.01 at τ = 0.02, 0.00 at τ = 0.05, 0.04 at τ = 0 (deterministic opponent). The opponent's moves do not identify ω within one game on this board, and τ does not fix it. The gate needs a decision (see the open decisions).

## Best-response agent

The agent is the same `PolicyNet` trained by `train_ppo`. Only the observation function changes: it appends the belief from the inference module to board and own ω. This is the same pattern the oracle and the ω̂ agent use (STATE.md, handoff interface).

- **Belief input, three options:** the full posterior (recommended, it keeps uncertainty), the MAP one-hot, or both.
- **Training opponents:** sampled from the train groups only (`splits.py`).
- **Episode start:** the filter starts from the uniform prior at the start of each episode, so no information leaks across episodes.

Reference bounds, from the README's expected ordering:

| Agent | Input | Role |
|---|---|---|
| Exact best response | full knowledge of the opponent | upper bound (TD-01, not built yet) |
| Oracle | true opponent ω (`expose_opponent_omega=True`) | ceiling for inference |
| ω̂ agent (this design) | posterior from observed moves | the method under test |
| MO-PPO | own ω only | floor |

Open question: a feed-forward policy with a belief input may not reach the exact best response. The gap to the oracle measures the cost of inference, not of the policy class.

## Evaluation

All agents use the same harness, so the numbers are comparable: `evaluate_policy` for each cell, then `summarize` across seeds.

| Metric | Definition | Source |
|---|---|---|
| Mean utility | ω · R at the end of the game, per cell | `evaluate.py` |
| Win, draw, loss rate | Shares of games per cell | `evaluate.py` |
| Inference error | Posterior mass on the true behaviour group, per move | new |
| Regret | Gap to the exact best response | blocked by TD-01 |

Protocol: 15 own ω × test opponents × both sides × 100 games per cell, greedy policy, 5 training seeds, 95% CI over seeds. Random and MO-PPO run through the same cells.

No-leakage rules:

1. Test opponents never enter the training set.
2. Headline numbers come from the test split only; train results are diagnostics.
3. The oracle's true ω input is off at evaluation (`expose_opponent_omega=False`).
4. The filter's hypothesis set is declared in the report. Whether it includes test tables is open decision D1.

## Phases

Four phases, three gates; dates not set. The phases run left to right. A gate must pass before the next phase starts, and each gate uses a check that can fail.

## Open decisions

These four decisions block implementation. Each has a recommended default, which the team can overturn.

| ID | Decision | Options | Recommended default | Owner | Blocks |
|---|---|---|---|---|---|
| D1 | Hypothesis set of the filter | Train groups only, or full 15-point grid | **Chosen:** train groups for headline results; full grid as a sensitivity run | P4, with P1 | Phase 1 and evaluation |
| D2 | Meaning of "dominates" | Utility against that opponent, or regret | Mean utility now; add regret once TD-01 exists | P1 | Evaluation |
| D3 | Scope of the first version | Fixed opponents, or a learning peer | **Fixed opponents first (Phases 0 to 2)**, with a placeholder for the learning peer | P4, with the professor (TD-08) | Phase 3 |
| D4 | Value of τ | Calibrate from data, or keep the placeholder | Calibrate before Phase 0 | P1 (TD-02) | Phase 0 |

## Risks

- **The advantage comes from the tables, not from inference.** The filter uses exact policies, which is stronger knowledge than the paper's agents had. Mitigation: report the oracle and a run with a learned predictor in place of the filter.
- **τ is wrong.** A mismatched softmax temperature miscalibrates the posterior. Mitigation: sweep τ in Phase 0 before any headline run.
- **Behaviour groups leak across the split.** Two omegas that play the same belong on the same side (TD-03). Mitigation: split by group, as `splits.py` already does.
- **The belief policy stays far from the exact best response.** The result is then an honest negative with the gap reported. Mitigation: report the gap to the oracle and to the exact best response.

## Sources

Paper: Obiang & Rădulescu (2026), *Opponent Modelling for Multi-Objective Multi-Agent Reinforcement Learning in Sequential Settings*. Local copy: `6_Opponent_Modelling_for_Multi.pdf`. Sections used: 1, 2.1, 2.2, 4 (assumptions), 4.1 to 4.3, 5, 6, 7 and Appendix A.

Repo, `main` at 74b7af5:

- `README.md`: research framing, hypotheses, expected ordering, roles
- `STATE.md`: handoff interface and key decisions (opponent assumes the agent's ω)
- `TECH_DEBT.md`: TD-01, TD-02, TD-03, TD-05, TD-08, TD-16, TD-17
- `src/opponent_inference/env.py`, `agents/policy.py`, `agents/ppo.py`, `evaluate.py`, `splits.py`, `solver.py`
