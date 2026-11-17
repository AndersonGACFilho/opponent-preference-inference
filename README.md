# Opponent Preference Inference in Multi-Objective Board Games: Beyond Action Prediction

Inferring opponent preferences in multi-objective board games: going beyond action prediction to model the trade-offs and objective weights that drive opponent behavior.

> **Status:** short paper with preliminary results, due **11/30/2026**, to be expanded later into a full paper.
>
> **Paper (Overleaf):** _link to be added_ ([#34](https://github.com/AndersonGACFilho/opponent-preference-inference/issues/34))

## Overview

In multi-objective multi-agent reinforcement learning (MOMARL), each player weighs several objectives through a preference vector ω. Existing opponent modeling in this setting predicts the opponent's **next action**. This project instead infers the opponent's **preference vector ω̂** from its moves and conditions the agent's policy on (ω, ω̂).

**Base paper:** Obiang & Rădulescu (2026), *Opponent Modelling for MOMARL in Sequential Settings* (MODeM @ IJCAI-ECAI 2026). It applies action-prediction opponent modeling to MO-Connect4 with fixed per-player utilities.

### What changes from the base paper

1. **What is modeled:** the opponent's preference vector ω, not its next action.
2. **Generalization:** evaluation against opponents whose preferences were not seen during training.
3. **Evaluation:** an exact reference by backward induction on the 4×4 board, so regret can be measured exactly.

## Hypothesis

**H1:** against opponents with ω unseen during training, an agent that infers ω̂ and uses a policy conditioned on (ω, ω̂) achieves higher utility (lower regret) than MO-PPO without an opponent model. An oracle that receives the opponent's true ω sets the upper bound.

Expected ordering:

```
exact best response ≥ oracle ≥ ω̂ agent ≥ MO-PPO
```

- Gap between the exact best response and the oracle: cost of the RL approximation.
- Gap between the oracle and the ω̂ agent: cost of not knowing the opponent's preferences.

## Method

| Component | Short paper setup |
|---|---|
| **Environment** | MO-Connect4 4×4 with three objectives per player: win, win fast and column control (MOMAland column average) |
| **Utility** | Linear, u = ω·R (SER and ESR coincide) |
| **Exact solver** | Backward induction for every (ω₁, ω₂) pair, ~134k non-terminal states, solved in seconds |
| **Opponents** | Exact policies for ω on a simplex grid (step 0.25, 15 points) with softmax noise; ω split into train and test sets |
| **Metrics** | Mean utility, regret against the exact best response, win rate, \|ω̂ − ω\| error per move; 5 seeds, 95% CI |

### Agents

| Agent | Input | Role |
|---|---|---|
| **MO-PPO** | Board + own ω | Baseline without an opponent model |
| **ω̂ agent** | Board + own ω + inferred ω̂ | Proposed method: supervised inference module (inspired by DWPI) over the opponent's move history |
| **Oracle** | Board + own ω + true opponent ω | Upper bound with the same policy architecture as the ω̂ agent |

All three agents face the same population of exact opponents, for every own-ω × test-opponent-ω combination. Learning agents do not play each other in the short paper.

### What the environment provides

- **Observation (all agents):** board from the agent's point of view, own ω and column mask.
- **`info` at every step:** the opponent's move.
- **True opponent ω:** given only as input to the oracle and as a training label for the ω̂ inference module, never during evaluation.
- **ω̂:** computed inside the ω̂ agent; the environment never computes or provides it.

## Preliminary finding

On MO-Connect4 4×4 with only *win* and *win fast*, the optimal outcome is the same (a draw) for every preference pair tested, so preference heterogeneity does not matter. Adding *column control* makes the outcome depend strongly on the preference pair: a player that prioritizes winning beats one that prioritizes columns, and the result flips when roles are swapped. The study therefore uses all three objectives.

## Scope

**Short paper:** everything above.

**Extended version:** MO-PPO-OM baseline (Obiang & Rădulescu's architecture with ω as input), self-play with learning opponents, 5×4 board, MO-Breakthrough, D3PO and non-linear utilities.

## Project management

Work is tracked in the [GitHub Project](https://github.com/users/AndersonGACFilho/projects/18) (Kanban, Roadmap, Goals, Epics, Priorities and one board per person).

### Milestones

| Milestone | Due | Goal |
|---|---|---|
| M1: Validation & solver | 11/17 | Proposal validated, exact solver tables ready |
| M2: Environment & baseline | 11/22 | Gym environment and MO-PPO baseline with results |
| M3: Agents trained | 11/24 | ω̂ inference module, oracle and ω̂ agent trained |
| M4: Results | 11/26 | Experiments, results table and figures |
| M5: Paper draft | 11/28 | All sections written and reviewed |
| M6: Submission | 11/30 | Short paper submitted |

### Roles

| Role | Responsibility | Paper sections |
|---|---|---|
| **P1** | Exact solver, policy and best-response tables, results, figures | Results |
| **P2** | MOMAland consistency test, Gym environment, MO-PPO | Methodology and setup |
| **P3** | Obiang's thesis, oracle agent | Related work |
| **P4** | Professor validation, ω̂ agent, coordination, submission | Introduction, discussion, conclusion, abstract |

### Conventions

- **Issue types:** Epic, Feature, Task, Experiment, Writing, Research, Bug.
- **Templates:** Task, Experiment, Epic and Bug forms in [`.github/ISSUE_TEMPLATE`](.github/ISSUE_TEMPLATE); each sets its type and adds the issue to the Project.
- **Epics** group tasks as sub-issues; **blocked by / blocking** links follow the critical path (solver → environment → MO-PPO → agents → experiments → writing → submission).
- **Labels:** `P1`–`P4` for the people involved, plus area labels (`solver`, `environment`, `agent`, `writing`, `coordination`, `experiment`).
- **Fields:** Status, Priority, Effort, Estimate (planning-poker points), Start date, Target date, Week, Area and Owner.

### Risks and cut lines

- **PPO does not learn in time:** switch to tabular Q-learning conditioned on the ω index (the 4×4 board fits in memory).
- **ω̂ is not informative:** report it as a negative result, with the error-per-move curve.
- **The framing is not approved:** the solver and environment serve any variant; only the hypotheses change.

## References

- Obiang & Rădulescu (2026). *Opponent Modelling for MOMARL in Sequential Settings.* MODeM @ IJCAI-ECAI 2026.
- Rădulescu et al. (2021).
- MOMAland.
- Lu et al. DWPI.
- D3PO.

Full citations will be added with the paper.

## License

[MIT](LICENSE)
