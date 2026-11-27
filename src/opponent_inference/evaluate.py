"""Evaluation harness shared by every agent (MO-PPO baseline, oracle, omega-hat agent).

`evaluate_policy` plays an agent against fixed opponents for every combination of
(own omega, opponent omega, side) and returns one row per cell. `summarize` averages over
training seeds and adds a 95% confidence interval (Student t over seeds).

The regret with respect to the exact best response is NOT computed here yet; it needs the
best-response module (TECH_DEBT.md, TD-01). When it exists, merge its values by
(own_omega_id, opponent_id, side).
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from scipy import stats

from .env import ConnectFourOpponentEnv
from .solver import QTables

PolicyFn = Callable[[Dict[str, np.ndarray]], int]
EnvWrapper = Callable[[ConnectFourOpponentEnv], gym.Env]
SIDES = ("X", "O")
METRICS = ("mean_utility", "win_rate", "draw_rate", "loss_rate")


def random_policy(seed: int = 0) -> PolicyFn:
    """Uniformly random legal move (sanity baseline)."""
    rng = np.random.default_rng(seed)

    def policy(obs: Dict[str, np.ndarray]) -> int:
        return int(rng.choice(np.flatnonzero(obs["action_mask"])))

    return policy


def evaluate_policy(
    policy_fn: PolicyFn,
    tables: QTables,
    opponent_ids: Iterable[int],
    own_omega_ids: Iterable[int],
    n_games: int,
    tau: float,
    seed: int = 0,
    expose_opponent_omega: bool = False,
    env_wrapper: Optional[EnvWrapper] = None,
) -> pd.DataFrame:
    """Play `n_games` per (own omega, opponent, side) cell; one DataFrame row per cell.

    `own_omega_ids` and `opponent_ids` index the omega grid of `tables`. Ties and draws count
    as draws (win_rate + draw_rate + loss_rate = 1). `mean_utility` is `omega . R` of the
    agent's own omega, averaged over the games. `env_wrapper` (e.g. `BeliefWrapper`) is applied
    to each cell's environment, so the policy sees whatever observation the wrapper adds.
    """
    rows = []
    for j in own_omega_ids:
        for k in opponent_ids:
            for side_idx, side in enumerate(SIDES):
                cell_seed = int(np.random.SeedSequence([seed, j, k, side_idx]).generate_state(1)[0])
                env: gym.Env = ConnectFourOpponentEnv(
                    tables, [k], own_omega=tables.weights[j], tau=tau, agent_side=side,
                    expose_opponent_omega=expose_opponent_omega,
                )
                if env_wrapper is not None:
                    env = env_wrapper(env)
                utilities, wins, losses = [], 0, 0
                for game in range(n_games):
                    obs, _ = env.reset(seed=cell_seed if game == 0 else None)
                    done = False
                    while not done:
                        obs, reward, done, _, info = env.step(policy_fn(obs))
                    utilities.append(info["utility"])
                    wins += reward[0] > 0
                    losses += reward[0] < 0
                rows.append({
                    "own_omega_id": j, "opponent_id": k, "side": side, "n_games": n_games,
                    "mean_utility": float(np.mean(utilities)),
                    "win_rate": wins / n_games,
                    "draw_rate": (n_games - wins - losses) / n_games,
                    "loss_rate": losses / n_games,
                })
    return pd.DataFrame(rows)


def _ci95(values: pd.Series) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))


def summarize(df: pd.DataFrame, by: Sequence[str] = ()) -> pd.DataFrame:
    """Mean and 95% CI half-width over seeds of each metric, grouped by the columns in `by`.

    `df` must have a `seed` column (one value per trained agent). Within each seed the metrics
    are first averaged over every other column (own omega, opponent, side), so the CI measures
    variability between trained agents, not between games.
    """
    by = list(by)
    per_seed = df.groupby(["seed", *by])[list(METRICS)].mean().reset_index()
    if by:
        grouped = per_seed.groupby(by)
    else:
        grouped = per_seed.assign(_all=0).groupby("_all")
    out = grouped[list(METRICS)].agg(["mean", _ci95])
    out.columns = [
        f"{metric}_{'ci95' if stat == '_ci95' else stat}" for metric, stat in out.columns
    ]
    out["n_seeds"] = grouped["seed"].nunique()
    return out.reset_index(drop=not by)
