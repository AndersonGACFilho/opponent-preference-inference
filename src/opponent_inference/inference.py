"""Omega-hat inference: Bayesian filtering of the opponent's preference vector.

The opponent is one of the exact softmax policies of the Q tables, indexed by k on the omega
grid: at every move it samples a column from `softmax(Q_k / tau)`. The agent does not see k.
`OpponentFilter` keeps a posterior over the hypotheses in `hypothesis_ids` and updates it after
each opponent move:

    P(k | h_t) proportional to P(k) * prod_s pi_k(a_s | board_s)

computed in log space. The filter works over the 15 raw grid indices, not over behaviour
groups: at tau > 0 two omegas in one group can have different move probabilities (up to 0.39
on the 4x4 tables), so their likelihoods differ.

`BeliefWrapper` runs the filter inside the agent's loop and adds the posterior to every
observation as `obs["belief"]`. It sits on the agent side: the environment never computes the
belief (it only reports the opponent's last move and, optionally, the true omega for training).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .env import ConnectFourOpponentEnv, flatten_obs
from .solver import Board, QTables, empty_board, legal_moves, play

LIKELIHOOD_FLOOR = 1e-12  # keeps log-likelihoods finite for a move the model gives zero mass


class OpponentFilter:
    """Posterior over opponent grid indices, updated move by move.

    Args:
        tables: Q tables defining the opponent policies.
        hypothesis_ids: grid indices the filter may assign mass to (uniform prior over them).
            Indices outside this set keep posterior zero forever.
        tau: softmax temperature the filter assumes for the opponent. It must match the
            opponent's real tau (TD-02: still a placeholder).
    """

    def __init__(self, tables: QTables, hypothesis_ids: Sequence[int], tau: float):
        ids = sorted({int(k) for k in hypothesis_ids})
        n = len(tables.weights)
        if not ids:
            raise ValueError("hypothesis_ids cannot be empty")
        if ids[0] < 0 or ids[-1] >= n:
            raise ValueError(f"hypothesis_ids must be grid indices in [0, {n - 1}]")
        self.tables = tables
        self.tau = float(tau)
        self.hypothesis_ids = np.asarray(ids, dtype=np.int64)
        self._log_prior = np.full(n, -np.inf)
        self._log_prior[self.hypothesis_ids] = -np.log(len(ids))
        self._log_post = self._log_prior.copy()

    def reset(self) -> None:
        """Back to the prior: call at the start of every episode (no information carries over)."""
        self._log_post = self._log_prior.copy()

    @property
    def posterior(self) -> np.ndarray:
        """Posterior over all grid indices, shape (K,), summing to one."""
        p = np.exp(self._log_post - self._log_post.max())
        return p / p.sum()

    def group_posterior(self, groups: Sequence[Sequence[int]]) -> np.ndarray:
        """Posterior mass of each behaviour group (for reporting, not for the policy input)."""
        p = self.posterior
        return np.array([p[list(g)].sum() for g in groups])

    def opponent_probs(self, board: Board) -> np.ndarray:
        """Move distribution of every grid opponent at `board`, shape (K, W).

        Matches `QTables.action_probs` row by row, including the tau = 0 tie rule.
        """
        qv = self.tables.q[:, self.tables.index[board], :].astype(np.float64)
        best = qv.max(axis=1, keepdims=True)
        if self.tau <= 0:
            probs = (qv == best).astype(np.float64)
        else:
            probs = np.exp((qv - best) / self.tau)  # -inf (full column) becomes zero
        return probs / probs.sum(axis=1, keepdims=True)

    def update(self, board_before: Board, column: int) -> None:
        """Condition on one opponent move: it was played in `column` from `board_before`."""
        likelihood = self.opponent_probs(board_before)[:, column]
        self._log_post += np.log(np.maximum(likelihood, LIKELIHOOD_FLOOR))
        self._log_post -= self._log_post.max()  # keep values bounded; posterior is unchanged


class BeliefWrapper(gym.Wrapper):
    """Adds `obs["belief"]` (the filter's posterior over grid indices) to every observation.

    The wrapped environment must be a `ConnectFourOpponentEnv` (possibly under other wrappers).
    The filter's board before the opponent move is recovered from the agent's own move, so the
    environment needs no change.
    """

    def __init__(
        self,
        env: gym.Env,
        tables: QTables,
        hypothesis_ids: Sequence[int],
        tau: Optional[float] = None,
    ):
        super().__init__(env)
        base: ConnectFourOpponentEnv = env.unwrapped
        self.tables = tables
        self.filter = OpponentFilter(
            tables, hypothesis_ids, base.tau if tau is None else tau
        )
        n = len(tables.weights)
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "belief": spaces.Box(0.0, 1.0, (n,), dtype=np.float32),
        })

    @property
    def omega(self) -> np.ndarray:
        """The agent's own omega of the current episode (read by `train_ppo`)."""
        return self.env.unwrapped.omega

    def _with_belief(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        obs = dict(obs)
        obs["belief"] = self.filter.posterior.astype(np.float32)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.filter.reset()
        if info["opponent_move"] is not None:  # the opponent moved first, from the empty board
            base = self.env.unwrapped
            self.filter.update(empty_board(base.width), info["opponent_move"])
        return self._with_belief(obs), info

    def step(self, action):
        base = self.env.unwrapped
        action = int(action)
        board_before = None
        if action in legal_moves(base.board, base.height):
            board_before = play(base.board, action, base.agent_piece)  # opponent moves from here
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info["opponent_move"] is not None:
            self.filter.update(board_before, info["opponent_move"])
        return self._with_belief(obs), reward, terminated, truncated, info


def belief_obs_fn(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Policy input of the omega-hat agent: board, own omega, then the posterior over grid indices.

    Works for a single observation or a batch, like `flatten_obs`.
    """
    belief = np.asarray(obs["belief"], dtype=np.float32)
    return np.concatenate([flatten_obs(obs), belief], axis=-1)
