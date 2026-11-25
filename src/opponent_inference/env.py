"""Gymnasium environment: an agent plays MO-Connect4 against a fixed softmax opponent.

The opponent is one of the exact policies from the Q tables (`solver.QTables`): at every
move it samples a column from `softmax(Q_k / tau)`, where k is the index of its preference
vector omega on the grid. The environment returns a *vector* reward (3 objectives: win,
win fast, column control) from the agent's point of view; it is zero until the game ends,
because column control is only computed at the end, as in MOMAland. Use
`LinearScalarization` to get the scalar reward `omega . R` for single-objective learners.

What the environment gives to the agent:
    - observation: board from the agent's point of view, the agent's own omega and the
      mask of legal columns;
    - info (every step): the opponent's last move (`opponent_move`, None if it did not move);
    - the true opponent omega ONLY when `expose_opponent_omega=True` (oracle input and
      training label of the omega-hat inference module, never used at evaluation).
The environment never computes or provides omega-hat.

Board encoding: `board` has shape (2, H, W); channel 0 are the agent's pieces, channel 1
the opponent's; row 0 is the bottom of the column (same order as the solver's strings).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .solver import (
    N_OBJECTIVES,
    Board,
    QTables,
    as_weights,
    empty_board,
    legal_moves,
    play,
    player_to_move,
    terminal_outcome,
)

AGENT_SIDES = ("X", "O", "random")


def encode_board(board: Board, height: int, agent_piece: str) -> np.ndarray:
    """(2, H, W) float32 planes: agent's pieces, opponent's pieces (row 0 = bottom)."""
    width = len(board)
    planes = np.zeros((2, height, width), dtype=np.float32)
    for c, col in enumerate(board):
        for r, piece in enumerate(col):
            planes[0 if piece == agent_piece else 1, r, c] = 1.0
    return planes


def flat_obs_dim(height: int, width: int, expose_opponent_omega: bool = False) -> int:
    """Length of the vector returned by `flatten_obs`."""
    return 2 * height * width + N_OBJECTIVES * (2 if expose_opponent_omega else 1)


def flatten_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate board, own omega and (if present) true opponent omega into one vector.

    Works for a single observation or for a batch (leading dimensions are preserved). The
    action mask is not included: it is passed separately to the policy.
    """
    board = np.asarray(obs["board"], dtype=np.float32)
    omega = np.asarray(obs["omega"], dtype=np.float32)
    lead = omega.shape[:-1]
    parts = [board.reshape(*lead, -1), omega]
    if "opponent_omega" in obs:
        parts.append(np.asarray(obs["opponent_omega"], dtype=np.float32))
    return np.concatenate(parts, axis=-1)


class ConnectFourOpponentEnv(gym.Env):
    """MO-Connect4 against one opponent sampled per episode from `opponent_ids`.

    Args:
        tables: loaded Q tables (`solver.load_q_tables`); they define board size and omegas.
        opponent_ids: indices k (rows of `tables.weights`) from which the opponent of each
            episode is sampled uniformly.
        own_omega: the agent's preference vector (sums to 1). If None, it is sampled
            uniformly per episode from the omega grid of the tables.
        tau: opponent softmax temperature (0 = optimal move, ties broken uniformly).
        agent_side: "X" (moves first), "O" (the opponent moves first) or "random" per episode.
        expose_opponent_omega: add the true opponent omega to the observation and `info`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        tables: QTables,
        opponent_ids: Sequence[int],
        own_omega: Optional[Sequence[float]] = None,
        tau: float = 0.1,
        agent_side: str = "random",
        expose_opponent_omega: bool = False,
    ):
        super().__init__()
        if agent_side not in AGENT_SIDES:
            raise ValueError(f"agent_side must be one of {AGENT_SIDES}, got {agent_side!r}")
        if len(opponent_ids) == 0:
            raise ValueError("opponent_ids cannot be empty")
        self.tables = tables
        self.width, self.height = tables.width, tables.height
        self.opponent_ids = np.asarray(opponent_ids, dtype=np.int64)
        self._own_omega = None if own_omega is None else np.asarray(as_weights(own_omega))
        self.tau = float(tau)
        self.agent_side = agent_side
        self.expose_opponent_omega = expose_opponent_omega

        obs_spaces = {
            "board": spaces.Box(0.0, 1.0, (2, self.height, self.width), dtype=np.float32),
            "omega": spaces.Box(0.0, 1.0, (N_OBJECTIVES,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, (self.width,), dtype=np.int8),
        }
        if expose_opponent_omega:
            obs_spaces["opponent_omega"] = spaces.Box(0.0, 1.0, (N_OBJECTIVES,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Discrete(self.width)
        self.reward_space = spaces.Box(-1.0, 1.0, (N_OBJECTIVES,), dtype=np.float32)

        self._board: Board = empty_board(self.width)
        self._agent_piece = "X"
        self._opponent_index = int(self.opponent_ids[0])
        self._omega = np.zeros(N_OBJECTIVES)

    # ------------------------------------------------------------------ properties
    @property
    def omega(self) -> np.ndarray:
        """The agent's own preference vector in the current episode."""
        return self._omega

    @property
    def opponent_index(self) -> int:
        """Grid index of the current opponent's omega (ground truth, for analysis only)."""
        return self._opponent_index

    @property
    def board(self) -> Board:
        return self._board

    @property
    def agent_piece(self) -> str:
        return self._agent_piece

    # ------------------------------------------------------------------ helpers
    def _observation(self) -> Dict[str, np.ndarray]:
        mask = np.zeros(self.width, dtype=np.int8)
        mask[legal_moves(self._board, self.height)] = 1
        obs = {
            "board": encode_board(self._board, self.height, self._agent_piece),
            "omega": self._omega.astype(np.float32),
            "action_mask": mask,
        }
        if self.expose_opponent_omega:
            obs["opponent_omega"] = self.tables.weights[self._opponent_index].astype(np.float32)
        return obs

    def _info(self, opponent_move: Optional[int]) -> Dict[str, Any]:
        info: Dict[str, Any] = {"opponent_move": opponent_move, "agent_piece": self._agent_piece}
        if self.expose_opponent_omega:
            info["opponent_index"] = self._opponent_index
            info["opponent_omega"] = self.tables.weights[self._opponent_index].copy()
        return info

    def _opponent_plays(self) -> int:
        """Sample and apply the opponent's move; return the chosen column."""
        probs = self.tables.action_probs(self._board, self._opponent_index, self.tau)
        column = int(self.np_random.choice(self.width, p=probs))
        self._board = play(self._board, column, player_to_move(self._board))
        return column

    def _agent_reward(self, outcome_x: Sequence[float]) -> np.ndarray:
        """Outcome vector (from X's point of view) -> reward from the agent's point of view."""
        sign = 1.0 if self._agent_piece == "X" else -1.0
        return (sign * np.asarray(outcome_x)).astype(np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._opponent_index = int(self.np_random.choice(self.opponent_ids))
        if self._own_omega is None:
            row = self.np_random.integers(len(self.tables.weights))
            self._omega = np.asarray(as_weights(self.tables.weights[row]))
        else:
            self._omega = self._own_omega.copy()
        if self.agent_side == "random":
            self._agent_piece = "X" if self.np_random.integers(2) == 0 else "O"
        else:
            self._agent_piece = self.agent_side
        self._board = empty_board(self.width)
        opponent_move = None
        if self._agent_piece == "O":  # X (the opponent) moves first
            opponent_move = self._opponent_plays()
        return self._observation(), self._info(opponent_move)

    def step(self, action: int):
        action = int(action)
        if action not in legal_moves(self._board, self.height):
            raise ValueError(f"illegal move: column {action} on board {self._board}")

        self._board = play(self._board, action, self._agent_piece)
        outcome = terminal_outcome(self._board, self.height, self._agent_piece)
        opponent_move = None
        if outcome is None:
            opponent_move = self._opponent_plays()
            opponent_piece = "O" if self._agent_piece == "X" else "X"
            outcome = terminal_outcome(self._board, self.height, opponent_piece)

        terminated = outcome is not None
        reward = (
            self._agent_reward(outcome) if terminated else np.zeros(N_OBJECTIVES, np.float32)
        )
        info = self._info(opponent_move)
        if terminated:
            info["utility"] = float(np.dot(self._omega, reward))
        return self._observation(), reward, terminated, False, info


class LinearScalarization(gym.Wrapper):
    """Scalar reward `omega . R` using the agent's own omega of the current episode."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        scalar = float(np.dot(self.env.unwrapped.omega, reward))
        return obs, scalar, terminated, truncated, info
