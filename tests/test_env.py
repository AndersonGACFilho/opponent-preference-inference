from pathlib import Path

import numpy as np
import pytest

from opponent_inference.env import (
    ConnectFourOpponentEnv,
    LinearScalarization,
    flat_obs_dim,
    flatten_obs,
)
from opponent_inference.solver import (
    empty_board,
    legal_moves,
    load_q_tables,
    play,
    terminal_outcome,
)

TABLES = Path(__file__).resolve().parents[1] / "data" / "q_tables_4x4.npz"


@pytest.fixture(scope="module")
def tables():
    return load_q_tables(TABLES)


def _random_episode(env, rng, seed):
    """Play one episode with random legal moves; return (final reward, info, board replay)."""
    obs, info = env.reset(seed=seed)
    piece = info["agent_piece"]
    opp_piece = "O" if piece == "X" else "X"
    board = empty_board(env.width)
    if info["opponent_move"] is not None:
        board = play(board, info["opponent_move"], opp_piece)
    while True:
        legal = legal_moves(board, env.height)
        assert list(np.flatnonzero(obs["action_mask"])) == legal
        action = int(rng.choice(legal))
        obs, reward, terminated, truncated, info = env.step(action)
        assert not truncated
        board = play(board, action, piece)
        if info["opponent_move"] is not None:
            board = play(board, info["opponent_move"], opp_piece)
        assert board == env.board
        if terminated:
            return reward, info, board


@pytest.mark.parametrize("side", ["X", "O"])
def test_random_agent_finishes_games_against_all_opponents(tables, side):
    rng = np.random.default_rng(0)
    for k in range(len(tables.weights)):
        env = ConnectFourOpponentEnv(tables, [k], own_omega=(1, 0, 0), agent_side=side)
        for episode in range(3):
            reward, info, _ = _random_episode(env, rng, seed=episode)
            assert reward.shape == (3,)
            assert info["utility"] == pytest.approx(reward[0])  # omega = (1, 0, 0)


def test_terminal_reward_matches_solver_outcome(tables):
    rng = np.random.default_rng(1)
    for side in ("X", "O"):
        env = ConnectFourOpponentEnv(tables, range(15), agent_side=side)
        for episode in range(30):
            reward, info, board = _random_episode(env, rng, seed=episode)
            sign = 1.0 if side == "X" else -1.0
            # The last piece played is X's or O's depending on the number of pieces.
            last = "X" if sum(map(len, board)) % 2 == 1 else "O"
            expected = sign * np.asarray(terminal_outcome(board, env.height, last))
            assert reward == pytest.approx(expected, abs=1e-6)


def test_reward_is_zero_before_the_end(tables):
    env = ConnectFourOpponentEnv(tables, [0], own_omega=(0.2, 0, 0.8), agent_side="X")
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(0)
    assert not terminated
    assert not reward.any()


def test_opponent_moves_first_when_agent_is_o(tables):
    env = ConnectFourOpponentEnv(tables, [3], own_omega=(0.5, 0.5, 0), agent_side="O")
    obs, info = env.reset(seed=0)
    assert info["opponent_move"] is not None
    assert sum(map(len, env.board)) == 1
    assert obs["board"][1].sum() == 1  # one opponent piece, none of the agent's


def test_illegal_move_raises(tables):
    env = ConnectFourOpponentEnv(tables, [0], own_omega=(1, 0, 0), agent_side="X")
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(99)


def test_seed_reproducibility(tables):
    def trajectory(seed):
        env = ConnectFourOpponentEnv(tables, range(15), tau=0.5)
        rng = np.random.default_rng(123)
        _, info = env.reset(seed=seed)
        moves = [info["opponent_move"]]
        done = False
        while not done:
            action = int(rng.choice(legal_moves(env.board, env.height)))
            _, _, done, _, info = env.step(action)
            moves.append(info["opponent_move"])
        return moves, env.opponent_index

    assert trajectory(7) == trajectory(7)


def test_opponent_omega_is_hidden_unless_exposed(tables):
    hidden = ConnectFourOpponentEnv(tables, [5], own_omega=(1, 0, 0))
    obs, info = hidden.reset(seed=0)
    assert "opponent_omega" not in obs and "opponent_omega" not in info

    exposed = ConnectFourOpponentEnv(
        tables, [5], own_omega=(1, 0, 0), expose_opponent_omega=True
    )
    obs, info = exposed.reset(seed=0)
    assert obs["opponent_omega"] == pytest.approx(tables.weights[5])
    assert info["opponent_index"] == 5
    assert obs["opponent_omega"] in exposed.observation_space["opponent_omega"]


def test_flatten_obs_dimension_and_batching(tables):
    for expose in (False, True):
        env = ConnectFourOpponentEnv(tables, [0], own_omega=(1, 0, 0), expose_opponent_omega=expose)
        obs, _ = env.reset(seed=0)
        flat = flatten_obs(obs)
        assert flat.shape == (flat_obs_dim(env.height, env.width, expose),)
        batch = {k: np.stack([v, v]) for k, v in obs.items()}
        assert flatten_obs(batch).shape == (2, flat.shape[0])


def test_linear_scalarization_uses_own_omega(tables):
    env = LinearScalarization(
        ConnectFourOpponentEnv(tables, [0], own_omega=(0.2, 0.3, 0.5), agent_side="X")
    )
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    done = False
    while not done:
        action = int(rng.choice(legal_moves(env.unwrapped.board, env.unwrapped.height)))
        _, reward, done, _, info = env.step(action)
        assert isinstance(reward, float)
    assert reward == pytest.approx(info["utility"])
