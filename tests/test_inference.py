from pathlib import Path

import numpy as np
import pytest

from opponent_inference.env import ConnectFourOpponentEnv
from opponent_inference.inference import BeliefWrapper, OpponentFilter, belief_obs_fn
from opponent_inference.solver import (
    empty_board,
    legal_moves,
    load_q_tables,
    play,
    terminal_outcome,
)
from opponent_inference.splits import behaviour_groups

TABLES = Path(__file__).resolve().parents[1] / "data" / "q_tables_4x4.npz"


@pytest.fixture(scope="module")
def tables():
    return load_q_tables(TABLES)


def _random_positions(tables, n_games, seed):
    """Non-terminal positions reached by random play; all of them are covered by the tables."""
    rng = np.random.default_rng(seed)
    boards = []
    for _ in range(n_games):
        board, piece = empty_board(tables.width), "X"
        while True:
            moves = legal_moves(board, tables.height)
            if not moves:
                break
            boards.append(board)
            board = play(board, int(rng.choice(moves)), piece)
            if terminal_outcome(board, tables.height, piece) is not None:
                break
            piece = "O" if piece == "X" else "X"
    return boards


@pytest.mark.parametrize("tau", [0.0, 0.1, 0.7])
def test_opponent_probs_match_qtables_action_probs(tables, tau):
    filt = OpponentFilter(tables, range(len(tables.weights)), tau)
    for board in _random_positions(tables, n_games=20, seed=1):
        assert board in tables.index
        probs = filt.opponent_probs(board)
        for k in range(len(tables.weights)):
            np.testing.assert_allclose(probs[k], tables.action_probs(board, k, tau), atol=1e-12)


def test_prior_is_uniform_over_hypotheses_and_zero_elsewhere(tables):
    hyp = [0, 4, 9]
    filt = OpponentFilter(tables, hyp, tau=0.1)
    p = filt.posterior
    assert p.sum() == pytest.approx(1.0)
    assert p[hyp] == pytest.approx(np.full(3, 1 / 3))
    assert p[[i for i in range(len(p)) if i not in hyp]] == pytest.approx(0.0)


def test_update_never_gives_mass_outside_hypotheses(tables):
    hyp = [0, 4, 9]
    filt = OpponentFilter(tables, hyp, tau=0.1)
    board = empty_board(tables.width)
    filt.update(board, 1)
    p = filt.posterior
    assert p.sum() == pytest.approx(1.0)
    assert p[[i for i in range(len(p)) if i not in hyp]] == pytest.approx(0.0)


def test_true_opponent_keeps_positive_mass(tables):
    """With tau = 0 the opponent plays only optimal moves; its own index is never excluded."""
    k_true = 9
    env = BeliefWrapper(
        ConnectFourOpponentEnv(tables, [k_true], tau=0.0, agent_side="X"),
        tables, range(len(tables.weights)), tau=0.0,
    )
    rng = np.random.default_rng(0)
    for episode in range(10):
        obs, _ = env.reset(seed=episode)
        done = False
        while not done:
            action = int(rng.choice(np.flatnonzero(obs["action_mask"])))
            obs, _, done, _, _ = env.step(action)
            assert obs["belief"][k_true] > 0.0
        assert obs["belief"].sum() == pytest.approx(1.0)


def test_reset_restores_prior(tables):
    hyp = list(range(len(tables.weights)))
    filt = OpponentFilter(tables, hyp, tau=0.1)
    # A position where the opponents disagree on the move (the empty board does not: every
    # opponent opens in the same way), so the update must move the posterior away from uniform.
    board, column = next(
        (b, c)
        for b in _random_positions(tables, n_games=20, seed=2)
        for c in legal_moves(b, tables.height)
        if np.ptp(filt.opponent_probs(b)[:, c]) > 1e-3
    )
    filt.update(board, column)
    assert not np.allclose(filt.posterior, np.full(len(hyp), 1 / len(hyp)))
    filt.reset()
    assert filt.posterior == pytest.approx(np.full(len(hyp), 1 / len(hyp)))


def test_group_posterior_sums_to_one(tables):
    filt = OpponentFilter(tables, range(len(tables.weights)), tau=0.1)
    gp = filt.group_posterior(behaviour_groups(tables))
    assert gp.sum() == pytest.approx(1.0)
    assert len(gp) == 12


def test_wrapper_adds_belief_and_passes_env_through(tables):
    hyp = range(len(tables.weights))
    base = ConnectFourOpponentEnv(tables, [5], tau=0.1, agent_side="X")
    wrapped = BeliefWrapper(
        ConnectFourOpponentEnv(tables, [5], tau=0.1, agent_side="X"), tables, hyp, tau=0.1
    )
    obs_a, info_a = base.reset(seed=3)
    obs_b, info_b = wrapped.reset(seed=3)
    assert "belief" in obs_b and obs_b["belief"].shape == (len(tables.weights),)
    np.testing.assert_array_equal(obs_a["board"], obs_b["board"])
    assert info_a["opponent_move"] == info_b["opponent_move"]
    done = False
    while not done:
        action = int(np.flatnonzero(obs_a["action_mask"])[0])
        obs_a, r_a, done, _, info_a = base.step(action)
        obs_b, r_b, done_b, _, info_b = wrapped.step(action)
        assert done == done_b
        np.testing.assert_array_equal(r_a, r_b)
        assert info_a["opponent_move"] == info_b["opponent_move"]


def test_belief_obs_fn_dimension(tables):
    wrapped = BeliefWrapper(
        ConnectFourOpponentEnv(tables, [5], tau=0.1), tables, range(len(tables.weights))
    )
    obs, _ = wrapped.reset(seed=0)
    x = belief_obs_fn(obs)
    # board (2 * 4 * 4) + own omega (3) + belief over the 15 grid indices
    assert x.shape == (2 * tables.height * tables.width + 3 + len(tables.weights),)
    assert np.all(np.isfinite(x))
