"""Consistency of the solver's game rules with MOMAland's MO-Connect4 (4x4, column objectives).

Plays random games in both implementations with the same move sequence and checks, at every
step, that legal moves, the turn order and the termination agree, and that at the end the
reward vectors match: (win, win fast, mean of the per-column rewards) for X = player_0, and
the negative for O = player_1.

Needs the optional dependency group: `uv sync --group consistency`. Skipped otherwise.
"""
import random

import numpy as np
import pytest

from opponent_inference.solver import (
    empty_board,
    legal_moves,
    play,
    player_to_move,
    terminal_outcome,
)

raw_env = pytest.importorskip("momaland.envs.connect4.connect4").raw_env

WIDTH = HEIGHT = 4
N_GAMES = 2000


def _play_random_game(rng: random.Random) -> tuple[str | None, int]:
    """Play one random game in both implementations; return (winner or None, n_moves)."""
    ref = raw_env(board_width=WIDTH, board_height=HEIGHT, column_objectives=True)
    ref.reset()
    board = empty_board(WIDTH)
    n_moves = 0
    while True:
        moves = legal_moves(board, HEIGHT)
        assert ref._legal_moves() == moves
        piece = player_to_move(board)
        assert ref.agent_selection == ("player_0" if piece == "X" else "player_1")

        column = rng.choice(moves)
        ref.step(column)
        board = play(board, column, piece)
        n_moves += 1

        outcome = terminal_outcome(board, HEIGHT, piece)
        assert all(ref.terminations.values()) == (outcome is not None)

        r_x = np.asarray(ref.rewards["player_0"], dtype=float)
        r_o = np.asarray(ref.rewards["player_1"], dtype=float)
        if outcome is None:
            assert not r_x.any() and not r_o.any()
            continue

        ref_outcome = (r_x[0], r_x[1], r_x[2:].mean())
        assert ref_outcome == pytest.approx(outcome)
        assert r_o == pytest.approx(-r_x)
        winner = None if outcome[0] == 0 else ("X" if outcome[0] > 0 else "O")
        return winner, n_moves


def test_random_games_match_momaland():
    rng = random.Random(0)
    winners = [_play_random_game(rng)[0] for _ in range(N_GAMES)]
    # Both sides must win sometimes, otherwise the sign handling is not being exercised.
    assert "X" in winners and "O" in winners
