from pathlib import Path

import numpy as np
import pytest

from opponent_inference.solver import (
    as_weights,
    load_q_tables,
    simplex_grid,
    solve_spe,
)

TABLES = Path(__file__).resolve().parents[1] / "data" / "q_tables_4x4.npz"


def test_as_weights_rejects_outside_simplex():
    with pytest.raises(ValueError):
        as_weights((0.5, 0.5, 0.5))  # sums to 1.5
    with pytest.raises(ValueError):
        as_weights((-0.1, 0.6, 0.5))  # negative component


def test_simplex_grid_has_15_points_summing_to_one():
    grid = simplex_grid()
    assert len(grid) == 15
    assert all(abs(sum(w) - 1.0) < 1e-9 for w in grid)


def test_solve_spe_root_is_draw_for_win_and_fast_only():
    # 11/16 diagnosis: with only win and win-fast, the optimal outcome is a draw.
    res = solve_spe((0.5, 0.5, 0), (0.5, 0.5, 0))
    assert res.root_outcome[0] == 0.0
    assert res.n_states > 100_000


def test_q_tables_load_and_action_probs_are_distributions():
    tables = load_q_tables(TABLES)
    assert tables.q.shape[0] == 15
    board = tuple([""] * tables.width)
    for k in range(tables.q.shape[0]):
        probs = tables.action_probs(board, k, tau=0.1)
        assert np.isclose(probs.sum(), 1.0)
        assert np.all(probs[np.isneginf(tables.q_values(board, k))] == 0)
