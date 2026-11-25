from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opponent_inference.evaluate import evaluate_policy, random_policy, summarize
from opponent_inference.solver import load_q_tables

TABLES = Path(__file__).resolve().parents[1] / "data" / "q_tables_4x4.npz"


@pytest.fixture(scope="module")
def tables():
    return load_q_tables(TABLES)


def test_evaluate_policy_cells_and_rates(tables):
    df = evaluate_policy(random_policy(0), tables, [0, 5], [0, 14], n_games=10, tau=0.5)
    assert len(df) == 2 * 2 * 2  # opponents x own omegas x sides
    assert set(df["side"]) == {"X", "O"}
    assert (df[["win_rate", "draw_rate", "loss_rate"]].sum(axis=1) - 1).abs().max() < 1e-12
    assert df["mean_utility"].between(-1, 1).all()


def test_evaluate_policy_is_deterministic_given_seeds(tables):
    def run():
        return evaluate_policy(random_policy(3), tables, [2], [7], n_games=20, tau=0.3, seed=4)

    pd.testing.assert_frame_equal(run(), run())


def test_utility_matches_omega_dot_outcome_for_pure_win_omega(tables):
    # With omega = (1, 0, 0) the utility is win - loss, so it must equal win_rate - loss_rate.
    j = tables.weights_index((1, 0, 0))
    df = evaluate_policy(random_policy(0), tables, [3], [j], n_games=50, tau=0.5)
    expected = (df["win_rate"] - df["loss_rate"]).to_numpy()
    assert df["mean_utility"].to_numpy() == pytest.approx(expected)


def test_summarize_averages_over_seeds_with_ci():
    rows = []
    for seed, utility in enumerate([0.0, 0.2, 0.4]):
        rows.append({"seed": seed, "opponent_id": 1, "mean_utility": utility, "win_rate": 0.5,
                     "draw_rate": 0.0, "loss_rate": 0.5})
    out = summarize(pd.DataFrame(rows), by=["opponent_id"])
    assert out["mean_utility_mean"].iloc[0] == pytest.approx(0.2)
    # t(0.975, df=2) = 4.3027; std = 0.2; half-width = 4.3027 * 0.2 / sqrt(3)
    assert out["mean_utility_ci95"].iloc[0] == pytest.approx(4.3027 * 0.2 / np.sqrt(3), rel=1e-3)
    assert out["n_seeds"].iloc[0] == 3


def test_summarize_without_grouping_columns():
    df = pd.DataFrame({"seed": [0, 1], "mean_utility": [0.1, 0.3], "win_rate": [0.5, 0.5],
                       "draw_rate": [0.0, 0.0], "loss_rate": [0.5, 0.5]})
    out = summarize(df)
    assert out["mean_utility_mean"].iloc[0] == pytest.approx(0.2)
    assert out["n_seeds"].iloc[0] == 2
