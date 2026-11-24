from pathlib import Path

import pytest

from opponent_inference.solver import load_q_tables
from opponent_inference.splits import behaviour_groups, train_test_split

TABLES = Path(__file__).resolve().parents[1] / "data" / "q_tables_4x4.npz"


@pytest.fixture(scope="module")
def tables():
    return load_q_tables(TABLES)


def test_grid_has_12_distinct_behaviours(tables):
    groups = behaviour_groups(tables)
    assert len(groups) == 12
    # Known coincidences of the 11/18 analysis (indices on the simplex grid):
    # (.75,.25,0) = (.5,.5,0) = (.25,.75,0) and (.75,0,.25) = (.5,0,.5).
    assert [1, 3, 6] in groups
    assert [2, 5] in groups


def test_groups_partition_the_grid(tables):
    flat = sorted(k for g in behaviour_groups(tables) for k in g)
    assert flat == list(range(len(tables.weights)))


def test_split_covers_grid_and_keeps_groups_together(tables):
    train, test = train_test_split(tables)
    assert sorted(train + test) == list(range(len(tables.weights)))
    assert not set(train) & set(test)
    for group in behaviour_groups(tables):
        assert set(group) <= set(train) or set(group) <= set(test)


def test_split_is_deterministic_and_depends_on_seed(tables):
    assert train_test_split(tables, seed=3) == train_test_split(tables, seed=3)
    baseline = train_test_split(tables, seed=0)
    assert any(train_test_split(tables, seed=s) != baseline for s in range(1, 6))


def test_split_rejects_invalid_sizes(tables):
    with pytest.raises(ValueError):
        train_test_split(tables, n_test_groups=0)
    with pytest.raises(ValueError):
        train_test_split(tables, n_test_groups=12)
