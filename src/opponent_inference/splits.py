"""Train/test split of the opponent omegas.

Different omegas on the grid can produce exactly the same behaviour (the same optimal moves
in every position): the 15 omegas of the 4x4 grid give only 12 distinct behaviours. If two
omegas that play identically ended up on different sides of the split, the "unseen" test
opponent would not really be a new behaviour. So the split is done over whole *behaviour
groups*: all omegas of a group go to train, or all go to test.

This is a proposal that must be confirmed with the team (see TECH_DEBT.md, TD-03).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .solver import QTables

DEFAULT_SEED = 0
DEFAULT_N_TEST_GROUPS = 3


def behaviour_groups(tables: QTables, tol: float = 1e-6) -> List[List[int]]:
    """Group omega indices by identical behaviour at tau = 0.

    Two omegas have the same behaviour when, in every position, the set of optimal columns
    (those within `tol` of the best Q-value) is the same. Groups are ordered by their
    smallest index, and indices inside a group are ascending.
    """
    q = tables.q
    optimal = q >= q.max(axis=-1, keepdims=True) - tol   # (K, N, W); -inf columns are False
    groups: Dict[bytes, List[int]] = {}
    for k in range(q.shape[0]):
        groups.setdefault(optimal[k].tobytes(), []).append(k)
    return sorted(groups.values(), key=lambda g: g[0])


def train_test_split(
    tables: QTables,
    n_test_groups: int = DEFAULT_N_TEST_GROUPS,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[int], List[int]]:
    """Split omega indices into (train, test), keeping behaviour groups together.

    `n_test_groups` whole groups, chosen with a seeded RNG, form the test set; every other
    group is the train set. Both lists are sorted and together cover the whole grid.
    """
    groups = behaviour_groups(tables)
    if not 0 < n_test_groups < len(groups):
        raise ValueError(f"n_test_groups must be in [1, {len(groups) - 1}], got {n_test_groups}")
    rng = np.random.default_rng(seed)
    test_group_ids = set(rng.choice(len(groups), size=n_test_groups, replace=False).tolist())
    train = sorted(k for i, g in enumerate(groups) if i not in test_group_ids for k in g)
    test = sorted(k for i, g in enumerate(groups) if i in test_group_ids for k in g)
    return train, test
