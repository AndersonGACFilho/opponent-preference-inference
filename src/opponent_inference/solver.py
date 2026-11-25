"""Exact solver (backward induction) for the MOMAland MO-Connect4.

For a fixed pair of preference vectors (w_X, w_O), computes the subgame-perfect
equilibrium: at every position, the player to move picks the move that maximizes their
own linear utility w · R, knowing how the opponent will respond afterwards.

Representation
--------------
- Board: tuple with one string per column, bottom to top. E.g. ('XO', '', 'X', '').
- X moves first (player_0 in MOMAland); O is player_1.
- Outcome vector (always from X's point of view), 3 objectives:
    (win, win fast, column control)
  O's vector is the negative, because all MO-Connect4 objectives are zero-sum.
- "Column control" is the average of MOMAland's per-column rewards (+1, 0, -1).

Tie-break: when two moves give the same utility to the player to move, the lowest column
index wins. This defines ONE equilibrium; in the general-sum case (w_X != w_O) it may not
be the only one. State this rule in the methodology.

Usage
-----
    from opponent_inference.solver import solve_spe
    result = solve_spe(w_x=(1, 0, 0), w_o=(0.2, 0, 0.8), width=4, height=4)
    result.root_outcome          # vector (win, speed, columns) for X, at the root
    result.values[board]         # outcome vector of any non-terminal position
    result.best_move[board]      # column chosen by the player to move at that position

Opponent tables (decision of 11/18: the opponent assumes the agent has the same omega):
    from opponent_inference.solver import build_q_tables, load_q_tables
    build_q_tables("data/q_tables_4x4.npz")     # 15 omegas on the grid (step 0.25), ~2 min
    tables = load_q_tables("data/q_tables_4x4.npz")
    tables.q_values(board, k)               # utility of each column for the player to move
    tables.action_probs(board, k, tau=0.1)  # softmax policy of opponent k
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Board = Tuple[str, ...]
Outcome = Tuple[float, float, float]   # (win, win fast, columns), from X's point of view
Weights = Tuple[float, float, float]   # preference vector omega over the same 3 objectives

N_OBJECTIVES = 3
_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


# ----------------------------------------------------------------------------- preferences
def as_weights(w: Sequence[float], tol: float = 1e-9) -> Weights:
    """Convert any sequence (list, tuple, NumPy array) to `Weights`.

    Guarantees 3 native `float` components, non-negative and summing to 1 (omega on the
    simplex). The sum of 1 matters because the softmax opponent's temperature depends on
    the utility scale.
    """
    values = tuple(float(v) for v in w)
    if len(values) != N_OBJECTIVES:
        raise ValueError(
            f"omega needs {N_OBJECTIVES} components, got {len(values)}: {values}"
        )
    if any(v < -tol for v in values):
        raise ValueError(f"omega cannot have negative components: {values}")
    if abs(sum(values) - 1.0) > 1e-6:
        raise ValueError(f"omega must sum to 1 (simplex), sum = {sum(values):.6f}: {values}")
    return values  # type: ignore[return-value]


# ----------------------------------------------------------------------------- rules
def empty_board(width: int) -> Board:
    """Empty board with `width` columns."""
    return tuple([""] * width)


def player_to_move(board: Board) -> str:
    """'X' if the number of pieces is even, otherwise 'O'."""
    return "X" if sum(map(len, board)) % 2 == 0 else "O"


def legal_moves(board: Board, height: int) -> list[int]:
    """Columns that are not yet full."""
    return [c for c, col in enumerate(board) if len(col) < height]


def play(board: Board, column: int, piece: str) -> Board:
    """New board with `piece` dropped in `column` (does not validate the move)."""
    new = list(board)
    new[column] = board[column] + piece
    return tuple(new)


def has_four(board: Board, piece: str) -> bool:
    """True if `piece` has 4 in a row (horizontal, vertical or diagonal)."""
    width = len(board)

    def cell(c: int, r: int) -> str:
        return board[c][r] if 0 <= c < width and 0 <= r < len(board[c]) else "."

    for c in range(width):
        for r in range(len(board[c])):
            if board[c][r] != piece:
                continue
            for dc, dr in _DIRECTIONS:
                if all(cell(c + dc * k, r + dr * k) == piece for k in range(1, 4)):
                    return True
    return False


def column_score(board: Board) -> float:
    """Average column control from X's point of view: +1, 0 or -1 per column."""
    total = 0
    for col in board:
        x, o = col.count("X"), col.count("O")
        total += (x > o) - (o > x)
    return total / len(board)


def terminal_outcome(board: Board, height: int, last_piece: str) -> Optional[Outcome]:
    """Outcome vector (for X) if the game ended after the move of `last_piece`.

    Returns None if the game continues. Reproduces MOMAland's reward rules: win +-1;
    speed +-(1 - moves/cells) for the winner/loser; columns at the end (win or draw).
    """
    width = len(board)
    n_moves = sum(map(len, board))
    n_cells = width * height
    if has_four(board, last_piece):
        sign = 1 if last_piece == "X" else -1
        return (float(sign), sign * (1 - n_moves / n_cells), column_score(board))
    if n_moves == n_cells:
        return (0.0, 0.0, column_score(board))
    return None


def utility(weights: Weights, outcome_x: Outcome, piece: str) -> float:
    """Linear utility of `piece` for an outcome vector given from X's point of view."""
    sign = 1 if piece == "X" else -1
    return sign * sum(w * v for w, v in zip(weights, outcome_x))


# ----------------------------------------------------------------------------- solver
@dataclass
class SolveResult:
    """Result of `solve_spe` for a pair (w_X, w_O)."""

    w_x: Weights
    w_o: Weights
    width: int
    height: int
    root_outcome: Outcome           # game outcome starting from the empty board
    values: Dict[Board, Outcome]    # non-terminal position -> outcome vector (for X)
    best_move: Dict[Board, int]     # non-terminal position -> column chosen by the mover

    @property
    def n_states(self) -> int:
        """Number of non-terminal positions evaluated."""
        return len(self.values)


def solve_spe(
    w_x: Sequence[float],
    w_o: Sequence[float],
    width: int = 4,
    height: int = 4,
) -> SolveResult:
    """Solve the `width`x`height` MO-Connect4 by backward induction for the pair (w_x, w_o).

    Each player maximizes their own linear utility. With w_x == w_o the scalarized game
    is zero-sum and the result matches minimax.

    `w_x` and `w_o` can be any sequence (list, tuple, NumPy array); they are converted
    exactly once, here, to `Weights` by `as_weights`.
    """
    w_x, w_o = as_weights(w_x), as_weights(w_o)
    values: Dict[Board, Outcome] = {}
    best_move: Dict[Board, int] = {}
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * width * height + 1000))

    @lru_cache(maxsize=None)
    def value(board: Board) -> Outcome:
        piece = player_to_move(board)
        weights = w_x if piece == "X" else w_o
        best_u, best_outcome, best_col = None, None, None
        for col in legal_moves(board, height):
            child = play(board, col, piece)
            outcome = terminal_outcome(child, height, piece)
            if outcome is None:
                outcome = value(child)
            u = utility(weights, outcome, piece)
            if best_u is None or u > best_u + 1e-12:  # tie-break: lowest column index
                best_u, best_outcome, best_col = u, outcome, col
        values[board] = best_outcome
        best_move[board] = best_col
        return best_outcome

    root = value(empty_board(width))
    return SolveResult(w_x, w_o, width, height, root, values, best_move)


# ----------------------------------------------------------------------------- Q tables
TIE_BREAK = "lowest column index"


def simplex_grid(divisions: int = 4) -> List[Weights]:
    """Regular grid on the 3-objective simplex: all omegas with components that are
    multiples of 1/`divisions` and sum to 1. `divisions=4` (step 0.25) gives 15 points.

    Fixed, reproducible order: (i, j, k) with i decreasing, then j decreasing, where
    omega = (i, j, k) / divisions = (win, speed, columns).
    """
    grid: List[Weights] = []
    for i in range(divisions, -1, -1):
        for j in range(divisions - i, -1, -1):
            k = divisions - i - j
            grid.append(as_weights((i / divisions, j / divisions, k / divisions)))
    return grid


def encode_board(board: Board) -> str:
    """Text key of a position, for storing in the .npz. E.g. ('XO','','X','') -> 'XO||X|'."""
    return "|".join(board)


def decode_board(key: str) -> Board:
    """Inverse of `encode_board`."""
    return tuple(key.split("|"))


def q_values_from_solution(res: SolveResult, board: Board) -> np.ndarray:
    """Q-values of `board` from a solution: utility, for the player to move, of the outcome
    of each column. Full columns get -inf. Uses the weights of the player to move in the
    solution."""
    piece = player_to_move(board)
    weights = res.w_x if piece == "X" else res.w_o
    q = np.full(res.width, -np.inf)
    for col in legal_moves(board, res.height):
        child = play(board, col, piece)
        outcome = terminal_outcome(child, res.height, piece)
        if outcome is None:
            outcome = res.values[child]
        q[col] = utility(weights, outcome, piece)
    return q


def build_q_tables(
    path: str | Path = "data/q_tables_4x4.npz",
    weights_grid: Optional[Sequence[Sequence[float]]] = None,
    width: int = 4,
    height: int = 4,
    verbose: bool = True,
) -> Path:
    """Generate the opponents' Q tables and save them in a single compressed .npz.

    For each omega on the grid, solves `solve_spe(omega, omega)` (the opponent assumes
    the agent has the same omega as itself) and stores, for every reachable non-terminal
    position, the utility of each column for the player to move. The same table serves
    the opponent playing as X or as O.

    File contents:
        boards   (N,)       text keys of the positions (see `encode_board`)
        q        (K, N, W)  float32; q[k, i, c] = utility of playing column c at position i
                            for the player to move, under omega_k; -inf for a full column
        weights  (K, 3)     the omega grid, in the order of index k
        width, height, tie_break
    """
    grid = [as_weights(w) for w in (weights_grid if weights_grid is not None else simplex_grid())]
    boards: Optional[List[Board]] = None
    index: Dict[Board, int] = {}
    q = None
    start = time.perf_counter()
    for k, w in enumerate(grid):
        res = solve_spe(w, w, width, height)
        if boards is None:
            boards = sorted(res.values)  # the set of positions does not depend on omega
            index = {b: i for i, b in enumerate(boards)}
            q = np.full((len(grid), len(boards), width), -np.inf, dtype=np.float32)
        elif len(res.values) != len(boards) or any(b not in index for b in res.values):
            raise RuntimeError(f"different set of positions for omega = {w}")
        for b, i in index.items():
            q[k, i] = q_values_from_solution(res, b)
        if verbose:
            print(f"[{k + 1:2d}/{len(grid)}] omega = {w}  root = "
                  f"{tuple(round(v, 3) for v in res.root_outcome)}  "
                  f"({time.perf_counter() - start:.0f} s)", flush=True)

    path = Path(path)
    np.savez_compressed(
        path,
        boards=np.array([encode_board(b) for b in boards]),
        q=q,
        weights=np.array(grid, dtype=np.float64),
        width=width,
        height=height,
        tie_break=TIE_BREAK,
    )
    if verbose:
        print(f"saved to {path} ({path.stat().st_size / 1e6:.1f} MB): "
              f"{len(grid)} omegas x {len(boards)} positions x {width} columns")
    return path


@dataclass
class QTables:
    """Q tables loaded from an .npz generated by `build_q_tables`."""

    q: np.ndarray                 # (K, N, W)
    weights: np.ndarray           # (K, 3)
    index: Dict[Board, int]       # position -> row
    width: int
    height: int
    tie_break: str

    def weights_index(self, w: Sequence[float], tol: float = 1e-9) -> int:
        """Grid index k matching omega (error if omega is not on the grid)."""
        diffs = np.abs(self.weights - np.asarray(as_weights(w))).max(axis=1)
        k = int(diffs.argmin())
        if diffs[k] > tol:
            raise KeyError(f"omega = {tuple(w)} is not on the grid")
        return k

    def q_values(self, board: Board, k: int) -> np.ndarray:
        """Utility of each column for the player to move at `board`, under omega_k
        (-inf = full column)."""
        return self.q[k, self.index[board]]

    def action_probs(self, board: Board, k: int, tau: float) -> np.ndarray:
        """Softmax policy of opponent k: P(c) proportional to exp(Q(c)/tau).

        tau = 0 gives the optimal move; with ties, samples uniformly among the tied moves
        (unlike the solver, which picks the lowest column). Tied moves have the same value
        for the player to move, so this does not change optimality, it only makes the
        opponent more varied.
        """
        qv = self.q_values(board, k).astype(np.float64)
        if tau <= 0:
            probs = (qv == qv.max()).astype(np.float64)
        else:
            z = (qv - qv.max()) / tau  # -inf becomes probability zero
            probs = np.exp(z)
        return probs / probs.sum()


def load_q_tables(path: str | Path) -> QTables:
    """Load an .npz generated by `build_q_tables`."""
    data = np.load(path, allow_pickle=False)
    boards = [decode_board(str(key)) for key in data["boards"]]
    return QTables(
        q=data["q"],
        weights=data["weights"],
        index={b: i for i, b in enumerate(boards)},
        width=int(data["width"]),
        height=int(data["height"]),
        tie_break=str(data["tie_break"]),
    )


# ----------------------------------------------------------------------------- demo
def _demo() -> None:
    """Reproduce the 11/16 diagnosis: root outcome for pairs of preferences."""
    import itertools

    prefs = {
        "win": (1, 0, 0),
        "fast": (0.5, 0.5, 0),
        "columns": (0.2, 0, 0.8),
        "columns only": (0, 0, 1),
    }
    for (name_x, wx), (name_o, wo) in itertools.product(prefs.items(), repeat=2):
        res = solve_spe(wx, wo)
        vec = tuple(round(v, 3) for v in res.root_outcome)
        print(f"X={name_x:12s} O={name_o:12s} -> (win, fast, columns) for X = {vec}"
              f"  [{res.n_states} states]")


if __name__ == "__main__":
    _demo()
