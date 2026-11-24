"""Solver exato (indução reversa) para o MO-Connect4 do MOMAland.

Para um par fixo de vetores de preferência (w_X, w_O), calcula o equilíbrio perfeito
em subjogos: em cada posição, quem joga escolhe o lance que maximiza a própria
utilidade linear w · R, sabendo como o adversário responderá depois.

Representação
-------------
- Tabuleiro: tupla com uma string por coluna, de baixo para cima. Ex.: ('XO', '', 'X', '').
- X joga primeiro (player_0 no MOMAland); O é player_1.
- Vetor de resultado (sempre do ponto de vista de X), 3 objetivos:
    (vitória, vencer rápido, controle de colunas)
  O vetor de O é o negativo, porque todos os objetivos do MO-Connect4 são de soma zero.
- "Controle de colunas" é a média das recompensas por coluna do MOMAland (+1, 0, −1).

Desempate: quando dois lances dão a mesma utilidade para quem joga, vence a coluna de
menor índice. Isso define UM equilíbrio; no caso de soma geral (w_X ≠ w_O) ele pode
não ser o único. Declarar essa regra na metodologia.

Uso
---
    from opponent_inference.solver import solve_spe
    resultado = solve_spe(w_x=(1, 0, 0), w_o=(0.2, 0, 0.8), width=4, height=4)
    resultado.root_outcome          # vetor (vitória, rapidez, colunas) para X, na raiz
    resultado.values[tabuleiro]     # vetor de resultado de qualquer posição não terminal
    resultado.best_move[tabuleiro]  # coluna escolhida por quem joga naquela posição

Tabelas dos oponentes (decisão de 18/11: o oponente supõe que o agente tem o mesmo ω):
    from opponent_inference.solver import build_q_tables, load_q_tables
    build_q_tables("data/q_tables_4x4.npz")     # 15 ω da grade (passo 0,25), ~2 min
    tabelas = load_q_tables("data/q_tables_4x4.npz")
    tabelas.q_values(tabuleiro, k)               # utilidade de cada coluna para quem joga
    tabelas.action_probs(tabuleiro, k, tau=0.1)  # política softmax do oponente k
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
Outcome = Tuple[float, float, float]   # (vitória, vencer rápido, colunas), do ponto de vista de X
Weights = Tuple[float, float, float]   # vetor de preferências ω sobre os mesmos 3 objetivos

N_OBJECTIVES = 3
_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


# ----------------------------------------------------------------------------- preferências
def as_weights(w: Sequence[float], tol: float = 1e-9) -> Weights:
    """Converte qualquer sequência (lista, tupla, array do NumPy) para `Weights`.

    Garante 3 componentes `float` nativos, não negativos e somando 1 (ω no simplex).
    A soma 1 importa porque a temperatura do oponente softmax depende da escala da utilidade.
    """
    values = tuple(float(v) for v in w)
    if len(values) != N_OBJECTIVES:
        raise ValueError(
            f"ω precisa de {N_OBJECTIVES} componentes, recebeu {len(values)}: {values}"
        )
    if any(v < -tol for v in values):
        raise ValueError(f"ω não pode ter componentes negativos: {values}")
    if abs(sum(values) - 1.0) > 1e-6:
        raise ValueError(f"ω precisa somar 1 (simplex), soma = {sum(values):.6f}: {values}")
    return values  # type: ignore[return-value]


# ----------------------------------------------------------------------------- regras
def empty_board(width: int) -> Board:
    """Tabuleiro vazio com `width` colunas."""
    return tuple([""] * width)


def player_to_move(board: Board) -> str:
    """'X' se o número de peças é par, senão 'O'."""
    return "X" if sum(map(len, board)) % 2 == 0 else "O"


def legal_moves(board: Board, height: int) -> list[int]:
    """Colunas que ainda não estão cheias."""
    return [c for c, col in enumerate(board) if len(col) < height]


def play(board: Board, column: int, piece: str) -> Board:
    """Novo tabuleiro com `piece` jogada em `column` (não valida a jogada)."""
    new = list(board)
    new[column] = board[column] + piece
    return tuple(new)


def has_four(board: Board, piece: str) -> bool:
    """True se `piece` tem 4 em linha (horizontal, vertical ou diagonal)."""
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
    """Média do controle de colunas do ponto de vista de X: +1, 0 ou −1 por coluna."""
    total = 0
    for col in board:
        x, o = col.count("X"), col.count("O")
        total += (x > o) - (o > x)
    return total / len(board)


def terminal_outcome(board: Board, height: int, last_piece: str) -> Optional[Outcome]:
    """Vetor de resultado (para X) se a partida acabou depois do lance de `last_piece`.

    Devolve None se a partida continua. Reproduz as regras de recompensa do MOMAland:
    vitória ±1; rapidez ±(1 − lances/casas) para o vencedor/perdedor; colunas no fim
    (vitória ou empate).
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
    """Utilidade linear de `piece` para um vetor de resultado dado do ponto de vista de X."""
    sign = 1 if piece == "X" else -1
    return sign * sum(w * v for w, v in zip(weights, outcome_x))


# ----------------------------------------------------------------------------- solver
@dataclass
class SolveResult:
    """Resultado de `solve_spe` para um par (w_X, w_O)."""

    w_x: Weights
    w_o: Weights
    width: int
    height: int
    root_outcome: Outcome           # resultado da partida a partir do tabuleiro vazio
    values: Dict[Board, Outcome]    # posição não terminal -> vetor de resultado (para X)
    best_move: Dict[Board, int]     # posição não terminal -> coluna escolhida por quem joga

    @property
    def n_states(self) -> int:
        """Número de posições não terminais avaliadas."""
        return len(self.values)


def solve_spe(
    w_x: Sequence[float],
    w_o: Sequence[float],
    width: int = 4,
    height: int = 4,
) -> SolveResult:
    """Resolve o MO-Connect4 `width`×`height` por indução reversa para o par (w_x, w_o).

    Cada jogador maximiza a própria utilidade linear. Com w_x == w_o o jogo escalarizado
    é de soma zero e o resultado coincide com o minimax.

    `w_x` e `w_o` podem ser qualquer sequência (lista, tupla, array do NumPy); são
    convertidos uma única vez, aqui, para `Weights` por `as_weights`.
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
            if best_u is None or u > best_u + 1e-12:  # desempate: menor índice de coluna
                best_u, best_outcome, best_col = u, outcome, col
        values[board] = best_outcome
        best_move[board] = best_col
        return best_outcome

    root = value(empty_board(width))
    return SolveResult(w_x, w_o, width, height, root, values, best_move)


# ----------------------------------------------------------------------------- tabelas Q
TIE_BREAK = "menor índice de coluna"


def simplex_grid(divisions: int = 4) -> List[Weights]:
    """Grade regular no simplex de 3 objetivos: todas as ω com componentes múltiplos de
    1/`divisions` que somam 1. `divisions=4` (passo 0,25) dá 15 pontos.

    Ordem fixa e reprodutível: (i, j, k) com i decrescente, depois j decrescente, onde
    ω = (i, j, k) / divisions = (vitória, rapidez, colunas).
    """
    grid: List[Weights] = []
    for i in range(divisions, -1, -1):
        for j in range(divisions - i, -1, -1):
            k = divisions - i - j
            grid.append(as_weights((i / divisions, j / divisions, k / divisions)))
    return grid


def encode_board(board: Board) -> str:
    """Chave em texto de uma posição, para guardar no .npz. Ex.: ('XO','','X','') -> 'XO||X|'."""
    return "|".join(board)


def decode_board(key: str) -> Board:
    """Inverso de `encode_board`."""
    return tuple(key.split("|"))


def q_values_from_solution(res: SolveResult, board: Board) -> np.ndarray:
    """Q-valores de `board` a partir de uma solução: utilidade, para quem joga, do resultado
    de cada coluna. Colunas cheias recebem −inf. Usa os pesos de quem joga na solução."""
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
    """Gera as tabelas Q dos oponentes e salva num único .npz comprimido.

    Para cada ω da grade resolve `solve_spe(ω, ω)` (o oponente supõe que o agente tem o
    mesmo ω que ele) e guarda, para toda posição não terminal alcançável, a utilidade de
    cada coluna para quem joga. A mesma tabela serve com o oponente jogando de X ou de O.

    Conteúdo do arquivo:
        boards   (N,)       chaves de texto das posições (ver `encode_board`)
        q        (K, N, W)  float32; q[k, i, c] = utilidade de jogar a coluna c na posição i
                            para quem joga, sob ω_k; −inf para coluna cheia
        weights  (K, 3)     a grade de ω, na ordem do índice k
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
            boards = sorted(res.values)  # o conjunto de posições não depende de ω
            index = {b: i for i, b in enumerate(boards)}
            q = np.full((len(grid), len(boards), width), -np.inf, dtype=np.float32)
        elif len(res.values) != len(boards) or any(b not in index for b in res.values):
            raise RuntimeError(f"conjunto de posições diferente para ω = {w}")
        for b, i in index.items():
            q[k, i] = q_values_from_solution(res, b)
        if verbose:
            print(f"[{k + 1:2d}/{len(grid)}] ω = {w}  raiz = "
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
        print(f"salvo em {path} ({path.stat().st_size / 1e6:.1f} MB): "
              f"{len(grid)} ω × {len(boards)} posições × {width} colunas")
    return path


@dataclass
class QTables:
    """Tabelas Q carregadas de um .npz gerado por `build_q_tables`."""

    q: np.ndarray                 # (K, N, W)
    weights: np.ndarray           # (K, 3)
    index: Dict[Board, int]       # posição -> linha
    width: int
    height: int
    tie_break: str

    def weights_index(self, w: Sequence[float], tol: float = 1e-9) -> int:
        """Índice k da grade correspondente a ω (erro se ω não estiver na grade)."""
        diffs = np.abs(self.weights - np.asarray(as_weights(w))).max(axis=1)
        k = int(diffs.argmin())
        if diffs[k] > tol:
            raise KeyError(f"ω = {tuple(w)} não está na grade")
        return k

    def q_values(self, board: Board, k: int) -> np.ndarray:
        """Utilidade de cada coluna para quem joga em `board`, sob ω_k (−inf = coluna cheia)."""
        return self.q[k, self.index[board]]

    def action_probs(self, board: Board, k: int, tau: float) -> np.ndarray:
        """Política softmax do oponente k: P(c) ∝ exp(Q(c)/τ).

        τ = 0 dá o lance ótimo; havendo empate, sorteia uniformemente entre os empatados
        (diferente do solver, que escolhe a menor coluna). Os empatados têm o mesmo valor
        para quem joga, então isso não muda a otimalidade, só deixa o oponente mais variado.
        """
        qv = self.q_values(board, k).astype(np.float64)
        if tau <= 0:
            probs = (qv == qv.max()).astype(np.float64)
        else:
            z = (qv - qv.max()) / tau  # −inf vira probabilidade zero
            probs = np.exp(z)
        return probs / probs.sum()


def load_q_tables(path: str | Path) -> QTables:
    """Carrega um .npz gerado por `build_q_tables`."""
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


# ----------------------------------------------------------------------------- demonstração
def _demo() -> None:
    """Reproduz o diagnóstico de 16/11: resultado na raiz para pares de preferências."""
    import itertools

    prefs = {
        "vitória": (1, 0, 0),
        "rápido": (0.5, 0.5, 0),
        "colunas": (0.2, 0, 0.8),
        "só colunas": (0, 0, 1),
    }
    for (name_x, wx), (name_o, wo) in itertools.product(prefs.items(), repeat=2):
        res = solve_spe(wx, wo)
        vec = tuple(round(v, 3) for v in res.root_outcome)
        print(f"X={name_x:10s} O={name_o:10s} -> (vitória, rápido, colunas) p/ X = {vec}"
              f"  [{res.n_states} estados]")


if __name__ == "__main__":
    _demo()
