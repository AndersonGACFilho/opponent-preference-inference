"""Phase 0 gate: can the filter identify a known opponent's behaviour group from its moves?

For every grid opponent k, plays games against a random agent (the agent's moves do not matter
for identification) and records the posterior mass on k's behaviour group after each opponent
move. The filter's hypothesis set is the full grid here: the question is identifiability, not
the train/test protocol (decision D1).

Gate (design doc): the mass on the true group should pass 0.9 within a fixed number of moves.
Writes per-opponent statistics to results/phase0_identification.csv.

Example:
    uv run python scripts/phase0_identification.py --games 200 --tau 0.1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from opponent_inference.env import ConnectFourOpponentEnv
from opponent_inference.inference import BeliefWrapper
from opponent_inference.solver import load_q_tables
from opponent_inference.splits import behaviour_groups

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.9
CHECKPOINTS = (1, 2, 4, 8)  # opponent moves at which the group mass is reported


def group_mass_curve(env: BeliefWrapper, group: list[int], rng: np.random.Generator) -> list[float]:
    """Group mass after each opponent move of one game (one value per opponent move)."""
    masses = []
    obs, info = env.reset(seed=int(rng.integers(2**31)))
    if info["opponent_move"] is not None:
        masses.append(float(obs["belief"][group].sum()))
    done = False
    while not done:
        action = int(rng.choice(np.flatnonzero(obs["action_mask"])))
        obs, _, done, _, info = env.step(action)
        if info["opponent_move"] is not None:
            masses.append(float(obs["belief"][group].sum()))
    return masses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=200, help="games per opponent")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--tables", type=Path, default=ROOT / "data" / "q_tables_4x4.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "phase0_identification.csv")
    args = parser.parse_args()

    tables = load_q_tables(args.tables)
    n = len(tables.weights)
    groups = behaviour_groups(tables)
    group_of = {k: g for g in groups for k in g}
    rng = np.random.default_rng(0)

    rows = []
    for k in range(n):
        group = group_of[k]
        env = BeliefWrapper(
            ConnectFourOpponentEnv(tables, [k], tau=args.tau, agent_side="random"),
            tables, range(n), args.tau,
        )
        curves = [group_mass_curve(env, group, rng) for _ in range(args.games)]
        reach = [next((i + 1 for i, m in enumerate(c) if m > THRESHOLD), None) for c in curves]
        reached = [r for r in reach if r is not None]
        row = {
            "opponent": k, "group": "/".join(map(str, group)), "games": args.games,
            "frac_pass_0.9": len(reached) / args.games,
            "median_moves_to_0.9": float(np.median(reached)) if reached else float("nan"),
        }
        for t in CHECKPOINTS:
            vals = [c[t - 1] for c in curves if len(c) >= t]
            row[f"mass_after_{t}"] = float(np.mean(vals)) if vals else float("nan")
        rows.append(row)
        print(
            f"k={k:2d} group={row['group']:>8}  pass>0.9: {row['frac_pass_0.9']:.2f}  "
            f"median moves: {row['median_moves_to_0.9']:.1f}  "
            f"mass@1/2/4/8: " + " ".join(f"{row[f'mass_after_{t}']:.2f}" for t in CHECKPOINTS),
            flush=True,
        )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(f"overall: {df['frac_pass_0.9'].mean():.2f} of games pass {THRESHOLD} "
          f"(by opponent: min {df['frac_pass_0.9'].min():.2f})")


if __name__ == "__main__":
    main()
